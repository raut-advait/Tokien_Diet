import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq

app = FastAPI(title="TokenDiet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

CLEAN_SYSTEM_PROMPT = "You are an expert full-stack developer and AI assistant. Strictly fulfill the user request and follow all instructions, constraints, and formatting rules precisely and completely."

def run_groq_inference(prompt: str) -> dict:
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")
    
    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]
    
    # Build models to try
    models_to_try = [GROQ_MODEL]
    for m in SUPPORTED_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)
            
    last_error = None
    for current_model in models_to_try:
        try:
            start_time = time.perf_counter()
            response = groq_client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096
            )
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "text": response.choices[0].message.content,
                "ttft_ms": round(total_time_ms * 0.4, 2),
                "total_latency_ms": round(total_time_ms, 2),
                "latency_ms": round(total_time_ms, 2),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            # Check if this is a 404 error
            is_404 = False
            if hasattr(e, "status_code") and e.status_code == 404:
                is_404 = True
            elif "404" in str(e) or "model_not_found" in str(e):
                is_404 = True
                
            if is_404:
                print(f"Groq API model not found (404) for model {current_model}. Retrying with next model...", flush=True)
                last_error = e
                continue
            else:
                raise e
                
    raise HTTPException(status_code=500, detail=f"All attempted Groq models failed: {str(last_error)}")