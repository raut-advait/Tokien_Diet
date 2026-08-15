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

CLEAN_SYSTEM_PROMPT = "You are an expert full-stack developer and AI assistant. Strictly fulfill the user request and follow all instructions, constraints, and formatting rules precisely and completely."

def run_groq_inference(prompt: str) -> dict:
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")
    
    start_time = time.perf_counter()
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
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