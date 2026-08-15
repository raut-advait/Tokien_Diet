import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq

# 1. Initialize FastAPI App
app = FastAPI(title="Token-Diet Dynamic Context Compressor API")

# 2. Configure CORS Middleware (Placed before route handlers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize Groq Client & Model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Active Groq Models:
GROQ_MODEL = "	llama3-8b-8192"  # High quality
# GROQ_MODEL = "llama-3.1-8b-instant"  # Ultra fast

# 4. Request / Response Schemas
class SearchAndCompressRequest(BaseModel):
    query: str
    context: Optional[str] = None
    ratio: Optional[float] = None

# 5. Route Handlers
@app.get("/")
async def root():
    return {"status": "online", "service": "Token-Diet API"}

@app.post("/api/search-and-compress")
async def search_and_compress(req: SearchAndCompressRequest):
    if not GROQ_API_KEY or not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")

    # Execute compressor logic and LLM calls here...
    return {"status": "success", "model": GROQ_MODEL}