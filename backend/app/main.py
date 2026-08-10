import os
import re
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.compressor import ContextCompressor
from app.vector_store import VectorStore
from scripts.ingest_benchmark import HybridRetriever, SAMPLE_CORPUS

app = FastAPI(title="Token-Diet Dynamic Context Compressor API")

# Setup configurations
use_mock = os.getenv("USE_MOCK_ENCODER", "false").lower() == "true"
compressor = ContextCompressor(use_mock_encoder=use_mock)
vector_store = VectorStore(use_mock_embeddings=use_mock)

# Initialize collection and seed in-memory
try:
    vector_store.init_collection()
    vector_store.upsert_documents(SAMPLE_CORPUS)
except Exception as e:
    print(f"Collection setup error: {e}")
    
retriever = HybridRetriever(SAMPLE_CORPUS, use_mock=use_mock, vector_store=vector_store)

# Pydantic Request/Response models
class CompressionRequest(BaseModel):
    query: str = Field(..., example="What is the difference between CPU and GPU rendering?")
    context: str | None = Field(None, example="CPU rendering is serial. GPU rendering is highly parallel...")
    mode: str | None = Field("adaptive", pattern="^(adaptive|fixed)$", example="adaptive")
    target_ratio: float = Field(0.5, ge=0.0, le=1.0, example=0.5)
    top_k: int | None = Field(None, ge=1, le=10, example=3)
    dynamic_compression: bool | None = Field(None, example=True)

class SentenceScore(BaseModel):
    sentence: str
    score: float
    retained: bool

class SentenceDiff(BaseModel):
    text: str
    retained: bool
    score: float

class CompressionResponse(BaseModel):
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    latency_ms: float
    sentence_scores: list[SentenceScore]
    sentence_diffs: list[SentenceDiff]

class SearchAndCompressRequest(BaseModel):
    query: str = Field(..., example="CPU vs GPU rendering processing")
    limit: int | None = Field(3, ge=1, le=10, example=3)
    top_k: int | None = Field(None, ge=1, le=10, example=3)
    context: str | None = Field(None, example="Optional custom context text")
    mode: str | None = Field("fixed", pattern="^(adaptive|fixed)$", example="fixed")
    target_ratio: float = Field(0.5, ge=0.0, le=1.0, example=0.5)
    dynamic_compression: bool | None = Field(None, example=True)

class LLMMetrics(BaseModel):
    ttft_ms: float
    total_latency_ms: float
    output: str
    tokens: int

class SearchAndCompressResponse(BaseModel):
    query: str
    full_context: str
    compressed_context: str
    compression_ratio: float
    compression_latency_ms: float
    retrieved_chunks: list[dict]
    full_context_llm: LLMMetrics
    compressed_context_llm: LLMMetrics
    sentence_diffs: list[SentenceDiff]

def synthesize_concise_answer(query: str, context: str) -> str:
    """
    Synthesizes a 1-2 sentence concise answer based strictly on the query matching
    the most relevant sentences within the retrieved/compressed context.
    """
    # Clean query words and strip punctuation
    q_words = set(re.findall(r'\b\w+\b', query.lower()))
    stopwords = {"what", "is", "are", "the", "difference", "between", "explain", "how", "and", "or", "in", "of", "to", "for", "a", "does", "impact"}
    content_words = q_words - stopwords
    if not content_words:
        content_words = q_words
        
    # Split context into sentences
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_end.split(context.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return "No relevant facts found in the retrieved context to answer the query."
        
    scored_sentences = []
    for s in sentences:
        s_words = set(re.findall(r'\b\w+\b', s.lower()))
        match_score = len(content_words.intersection(s_words))
        scored_sentences.append((s, match_score))
        
    # Sort by match score descending
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve top 1 or 2 matching sentences
    best_sentences = []
    for s, score in scored_sentences:
        if score > 0 or not best_sentences:
            best_sentences.append(s)
        if len(best_sentences) >= 2:
            break
            
    # Re-order based on original appearance in context to keep narrative flow
    ordered_sentences = []
    for s in sentences:
        if s in best_sentences:
            ordered_sentences.append(s)
            best_sentences.remove(s)
            
    return " ".join(ordered_sentences)

def query_groq_api(query: str, context: str) -> dict:
    """
    Calls Groq API to run Llama-3-8B completions.
    If GROQ_API_KEY is missing, runs a simulated response mirroring expected latency/TTFT.
    """
    api_key = os.getenv("GROQ_API_KEY")
    prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer concisely based on context:"
    
    if not api_key:
        # Generate a concise, fact-grounded synthesized RAG answer
        synthesized_output = synthesize_concise_answer(query, context)
        
        # Calculate simulated latency parameters
        char_count = len(prompt)
        simulated_ttft = float(30.0 + char_count * 0.05)
        simulated_generation = 120.0
        
        time.sleep((simulated_ttft + simulated_generation) / 1000.0) # sleep to mimic delay
        
        return {
            "ttft_ms": round(simulated_ttft, 2),
            "total_latency_ms": round(simulated_ttft + simulated_generation, 2),
            "output": synthesized_output,
            "tokens": len(synthesized_output) // 4
        }
        
    start_time = time.time()
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10.0)
        response.raise_for_status()
        resp_json = response.json()
        
        total_time = (time.time() - start_time) * 1000
        output_text = resp_json["choices"][0]["message"]["content"]
        
        # Estimate TTFT as time to headers (approx 70% of non-streamed time for Groq)
        ttft = total_time * 0.7
        
        return {
            "ttft_ms": round(ttft, 2),
            "total_latency_ms": round(total_time, 2),
            "output": output_text,
            "tokens": resp_json.get("usage", {}).get("total_tokens", 0)
        }
    except Exception as e:
        return {
            "ttft_ms": 150.0,
            "total_latency_ms": 400.0,
            "output": f"Error calling Groq API: {str(e)}",
            "tokens": 0
        }

@app.get("/")
async def root():
    return {"message": "Token-Diet API running"}

@app.post("/api/compress", response_model=CompressionResponse)
async def compress_route(req: CompressionRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        # Sanitize incoming query quotes
        sanitized_query = req.query.strip('"').strip("'").strip()
        
        # Determine context source (custom or fetched from vector store)
        if req.context and req.context.strip():
            full_context = req.context
        else:
            limit_val = req.top_k if req.top_k is not None else 3
            chunks = retriever.hybrid_search(sanitized_query, limit=limit_val)
            full_context = " ".join([c["text"] for c in chunks])
            
        # Determine resolved mode
        mode_val = "adaptive" if req.dynamic_compression is True else (req.mode if req.mode is not None else "adaptive")
        
        result = compressor.compress(
            query=sanitized_query,
            context=full_context,
            mode=mode_val,
            target_ratio=req.target_ratio
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-and-compress", response_model=SearchAndCompressResponse)
async def search_and_compress_route(req: SearchAndCompressRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        # Sanitize incoming query quotes
        sanitized_query = req.query.strip('"').strip("'").strip()
        
        # 1. Fetch chunks (User context override vs dynamic hybrid search)
        if req.context and req.context.strip():
            full_context = req.context
            chunks = [{"id": 0, "text": req.context, "metadata": {"source": "User Context Override"}}]
        else:
            limit_val = req.top_k if req.top_k is not None else (req.limit if req.limit is not None else 3)
            chunks = retriever.hybrid_search(sanitized_query, limit=limit_val)
            full_context = " ".join([c["text"] for c in chunks])
            
        # 2. Map resolved compression parameters
        mode_val = "adaptive" if req.dynamic_compression is True else (req.mode if req.mode is not None else "fixed")
        
        # 3. Compress context using ContextCompressor
        comp_start = time.time()
        comp_result = compressor.compress(
            query=sanitized_query,
            context=full_context,
            mode=mode_val,
            target_ratio=req.target_ratio
        )
        comp_latency_ms = (time.time() - comp_start) * 1000
        
        compressed_context = comp_result["compressed_context"]
        
        # 4. Invoke LLM dynamically
        full_metrics = query_groq_api(sanitized_query, full_context)
        comp_metrics = query_groq_api(sanitized_query, compressed_context)
        
        return {
            "query": sanitized_query,
            "full_context": full_context,
            "compressed_context": compressed_context,
            "compression_ratio": comp_result["compression_ratio"],
            "compression_latency_ms": round(comp_latency_ms, 2),
            "retrieved_chunks": chunks,
            "full_context_llm": full_metrics,
            "compressed_context_llm": comp_metrics,
            "sentence_diffs": comp_result["sentence_diffs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
