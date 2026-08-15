import math
import re
from typing import List, Dict, Any

CRITICAL_STRUCTURAL_CUES = [
    "objective", "requirements", "tech stack", "folder structure", 
    "index.html", "style.css", "script.js", "output format", 
    "required sections", "functional requirements", "hero", "about", "skills", "projects", "contact"
]

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

def score_and_prune_chunks(chunks: List[str], query: str, ratio: float = 0.5) -> Dict[str, Any]:
    if not chunks:
        return {"retained_text": "", "chunks": []}
    
    query_tokens = set(tokenize(query))
    scored_chunks = []
    
    for chunk in chunks:
        chunk_tokens = tokenize(chunk)
        if not chunk_tokens:
            scored_chunks.append({"text": chunk, "score": 0.05, "retained": False})
            continue
            
        # 1. Term overlap ratio
        overlap = len(query_tokens.intersection(set(chunk_tokens)))
        overlap_score = overlap / max(1, len(chunk_tokens) ** 0.5)
        
        # 2. Structural keyword bonus
        structural_bonus = 0.0
        lower_chunk = chunk.lower()
        for cue in CRITICAL_STRUCTURAL_CUES:
            if cue in lower_chunk:
                structural_bonus += 0.15
                break
                
        # 3. Positional bias (maintain document flow & instruction heads)
        final_score = round(min(0.99, max(0.10, overlap_score + structural_bonus + 0.1)), 3)
        scored_chunks.append({"text": chunk, "score": final_score, "retained": False})
    
    # Sort and retain top-K percentile
    k = max(1, int(len(scored_chunks) * ratio))
    sorted_indices = sorted(range(len(scored_chunks)), key=lambda i: scored_chunks[i]["score"], reverse=True)[:k]
    
    retained_indices = set(sorted_indices)
    retained_texts = []
    
    for idx, item in enumerate(scored_chunks):
        if idx in retained_indices:
            item["retained"] = True
            retained_texts.append(item["text"])
            
    return {
        "retained_text": "\n\n".join(retained_texts),
        "chunks": scored_chunks
    }