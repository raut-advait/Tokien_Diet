import re
import time
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

class ContextCompressor:
    def __init__(self, use_mock_encoder: bool = False):
        self.use_mock_encoder = use_mock_encoder
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model = None
        
        if not use_mock_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except ImportError:
                print("sentence-transformers not installed. Falling back to mock encoder.")
                self.use_mock_encoder = True

    def tokenize_sentences(self, text: str) -> list[str]:
        # Clean regex-based sentence splitter avoiding splits on abbreviations
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
        sentences = sentence_end.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def analyze_complexity(self, query: str) -> float:
        """
        Analyzes query complexity on a scale from 0.0 (very simple) to 1.0 (complex).
        Simple queries contain fewer words, no complex conjunctions, etc.
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Heuristics: multi-hop clues
        conjunctions = {"and", "or", "but", "because", "although", "since", "unless", "compared", "difference", "both"}
        question_words = {"how", "why", "what", "where", "which"}
        
        conjunction_count = sum(1 for w in words if w in conjunctions)
        question_count = sum(1 for w in words if w in question_words)
        
        length_factor = min(len(words) / 12.0, 1.0)
        complexity = (0.4 * length_factor) + (0.4 * min(conjunction_count / 2.0, 1.0)) + (0.2 * min(question_count, 1.0))
        return float(np.clip(complexity, 0.0, 1.0))

    def get_adaptive_ratio(self, query: str) -> float:
        """
        Calculates compression ratio based on complexity.
        Simple query (0.0 complexity) -> 80% compression (0.80 target cut)
        Complex/multi-hop query (1.0 complexity) -> 30% compression (0.30 target cut)
        """
        complexity = self.analyze_complexity(query)
        # Linear interpolation between 0.8 (simple) and 0.3 (complex)
        return float(0.8 - 0.5 * complexity)

    def _mock_cross_encoder(self, query: str, sentences: list[str]) -> list[float]:
        # Word overlap based similarity simulation
        query_words = set(query.lower().split())
        scores = []
        for s in sentences:
            s_words = set(s.lower().split())
            intersection = query_words.intersection(s_words)
            # Jaccard index-like score
            union = query_words.union(s_words)
            overlap = len(intersection) / len(union) if union else 0.0
            scores.append(overlap)
        return scores

    def compute_hybrid_scores(self, query: str, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
            
        # 1. Cross-Encoder scores
        if self.use_mock_encoder or not self.model:
            ce_scores = self._mock_cross_encoder(query, sentences)
        else:
            pairs = [[query, s] for s in sentences]
            raw_ce = self.model.predict(pairs)
            # Sigmoid normalization (typical MS-MARCO range is -12 to 4)
            ce_scores = [float(1 / (1 + np.exp(-x))) for x in raw_ce]

        # 2. BM25 scores
        tokenized_corpus = [s.lower().split() for s in sentences]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        raw_bm25 = bm25.get_scores(tokenized_query)
        
        # Max-normalize BM25
        max_bm25 = max(raw_bm25) if len(raw_bm25) > 0 else 0.0
        bm25_scores = [float(x / max_bm25) if max_bm25 > 0.0 else 0.0 for x in raw_bm25]
        
        # 3. Hybrid fusion: 0.7 * CE + 0.3 * BM25
        hybrid = [0.7 * ce + 0.3 * bm for ce, bm in zip(ce_scores, bm25_scores)]
        return hybrid

    def reorder_lost_in_the_middle(self, kept: list[dict]) -> list[dict]:
        """
        Reorders context sentences to put the highest priority nodes at the head and tail.
        Kept sentences are partitioned:
        - Top 30% scoring sentences -> Head
        - Bottom 30% scoring sentences -> Tail
        - Middle 40% scoring sentences -> Core (middle)
        Maintains original layout flow within each partitioned bucket.
        """
        if len(kept) <= 2:
            return kept # Too small to split meaningfully
            
        # Sort by score descending to rank them
        sorted_kept = sorted(kept, key=lambda x: x["score"], reverse=True)
        n = len(sorted_kept)
        
        top_n = max(1, int(0.3 * n))
        bottom_n = max(1, int(0.3 * n))
        middle_n = n - top_n - bottom_n
        
        top_group = sorted_kept[:top_n]
        middle_group = sorted_kept[top_n:top_n+middle_n]
        bottom_group = sorted_kept[top_n+middle_n:]
        
        # Re-sort each group by their original index to preserve reading order/coherence
        top_group.sort(key=lambda x: x["original_index"])
        middle_group.sort(key=lambda x: x["original_index"])
        bottom_group.sort(key=lambda x: x["original_index"])
        
        return top_group + middle_group + bottom_group

    def compress(self, query: str, context: str, mode: str = "adaptive", target_ratio: float = 0.5) -> dict:
        start_time = time.time()
        
        sentences = self.tokenize_sentences(context)
        if not sentences:
            return {
                "compressed_context": "",
                "original_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 0.0,
                "latency_ms": (time.time() - start_time) * 1000,
                "sentence_scores": [],
                "sentence_diffs": []
            }
            
        original_tokens = self.count_tokens(context)
        
        # Determine target ratio
        if mode == "adaptive":
            actual_ratio = self.get_adaptive_ratio(query)
        else:
            actual_ratio = target_ratio
            
        target_tokens = int(original_tokens * (1 - actual_ratio))
        
        # Score sentences
        scores = self.compute_hybrid_scores(query, sentences)
        
        # Map to structured sentences with original index
        scored_sentences = []
        for idx, (s, score) in enumerate(zip(sentences, scores)):
            scored_sentences.append({
                "sentence": s,
                "score": score,
                "tokens": self.count_tokens(s),
                "original_index": idx
            })
            
        # Select highest-scoring sentences that fit under the target token budget
        # Sort by score descending to prioritize selection
        selection_pool = sorted(scored_sentences, key=lambda x: x["score"], reverse=True)
        
        kept = []
        current_tokens = 0
        for s in selection_pool:
            if current_tokens + s["tokens"] <= target_tokens or not kept:
                kept.append(s)
                current_tokens += s["tokens"]
                
        # Reorder kept sentences to combat Lost-in-the-Middle degradation
        reordered_kept = self.reorder_lost_in_the_middle(kept)
        
        # Compile compressed text
        compressed_context = " ".join([s["sentence"] for s in reordered_kept])
        compressed_tokens = self.count_tokens(compressed_context)
        
        # Calculate real compression ratio achieved
        achieved_ratio = 1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0
        
        # Mark retention on the full sentence scores list
        reordered_ids = {s["original_index"] for s in reordered_kept}
        final_scores = []
        sentence_diffs = []
        for s in scored_sentences:
            is_retained = s["original_index"] in reordered_ids
            final_scores.append({
                "sentence": s["sentence"],
                "score": s["score"],
                "retained": is_retained
            })
            sentence_diffs.append({
                "text": s["sentence"],
                "retained": is_retained,
                "score": s["score"]
            })
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "compressed_context": compressed_context,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": float(np.round(achieved_ratio, 4)),
            "latency_ms": float(np.round(latency_ms, 2)),
            "sentence_scores": final_scores,
            "sentence_diffs": sentence_diffs
        }
