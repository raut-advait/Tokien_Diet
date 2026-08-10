import os
import sys
from rank_bm25 import BM25Okapi

# Set path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.vector_store import VectorStore

SAMPLE_CORPUS = [
    {
        "id": 1,
        "text": "CPU rendering processes rendering instructions serially. It uses few but powerful cores to handle tasks one by one. This makes it highly precise but slow for massive parallel pixel calculations.",
        "metadata": {"source": "MS MARCO", "domain": "graphics"}
    },
    {
        "id": 2,
        "text": "GPU rendering utilizes thousands of smaller cores to calculate rendering operations in parallel. It is highly optimized for vector calculations, texture mapping, and shading, which drastically reduces frame render times.",
        "metadata": {"source": "MS MARCO", "domain": "graphics"}
    },
    {
        "id": 3,
        "text": "Retrieval-Augmented Generation (RAG) models fetch external document context from databases to ground LLM generations. This reduces hallucinations by providing factual, source-indexed references.",
        "metadata": {"source": "BEIR", "domain": "nlp"}
    },
    {
        "id": 4,
        "text": "Lost-in-the-middle degradation in LLMs is a phenomenon where models perform poorly when key information is located in the middle of long context prompts. The models heavily favor information at the beginning or end of context.",
        "metadata": {"source": "BEIR", "domain": "nlp"}
    },
    {
        "id": 5,
        "text": "Qdrant is a high-performance vector database optimized for similarity search. It supports cosine distance metrics and allows payload-based filtering, which is highly useful in RAG metadata constraints.",
        "metadata": {"source": "MS MARCO", "domain": "database"}
    },
    {
        "id": 6,
        "text": "Dynamic context compression trims non-essential syntax and low-scoring sentences from context prompts. It calculates hybrid scores to prune fluff while keeping critical entity tokens.",
        "metadata": {"source": "Custom", "domain": "optimization"}
    },
    {
        "id": 7,
        "text": "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. The primary function of the mitochondrion is to generate chemical energy in the form of ATP through oxidative phosphorylation. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP).",
        "metadata": {"source": "MS MARCO", "domain": "biology"}
    }
]

class HybridRetriever:
    def __init__(self, corpus: list[dict], use_mock: bool = False, vector_store: VectorStore = None):
        self.corpus = corpus
        self.vector_store = vector_store or VectorStore(use_mock_embeddings=use_mock)
        
        # Initialize sparse index
        tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_sparse(self, query: str, limit: int = 5) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Zip, sort and retrieve top limit
        ranked = sorted(zip(self.corpus, scores), key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in ranked[:limit]:
            if score > 0:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": score,
                    "metadata": doc.get("metadata", {})
                })
        return results

    def hybrid_search(self, query: str, limit: int = 5) -> list[dict]:
        # 1. Fetch dense vector results
        dense_results = self.vector_store.search_dense(query, limit=limit * 2)
        # 2. Fetch sparse BM25 results
        sparse_results = self.search_sparse(query, limit=limit * 2)
        
        # 3. Reciprocal Rank Fusion (RRF)
        # Score(d) = sum( 1 / (60 + rank) )
        rrf_scores = {}
        doc_map = {}
        
        for rank, item in enumerate(dense_results):
            doc_id = item["id"]
            doc_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank + 1))
            
        for rank, item in enumerate(sparse_results):
            doc_id = item["id"]
            doc_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank + 1))
            
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        final_results = []
        for doc_id in sorted_docs[:limit]:
            doc = doc_map[doc_id]
            final_results.append({
                "id": doc["id"],
                "text": doc["text"],
                "rrf_score": rrf_scores[doc_id],
                "metadata": doc.get("metadata", {})
            })
            
        return final_results

def run_ingestion():
    print("Initializing Qdrant collection...")
    store = VectorStore()
    store.init_collection()
    print(f"Upserting {len(SAMPLE_CORPUS)} sample passages...")
    store.upsert_documents(SAMPLE_CORPUS)
    print("Ingestion completed successfully.")

if __name__ == "__main__":
    run_ingestion()
