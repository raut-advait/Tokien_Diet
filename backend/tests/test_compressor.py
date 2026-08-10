import os
import pytest
from app.compressor import ContextCompressor
from app.vector_store import VectorStore
from scripts.ingest_benchmark import HybridRetriever

@pytest.fixture
def compressor():
    # Use mock encoder for rapid unit tests to guarantee <50ms execution time
    c = ContextCompressor(use_mock_encoder=True)
    # Warm up to avoid cold-start overhead (such as loading tiktoken vocabulary)
    c.compress("warmup", "FastAPI is running. This is a warmup sentence.", mode="fixed", target_ratio=0.5)
    return c

def test_sentence_tokenization(compressor):
    text = "FastAPI is great. Next.js is also amazing! What about Pytest?"
    sentences = compressor.tokenize_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "FastAPI is great."
    assert sentences[1] == "Next.js is also amazing!"
    assert sentences[2] == "What about Pytest?"

def test_complexity_analyzer(compressor):
    simple_query = "What is RAG?"
    complex_query = "Compare CPU and GPU performance in multi-hop vector database workloads and indexing architectures."
    
    simple_comp = compressor.analyze_complexity(simple_query)
    complex_comp = compressor.analyze_complexity(complex_query)
    
    assert simple_comp < complex_comp
    
    # Adaptive ratio check: simple queries compress more (cut more token percentage)
    assert compressor.get_adaptive_ratio(simple_query) > compressor.get_adaptive_ratio(complex_query)

def test_compression_logic(compressor):
    query = "CPU vs GPU"
    context = (
        "GPU rendering is highly parallel and uses thousands of cores. "
        "CPU rendering processes instructions serially on few cores. "
        "Apples are delicious fruits that grow on trees. "
        "Bananas are yellow and contain potassium."
    )
    
    # Fixed ratio compression of 50%
    result = compressor.compress(query, context, mode="fixed", target_ratio=0.5)
    
    assert result["compressed_context"] != ""
    assert result["original_tokens"] > result["compressed_tokens"]
    assert result["compression_ratio"] > 0.0
    assert result["latency_ms"] < 50.0 # Strict latency check
    
    # Semantic verification: Relevant sentences (CPU / GPU) should be retained
    retained_sentences = result["compressed_context"].lower()
    assert "gpu" in retained_sentences or "cpu" in retained_sentences

def test_lost_in_the_middle_reordering(compressor):
    # Setup 4 items with distinct scores and original order indices
    kept = [
        {"sentence": "S0", "score": 0.1, "tokens": 2, "original_index": 0},
        {"sentence": "S1", "score": 0.9, "tokens": 2, "original_index": 1},
        {"sentence": "S2", "score": 0.5, "tokens": 2, "original_index": 2},
        {"sentence": "S3", "score": 0.8, "tokens": 2, "original_index": 3},
    ]
    reordered = compressor.reorder_lost_in_the_middle(kept)
    
    # Sorted order of scores: S1 (0.9), S3 (0.8), S2 (0.5), S0 (0.1)
    # top 30% (n=1) -> [S1] (goes to Head)
    # bottom 30% (n=1) -> [S0] (goes to Tail)
    # middle 40% (n=2) -> [S3, S2] sorted by original index -> [S2, S3] (goes to Core)
    # Concatenated result: Head + Core + Tail = [S1] + [S2, S3] + [S0]
    
    assert reordered[0]["sentence"] == "S1"
    assert reordered[1]["sentence"] == "S2"
    assert reordered[2]["sentence"] == "S3"
    assert reordered[3]["sentence"] == "S0"

def test_zero_empty_outputs(compressor):
    result = compressor.compress("test", "", mode="fixed", target_ratio=0.5)
    assert result["compressed_context"] == ""
    assert result["original_tokens"] == 0
    assert result["compressed_tokens"] == 0

def test_vector_store_operations():
    # Test vector store initialization and upsert
    store = VectorStore(collection_name="test_collection", use_mock_embeddings=True)
    store.init_collection()
    
    docs = [
        {"id": 1, "text": "CPU rendering is slow.", "metadata": {"test": True}},
        {"id": 2, "text": "GPU rendering is fast.", "metadata": {"test": True}},
    ]
    store.upsert_documents(docs)
    
    results = store.search_dense("GPU fast", limit=1)
    assert len(results) == 1
    assert results[0]["id"] == 2
    assert results[0]["text"] == "GPU rendering is fast."

def test_hybrid_retriever_search():
    corpus = [
        {"id": 1, "text": "CPU processes instructions serially.", "metadata": {}},
        {"id": 2, "text": "GPU calculates rendering parallel.", "metadata": {}},
        {"id": 3, "text": "Apples are delicious fruits.", "metadata": {}}
    ]
    retriever = HybridRetriever(corpus, use_mock=True)
    
    # Ingest corpus into Qdrant memory collection
    retriever.vector_store.init_collection()
    retriever.vector_store.upsert_documents(corpus)
    
    # Search sparse
    sparse_res = retriever.search_sparse("CPU")
    assert len(sparse_res) == 1
    assert sparse_res[0]["id"] == 1
    
    # Search hybrid
    hybrid_res = retriever.hybrid_search("GPU parallel", limit=1)
    assert len(hybrid_res) == 1
    assert hybrid_res[0]["id"] == 2
