from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_search_and_compress_endpoint():
    payload = {
        "query": "CPU vs GPU rendering processing speed and thread allocation",
        "context": (
            "CPU rendering processes rendering instructions serially. It uses few but powerful cores to handle tasks one by one. This makes it highly precise but slow for massive parallel pixel calculations. "
            "GPU rendering utilizes thousands of smaller cores to calculate rendering operations in parallel. It is highly optimized for vector calculations, texture mapping, and shading, which drastically reduces frame render times. "
            "Bananas are yellow fruits that grow on trees and contain potassium. Apples are red and delicious too."
        ),
        "mode": "fixed",
        "target_ratio": 0.5
    }
    response = client.post("/api/search-and-compress", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assert unified response structure
    assert "original_tokens" in data
    assert "compressed_tokens" in data
    assert "compression_ratio" in data
    assert "chunks" in data
    assert "full_rag" in data
    assert "compressed_rag" in data
    
    # Verify values
    assert data["original_tokens"] > 0
    assert data["compressed_tokens"] > 0
    assert data["compressed_tokens"] < data["original_tokens"]
    
    # Verify ratio matches format
    assert data["compression_ratio"].endswith("%")
    
    # Verify chunks
    assert len(data["chunks"]) > 0
    for chunk in data["chunks"]:
        assert "text" in chunk
        assert "score" in chunk
        assert "retained" in chunk
        
    # Verify RAG responses
    assert "text" in data["full_rag"]
    assert "ttft_ms" in data["full_rag"]
    assert "latency_ms" in data["full_rag"]
    assert "total_latency_ms" in data["full_rag"]
    assert "input_tokens" in data["full_rag"]
    assert "output_tokens" in data["full_rag"]
    
    assert "text" in data["compressed_rag"]
    assert "ttft_ms" in data["compressed_rag"]
    assert "latency_ms" in data["compressed_rag"]
    assert "total_latency_ms" in data["compressed_rag"]
    assert "input_tokens" in data["compressed_rag"]
    assert "output_tokens" in data["compressed_rag"]
    
    # Verify input token compression
    assert data["compressed_rag"]["input_tokens"] < data["full_rag"]["input_tokens"]
    
    # Verify content validity
    assert len(data["full_rag"]["text"]) > 0
    assert len(data["compressed_rag"]["text"]) > 0
