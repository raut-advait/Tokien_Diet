import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStore:
    def __init__(self, collection_name: str = "documents", use_mock_embeddings: bool = False):
        self.collection_name = collection_name
        self.use_mock_embeddings = use_mock_embeddings
        
        # Configure client with memory fallback
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        else:
            print("QDRANT_URL not set. Falling back to in-memory Qdrant client.")
            self.client = QdrantClient(":memory:")
            
        self.encoder = None
        if not use_mock_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except ImportError:
                print("sentence-transformers not installed. Falling back to mock embeddings.")
                self.use_mock_embeddings = True

    def _get_mock_embedding(self, text: str) -> list[float]:
        import hashlib
        import re
        vec = np.zeros(384, dtype=float)
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return vec.tolist()
            
        for word in words:
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % 384
            vec[idx] += 1.0
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def get_embedding(self, text: str) -> list[float]:
        if self.use_mock_embeddings or not self.encoder:
            return self._get_mock_embedding(text)
        return self.encoder.encode(text).tolist()

    def init_collection(self):
        # Create or recreate collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def upsert_documents(self, documents: list[dict]):
        """
        documents: list of dicts with {"id": int/str, "text": str, "metadata": dict}
        """
        points = []
        for doc in documents:
            vector = self.get_embedding(doc["text"])
            points.append(
                PointStruct(
                    id=doc["id"],
                    vector=vector,
                    payload={"text": doc["text"], "metadata": doc.get("metadata", {})}
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

    def search_dense(self, query: str, limit: int = 5) -> list[dict]:
        vector = self.get_embedding(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit
        ).points
        return [
            {
                "id": r.id,
                "text": r.payload["text"],
                "score": r.score,
                "metadata": r.payload.get("metadata", {})
            }
            for r in results
        ]
