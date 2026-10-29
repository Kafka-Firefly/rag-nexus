"""Qdrant storage with local SentenceTransformer embeddings."""
import asyncio
from typing import Any
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, url: str, collection: str, embedding_model: str):
        self.collection = collection
        self.client = QdrantClient(url=url, timeout=30)
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection_name=collection, vectors_config={
                "dense": models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)})

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    async def upsert(self, chunks: list[dict[str, Any]]) -> None:
        vectors = await asyncio.to_thread(self._encode, [item["text"] for item in chunks])
        points = [models.PointStruct(id=item["chunk_id"], vector={"dense": vec}, payload=item)
                  for item, vec in zip(chunks, vectors)]
        await asyncio.to_thread(self.client.upsert, collection_name=self.collection, points=points, wait=True)

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        vector = (await asyncio.to_thread(self._encode, [query]))[0]
        result = await asyncio.to_thread(self.client.query_points, collection_name=self.collection,
            query=vector, using="dense", limit=limit, with_payload=True)
        return [{**point.payload, "vector_score": float(point.score)} for point in result.points]

    async def all_chunks(self) -> list[dict[str, Any]]:
        chunks, offset = [], None
        while True:
            points, offset = await asyncio.to_thread(self.client.scroll, collection_name=self.collection,
                limit=1000, offset=offset, with_payload=True, with_vectors=False)
            chunks.extend(point.payload for point in points)
            if offset is None:
                return chunks
