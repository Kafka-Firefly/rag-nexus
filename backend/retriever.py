"""Hybrid dense + BM25 retrieval followed by BGE cross-encoder reranking."""
import asyncio
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from vector_store import VectorStore

def tokenize(text: str) -> list[str]:
    tokens = []
    for part in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                tokens.append(part)
        else:
            tokens.append(part)
    return tokens

class HybridRetriever:
    def __init__(self, store: VectorStore, reranker_model: str):
        self.store, self.reranker = store, CrossEncoder(reranker_model)

    async def retrieve(self, query: str) -> list[dict]:
        dense_results, corpus = await asyncio.gather(self.store.search(query, 10), self.store.all_chunks())
        merged = {item["chunk_id"]: {**item, "rrf": 0.0} for item in dense_results}
        if corpus:
            bm25 = BM25Okapi([tokenize(item["text"]) for item in corpus])
            top_bm25 = sorted(zip(corpus, bm25.get_scores(tokenize(query))),
                              key=lambda pair: pair[1], reverse=True)[:10]
            for rank, (item, _) in enumerate(top_bm25, 1):
                merged.setdefault(item["chunk_id"], {**item, "rrf": 0.0})["rrf"] += 1 / (60 + rank)
        for rank, item in enumerate(dense_results, 1):
            merged[item["chunk_id"]]["rrf"] += 1 / (60 + rank)
        candidates = sorted(merged.values(), key=lambda item: item["rrf"], reverse=True)[:20]
        if not candidates:
            return []
        scores = await asyncio.to_thread(self.reranker.predict,
            [(query, item["text"]) for item in candidates], show_progress_bar=False)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:3]
