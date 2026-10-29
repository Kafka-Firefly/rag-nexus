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
        self._corpus: list[dict] = []
        self._bm25 = None
        self._loaded = False
        self._refresh_lock = asyncio.Lock()
        self._rerank_lock = asyncio.Lock()

    @staticmethod
    def _build_index(corpus: list[dict]):
        indexed, tokenized = [], []
        for item in corpus:
            tokens = tokenize(item.get("text", ""))
            if tokens:
                indexed.append(item)
                tokenized.append(tokens)
        return indexed, BM25Okapi(tokenized) if indexed else None

    async def refresh(self) -> None:
        """Reload the lexical index after ingestion; ordinary queries use the cached snapshot."""
        async with self._refresh_lock:
            corpus = await self.store.all_chunks()
            self._corpus, self._bm25 = await asyncio.to_thread(self._build_index, corpus)
            self._loaded = True

    async def retrieve(self, query: str) -> list[dict]:
        if not self._loaded:
            await self.refresh()
        dense_results = await self.store.search(query, 10)
        merged = {item["chunk_id"]: {**item, "rrf": 0.0} for item in dense_results}
        corpus, bm25 = self._corpus, self._bm25
        query_tokens = tokenize(query)
        if corpus and bm25 and query_tokens:
            top_bm25 = sorted(zip(corpus, bm25.get_scores(query_tokens)),
                              key=lambda pair: pair[1], reverse=True)[:10]
            for rank, (item, lexical_score) in enumerate(top_bm25, 1):
                if lexical_score <= 0:
                    continue
                merged.setdefault(item["chunk_id"], {**item, "rrf": 0.0})["rrf"] += 1 / (60 + rank)
        for rank, item in enumerate(dense_results, 1):
            merged[item["chunk_id"]]["rrf"] += 1 / (60 + rank)
        candidates = sorted(merged.values(), key=lambda item: item["rrf"], reverse=True)[:20]
        if not candidates:
            return []
        async with self._rerank_lock:
            scores = await asyncio.to_thread(self.reranker.predict,
                [(query, item["text"]) for item in candidates], show_progress_bar=False)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:3]
