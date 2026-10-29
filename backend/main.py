"""FastAPI upload and SSE chat endpoints for the RAG service."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from ingest import ingest_upload
from retriever import HybridRetriever
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    max_upload_mb: int = 30
    cors_origins: str = "http://localhost:3000"

settings = Settings()
runtime = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store = await asyncio.to_thread(VectorStore, settings.qdrant_url,
            settings.qdrant_collection, settings.embedding_model)
        runtime["store"] = store
        runtime["retriever"] = await asyncio.to_thread(HybridRetriever, store, settings.reranker_model)
        runtime["llm"] = AsyncOpenAI(api_key=settings.llm_api_key or "ollama",
            base_url=settings.llm_base_url)
        logger.info("RAG services initialized")
    except Exception:
        logger.exception("Service initialization failed")
        raise
    yield
    runtime.clear()

app = FastAPI(title="RAG Nexus API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",")],
    allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok", "initialized": bool(runtime)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        return await ingest_upload(file, runtime["store"], settings.max_upload_mb)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(500, "文档索引失败，请检查服务日志") from exc

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, "请求体必须是有效 JSON") from exc
    try:
        question = str(body.get("question", "")).strip()
        if not question or len(question) > 8000:
            raise HTTPException(422, "question 必须为 1 至 8000 字符")
        sources = await runtime["retriever"].retrieve(question)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(503, "检索服务暂不可用") from exc
    citations = [{"index": i, "source": item["source"], "chunk_id": item["chunk_id"],
                  "page": item.get("page"), "text": item["text"], "score": item["rerank_score"]}
                 for i, item in enumerate(sources, 1)]
    context = "\n\n".join(f"[{item['index']}] source={item['source']} chunk_id={item['chunk_id']}\n{item['text']}"
                          for item in citations)
    system = ("你是严谨的知识库助手。只能根据给定 Context 回答，不得使用外部知识。"
              "若 Context 不足以回答，必须明确回复：根据现有文档无法回答。"
              "每项事实使用 [1] 这样的编号引用，只引用确实支持该事实的来源。不要伪造引用。")

    async def event_stream():
        try:
            yield f"event: citations\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"
            stream = await runtime["llm"].chat.completions.create(model=settings.llm_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": f"问题：{question}\n\nContext：\n{context}"}],
                temperature=0.1, stream=True)
            async for part in stream:
                if await request.is_disconnected():
                    break
                token = part.choices[0].delta.content
                if token:
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception:
            logger.exception("LLM stream failed")
            yield f"event: error\ndata: {json.dumps({'detail': '回答生成失败，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
