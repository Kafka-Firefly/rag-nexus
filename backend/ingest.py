"""Safe PDF/TXT ingestion with deterministic chunk identifiers."""
import hashlib
import io
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from vector_store import VectorStore

def parse_file(filename: str, raw: bytes) -> list[tuple[str, int | None]]:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(raw), strict=False)
            if reader.is_encrypted:
                raise HTTPException(422, "加密 PDF 暂不支持")
            return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
        if suffix == ".txt":
            return [(raw.decode("utf-8-sig"), None)]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "文件解析失败，请确认文件未损坏") from exc
    raise HTTPException(415, "仅支持 PDF 和 TXT 文件")

async def ingest_upload(file: UploadFile, store: VectorStore, max_upload_mb: int,
                        chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    filename = Path(file.filename or "upload").name
    raw = await file.read(max_upload_mb * 1024 * 1024 + 1)
    if not raw:
        raise HTTPException(400, "上传文件为空")
    if len(raw) > max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"文件超过 {max_upload_mb} MB 限制")
    document_id = hashlib.sha256(filename.encode("utf-8") + b"\0" + raw).hexdigest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        add_start_index=True, strip_whitespace=True)
    chunks = []
    for page_text, page in parse_file(filename, raw):
        for index, doc in enumerate(splitter.create_documents([page_text])):
            text = doc.page_content.strip()
            if not text:
                continue
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page}:{index}:{doc.metadata.get('start_index', 0)}"))
            chunks.append({"chunk_id": chunk_id, "document_id": document_id, "source": filename,
                "page": page, "chunk_index": index, "start_index": doc.metadata.get("start_index", 0), "text": text})
    if not chunks:
        raise HTTPException(422, "文档中没有提取到可索引文本")
    await store.upsert(chunks)
    return {"document_id": document_id, "source": filename, "chunks": len(chunks)}
