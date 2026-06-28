"""
RAG LlamaIndex - async FastAPI API
YouTube video ingestion + YOLO real-time detection + LLM querying
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid

import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel

from .ingestion import VideoIngestionPipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    IndexStatsResponse,
)
from .detection_router import router as detect_router, init_router as init_detect_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_index: VectorStoreIndex | None = None
_ingestion_pipeline: VideoIngestionPipeline | None = None
_job_status: dict[str, dict] = {}


def _build_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    chroma_collection = chroma_client.get_or_create_collection("video_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _ingestion_pipeline

    logger.info("🚀 Démarrage RAG LlamaIndex + YOLO detection…")

    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama-gpu:11434"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        request_timeout=300.0,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
    )

    _index = await asyncio.to_thread(_build_index)
    _ingestion_pipeline = VideoIngestionPipeline(index=_index)

    # Inject the index into the detection router
    init_detect_router(_index)

    logger.info("✅ ChromaDB index loaded + YOLO router initialized.")
    yield
    logger.info("🛑 Arrêt de l'application.")


app = FastAPI(
    title="RAG LlamaIndex - Video + YOLO",
    description="Async RAG pipeline : transcription vidéo + détection YOLO en temps réel",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage du router YOLO
app.include_router(detect_router)


# ── Routes existantes ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", index_ready=_index is not None)


@app.get("/stats", response_model=IndexStatsResponse)
async def stats():
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    collection = chroma_client.get_or_create_collection("video_rag")
    count = await asyncio.to_thread(collection.count)
    return IndexStatsResponse(total_chunks=count)


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(req: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _job_status[job_id] = {"status": "pending", "url": req.url}
    background_tasks.add_task(_run_ingest, job_id, req.url, req.metadata)
    return IngestResponse(
        message=f"Ingestion started for: {req.url}",
        url=req.url,
        status="pending",
    )


async def _run_ingest(job_id: str, url: str, metadata: dict):
    try:
        n = await _ingestion_pipeline.ingest_youtube_url(url, metadata)
        _job_status[job_id] = {"status": "done", "chunks": n}
    except Exception as exc:
        _job_status[job_id] = {"status": "error", "detail": str(exc)}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    return _job_status.get(job_id, {"status": "not_found"})


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    if _ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    background_tasks.add_task(
        _ingestion_pipeline.ingest_text,
        text,
        {"source": file.filename, "type": "file_upload"},
    )
    return IngestResponse(
        message=f"File '{file.filename}' is being ingested.",
        url=file.filename,
        status="pending",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    query_engine = _index.as_query_engine(similarity_top_k=req.top_k, streaming=False)
    response = await asyncio.to_thread(query_engine.query, req.question)
    sources = [
        {
            "text": node.get_content()[:300],
            "score": round(node.score or 0.0, 4),
            "metadata": node.metadata,
        }
        for node in (response.source_nodes or [])
    ]
    return ChatResponse(answer=str(response), sources=sources, model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    async def token_generator() -> AsyncGenerator[str, None]:
        query_engine = _index.as_query_engine(similarity_top_k=req.top_k, streaming=True)
        streaming_response = await asyncio.to_thread(query_engine.query, req.question)
        for token in streaming_response.response_gen:
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.delete("/index")
async def reset_index():
    global _index, _ingestion_pipeline
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    await asyncio.to_thread(chroma_client.delete_collection, "video_rag")
    await asyncio.to_thread(chroma_client.get_or_create_collection, "video_rag")
    _index = await asyncio.to_thread(_build_index)
    _ingestion_pipeline = VideoIngestionPipeline(index=_index)
    init_detect_router(_index)
    return {"message": "Index reset successfully."}
