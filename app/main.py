"""
RAG LlamaIndex - async FastAPI API
YouTube video ingestion + YOLO real-time detection + LLM querying


"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings as LlamaSettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from .core.config import get_settings
from .core.state import AppState
from .detection_router import router as detect_router
from .ingestion import VideoIngestionPipeline
from .routers import admin, chat, health, ingest
from .services.index_service import build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state = AppState()
    app.state.app_state = state

    logger.info("🚀 Démarrage RAG LlamaIndex + YOLO detection…")

    LlamaSettings.llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        request_timeout=settings.llm_request_timeout,
    )
    LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
    LlamaSettings.node_parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    state.index = await build_index(settings)
    state.ingestion_pipeline = VideoIngestionPipeline(index=state.index)

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

app.include_router(detect_router)
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(admin.router)