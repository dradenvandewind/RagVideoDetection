
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException, Request, WebSocket
from llama_index.core import VectorStoreIndex

from ..ingestion import VideoIngestionPipeline


@dataclass
class AppState:
    index: Optional[VectorStoreIndex] = None
    ingestion_pipeline: Optional[VideoIngestionPipeline] = None
    job_status: dict[str, dict] = field(default_factory=dict)

    detect_jobs: dict[str, dict] = field(default_factory=dict)
    ws_queues: dict[str, "asyncio.Queue"] = field(default_factory=dict)

    def ensure_ready(self) -> None:
        """Lève une erreur explicite si l'app n'a pas fini son lifespan startup."""
        if self.index is None:
            raise HTTPException(status_code=503, detail="Index not initialized")


def get_app_state(request: Request) -> AppState:
    """Dépendance FastAPI pour les routes HTTP classiques."""
    return request.app.state.app_state


def get_app_state_ws(websocket: WebSocket) -> AppState:
    """Équivalent de get_app_state pour les routes WebSocket.

    FastAPI ne partage pas Request et WebSocket dans Depends() : il faut
    une dépendance dédiée qui type-hint WebSocket.
    """
    return websocket.app.state.app_state