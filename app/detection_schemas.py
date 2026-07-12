"""Pydantic schemas — additions for YOLO / D-FINE detection endpoints."""

from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    url: str = Field(..., description="URL YouTube (vidéo ou live)")
    backend: Literal["yolo", "dfine"] = Field(
        default="yolo",
        description="Détecteur à utiliser : 'yolo' (Ultralytics YOLOv8) ou 'dfine' (D-FINE, transformers)",
    )
    model_path: str = Field(
        default="yolov8n.pt",
        description=(
            "Poids/modèle à charger. Pour backend='yolo' : fichier .pt "
            "(yolov8n/s/m/l/x.pt). Pour backend='dfine' : repo HuggingFace "
            "(ex. 'ustc-community/dfine-nano-coco'). Si omis pour 'dfine', "
            "un modèle nano par défaut est utilisé."
        ),
    )
    confidence: float = Field(
        default=0.4,
        ge=0.05,
        le=1.0,
        description="Seuil de confiance",
    )
    frame_skip: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Traiter 1 frame sur N (1 = toutes les frames)",
    )
    max_frames: int = Field(
        default=300,
        ge=1,
        le=5000,
        description="Nombre maximum de frames à traiter",
    )


# ── Responses ──────────────────────────────────────────────────────────────────

class DetectResponse(BaseModel):
    job_id: str
    message: str
    status: str


class DetectStatusResponse(BaseModel):
    job_id: str
    status: str                        # pending | running | done | error
    frames_processed: int = 0
    total_detections: int = 0
    indexed_chunks: int = 0
    detail: str | None = None


# ── WebSocket Messages ───────────────────────────────────────────────────────

class WSDetectionMessage(BaseModel):
    """Message sent over WebSocket for each processed frame."""
    type: str = "frame"                # "frame" | "done" | "error"
    job_id: str
    frame_id: int = 0
    timestamp: float = 0.0
    detections: list[dict[str, Any]] = Field(default_factory=list)
    jpeg_b64: str = ""                 # annotated frame (JPEG base64)
    frames_processed: int = 0
    total_detections: int = 0