"""
Router FastAPI — détection YOLO en temps réel.

Endpoints :
  POST /detect/url          → lance le job de détection (background)
  GET  /detect/status/{id}  → état du job
  WS   /detect/stream/{id}  → stream des frames annotées + détections
"""

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .detection import YOLOStreamDetector
from .detection_indexer import DetectionIndexer
from .detection_schemas import (
    DetectRequest,
    DetectResponse,
    DetectStatusResponse,
    WSDetectionMessage,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])

# ──────────────────────────────────────────────
# Shared state (injected from main.py)
# ──────────────────────────────────────────────

_index_ref: Any = None          # VectorStoreIndex
_detect_jobs: dict[str, dict] = {}
_ws_queues: dict[str, asyncio.Queue] = {}   # job_id → Queue de WSDetectionMessage


def init_router(index: Any) -> None:
    """Called at startup from main.py to inject the index."""
    global _index_ref
    _index_ref = index


# ──────────────────────────────────────────────
# POST /detect/url
# ──────────────────────────────────────────────

@router.post("/url", response_model=DetectResponse)
async def start_detection(req: DetectRequest):
    """Start a YOLO detection job in the background."""
    if _index_ref is None:
        raise HTTPException(status_code=503, detail="Index non initialisé")

    job_id = str(uuid.uuid4())
    _detect_jobs[job_id] = {
        "status": "pending",
        "url": req.url,
        "frames_processed": 0,
        "total_detections": 0,
        "indexed_chunks": 0,
    }
    _ws_queues[job_id] = asyncio.Queue(maxsize=100)

    asyncio.create_task(_run_detection_job(job_id, req))

    return DetectResponse(
        job_id=job_id,
        message=f"Job de détection lancé pour : {req.url}",
        status="pending",
    )


# ──────────────────────────────────────────────
# GET /detect/status/{job_id}
# ──────────────────────────────────────────────

@router.get("/status/{job_id}", response_model=DetectStatusResponse)
async def detection_status(job_id: str):
    job = _detect_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return DetectStatusResponse(job_id=job_id, **job)


# ──────────────────────────────────────────────
# WS /detect/stream/{job_id}
# ──────────────────────────────────────────────

@router.websocket("/stream/{job_id}")
async def detection_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket: sends annotated frames and detections in real time.
    The client receives JSON messages matching WSDetectionMessage.
    """
    if job_id not in _detect_jobs:
        await websocket.close(code=1008, reason="Job introuvable")
        return

    await websocket.accept()
    queue = _ws_queues.get(job_id)
    if not queue:
        await websocket.close(code=1011, reason="Queue introuvable")
        return

    logger.info("🔌 Client WebSocket connecté pour job %s", job_id)
    try:
        while True:
            try:
                msg: WSDetectionMessage = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Keepalive ping if no frames
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            await websocket.send_text(msg.model_dump_json())

            if msg.type in ("done", "error"):
                break

    except WebSocketDisconnect:
        logger.info("🔌 Client WebSocket déconnecté pour job %s", job_id)
    finally:
        # Queue cleanup
        _ws_queues.pop(job_id, None)


# ──────────────────────────────────────────────
# Background task
# ──────────────────────────────────────────────

async def _run_detection_job(job_id: str, req: DetectRequest) -> None:
    """
    Background task: HLS stream → YOLO → ChromaDB indexing + WS broadcast.
    """
    job = _detect_jobs[job_id]
    queue = _ws_queues.get(job_id)

    job["status"] = "running"
    logger.info("▶️  Job %s démarré pour %s", job_id, req.url)

    detector = YOLOStreamDetector(
        model_path=req.model_path,
        confidence=req.confidence,
        frame_skip=req.frame_skip,
        max_frames=req.max_frames,
    )
    indexer = DetectionIndexer(index=_index_ref)

    total_detections = 0
    frames_processed = 0

    try:
        async for frame_result in detector.stream_detections(req.url):
            frames_processed += 1
            n_det = len(frame_result.detections)
            total_detections += n_det

            # Update status
            job["frames_processed"] = frames_processed
            job["total_detections"] = total_detections

            # Async indexing into ChromaDB
            await indexer.add_frame_result(frame_result)

            # WebSocket broadcast (non-blocking: drop this frame if the queue is full)
            if queue:
                msg = WSDetectionMessage(
                    type="frame",
                    job_id=job_id,
                    frame_id=frame_result.frame_id,
                    timestamp=frame_result.timestamp,
                    detections=[d.to_dict() for d in frame_result.detections],
                    jpeg_b64=frame_result.jpeg_b64,
                    frames_processed=frames_processed,
                    total_detections=total_detections,
                )
                try:
                    queue.put_nowait(msg)
                except asyncio.QueueFull:
                    pass  # Client too slow, dropping this frame

            if frames_processed % 20 == 0:
                logger.info(
                    "📊 Job %s : %d frames, %d détections",
                    job_id,
                    frames_processed,
                    total_detections,
                )

        # End of stream: flush remaining chunks
        indexed = await indexer.flush_remaining()
        job["indexed_chunks"] = indexed
        job["status"] = "done"
        logger.info("✅ Job %s terminé — %d chunks indexés", job_id, indexed)

        # End signal to WS clients
        if queue:
            done_msg = WSDetectionMessage(
                type="done",
                job_id=job_id,
                frames_processed=frames_processed,
                total_detections=total_detections,
            )
            await queue.put(done_msg)

    except Exception as exc:
        logger.exception("❌ Erreur dans le job %s : %s", job_id, exc)
        job["status"] = "error"
        job["detail"] = str(exc)
        if queue:
            err_msg = WSDetectionMessage(type="error", job_id=job_id)
            try:
                queue.put_nowait(err_msg)
            except asyncio.QueueFull:
                pass
