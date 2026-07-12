"""Routes /ingest/*."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from ..core.state import AppState, get_app_state
from ..schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])


async def _run_ingest(state: AppState, job_id: str, url: str, metadata: dict) -> None:
    try:
        n = await state.ingestion_pipeline.ingest_youtube_url(url, metadata)
        state.job_status[job_id] = {"status": "done", "chunks": n}
    except Exception as exc:  # noqa: BLE001 - on veut capturer toute erreur pipeline
        state.job_status[job_id] = {"status": "error", "detail": str(exc)}


@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_app_state),
):
    state.ensure_ready()
    job_id = str(uuid.uuid4())
    state.job_status[job_id] = {"status": "pending", "url": req.url}
    background_tasks.add_task(_run_ingest, state, job_id, req.url, req.metadata)
    return IngestResponse(
        message=f"Ingestion started for: {req.url}",
        url=req.url,
        status="pending",
    )


@router.get("/status/{job_id}")
async def ingest_status(job_id: str, state: AppState = Depends(get_app_state)):
    return state.job_status.get(job_id, {"status": "not_found"})


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
):
    if state.ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    background_tasks.add_task(
        state.ingestion_pipeline.ingest_text,
        text,
        {"source": file.filename, "type": "file_upload"},
    )
    return IngestResponse(
        message=f"File '{file.filename}' is being ingested.",
        url=file.filename,
        status="pending",
    )