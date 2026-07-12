"""Routes d'administration de l'index (reset, etc.)."""

from fastapi import APIRouter, Depends

from ..core.config import Settings, get_settings
from ..core.state import AppState, get_app_state
from ..ingestion import VideoIngestionPipeline
from ..services.index_service import reset_index as reset_index_service

router = APIRouter(tags=["admin"])


@router.delete("/index")
async def reset_index(
    state: AppState = Depends(get_app_state),
    settings: Settings = Depends(get_settings),
):
    state.index = await reset_index_service(settings)
    state.ingestion_pipeline = VideoIngestionPipeline(index=state.index)
    return {"message": "Index reset successfully."}