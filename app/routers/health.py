"""Routes /health et /stats."""

from fastapi import APIRouter, Depends

from ..core.config import Settings, get_settings
from ..core.state import AppState, get_app_state
from ..schemas import HealthResponse, IndexStatsResponse
from ..services.index_service import count_chunks

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_app_state)):
    return HealthResponse(status="ok", index_ready=state.index is not None)


@router.get("/stats", response_model=IndexStatsResponse)
async def stats(
    state: AppState = Depends(get_app_state),
    settings: Settings = Depends(get_settings),
):
    state.ensure_ready()
    total = await count_chunks(settings)
    return IndexStatsResponse(total_chunks=total)