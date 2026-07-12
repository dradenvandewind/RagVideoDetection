"""Routes /chat et /chat/stream."""

import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..core.config import Settings, get_settings
from ..core.state import AppState, get_app_state
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    state: AppState = Depends(get_app_state),
    settings: Settings = Depends(get_settings),
):
    state.ensure_ready()
    query_engine = state.index.as_query_engine(similarity_top_k=req.top_k, streaming=False)
    response = await asyncio.to_thread(query_engine.query, req.question)
    sources = [
        {
            "text": node.get_content()[:300],
            "score": round(node.score or 0.0, 4),
            "metadata": node.metadata,
        }
        for node in (response.source_nodes or [])
    ]
    return ChatResponse(answer=str(response), sources=sources, model=settings.ollama_model)


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    state: AppState = Depends(get_app_state),
):
    state.ensure_ready()

    async def token_generator() -> AsyncGenerator[str, None]:
        query_engine = state.index.as_query_engine(similarity_top_k=req.top_k, streaming=True)
        streaming_response = await asyncio.to_thread(query_engine.query, req.question)
        for token in streaming_response.response_gen:
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")