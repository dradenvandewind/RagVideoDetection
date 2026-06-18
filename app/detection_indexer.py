"""
Index YOLO detections into ChromaDB via LlamaIndex.

Each detection becomes a timestamped text Document indexable
for RAG (ex: "When was a car seen in the video?").
"""

import asyncio
import logging
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from .detection import Detection, FrameResult

logger = logging.getLogger(__name__)


class DetectionIndexer:
    """
    Receives FrameResult objects and inserts detections into the vector index
    in batches to reduce ChromaDB calls.
    """

    BATCH_SIZE = 20  # number of frames buffered before flushing

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self._pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=32)]
        )
        self._buffer: list[Document] = []
        self._total_indexed = 0

    async def add_frame_result(self, frame_result: FrameResult) -> None:
        """Add a frame's detections to the buffer, flush if needed."""
        if not frame_result.detections:
            return

        for det in frame_result.detections:
            doc = Document(
                text=det.to_text(),
                metadata={
                    "source": "yolo_detection",
                    "video_url": det.video_url,
                    "frame_id": det.frame_id,
                    "timestamp": det.timestamp,
                    "label": det.label,
                    "score": det.score,
                    "box": str(det.box),
                },
                doc_id=f"{det.video_url}__f{det.frame_id}__{det.label}",
            )
            self._buffer.append(doc)

        if len(self._buffer) >= self.BATCH_SIZE:
            await self._flush()

    async def flush_remaining(self) -> int:
        """Flush the remaining buffer at the end of the stream."""
        if self._buffer:
            await self._flush()
        return self._total_indexed

    async def _flush(self) -> None:
        batch = self._buffer[:]
        self._buffer.clear()
        nodes = await asyncio.to_thread(self._pipeline.run, documents=batch)
        await asyncio.to_thread(self.index.insert_nodes, nodes)
        self._total_indexed += len(nodes)
        logger.info(
            "📦 Flush %d docs → %d chunks (total indexed: %d)",
            len(batch),
            len(nodes),
            self._total_indexed,
        )
