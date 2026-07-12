"""
Real-time D-FINE detection pipeline from a YouTube or Dailymotion HLS stream.

D-FINE (https://arxiv.org/abs/2410.13842) is a real-time DETR-style detector
available natively in `transformers` (>= 4.51) via
`DFineForObjectDetection` + `AutoImageProcessor`. It generally provides
better accuracy than YOLOv8 at comparable speed, at the cost of a slightly
larger model to load.

"""

import asyncio
import logging
from typing import AsyncGenerator

import cv2
import numpy as np
import torch
from PIL import Image

from .detection import (
    Detection,
    FrameResult,
    _detect_source,
    _resolve_hls_url,
    annotate_frame,
    encode_jpeg_b64,
)

logger = logging.getLogger(__name__)

# Most useful aliases (from lightest to most accurate):
#   ustc-community/dfine-nano-coco   ← fast, good default for streaming
#   ustc-community/dfine-small-coco
#   ustc-community/dfine-medium-coco
#   ustc-community/dfine-large-coco
#   ustc-community/dfine-xlarge-coco ← most accurate, slowest
DEFAULT_DFINE_MODEL = "ustc-community/dfine-nano-coco"


class DFineStreamDetector:
    """
    Open an HLS stream and perform D-FINE inference on every Nth frame.
    Interface identique à YOLOStreamDetector : même constructeur (à
    l'exception de model_path qui devient un repo HuggingFace), même
    stream_detections().
    """

    def __init__(
        self,
        model_path: str = DEFAULT_DFINE_MODEL,
        confidence: float = 0.4,
        frame_skip: int = 5,
        max_frames: int = 500,
        device: str | None = None,
    ):
        # Deferred import: transformers/torch are heavy dependencies,
        # no need to pay their import cost if the user only uses YOLO.
        from transformers import AutoImageProcessor, DFineForObjectDetection

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("⚙️  Loading D-FINE (%s) on %s…", model_path, self.device)

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = DFineForObjectDetection.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.confidence = confidence
        self.frame_skip = frame_skip
        self.max_frames = max_frames

    async def stream_detections(
        self,
        video_url: str,
    ) -> AsyncGenerator[FrameResult, None]:
        """Same interface as YOLOStreamDetector.stream_detections: resolves
        the HLS stream (YouTube/Dailymotion), reads frames, runs inference,
        and yields results."""
        source, video_id = _detect_source(video_url)
        if not video_id:
            raise ValueError(f"Unrecognized URL (not YouTube or Dailymotion): {video_url}")

        logger.info("🔗 Resolving %s stream for %s…", source, video_id)
        hls_url = await asyncio.to_thread(_resolve_hls_url, video_url)
        logger.info("✅ Stream resolved: %s…", hls_url[:80])

        cap = cv2.VideoCapture(hls_url)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open stream: {hls_url}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        processed = 0

        try:
            while processed < self.max_frames:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    logger.info("🏁 End of stream after %d processed frames.", processed)
                    break

                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    continue

                timestamp = frame_idx / fps
                result = await asyncio.to_thread(self._infer, frame, frame_idx, timestamp, video_url)
                processed += 1
                yield result
        finally:
            cap.release()

    def _infer(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        video_url: str,
    ) -> FrameResult:
        """Synchronous D-FINE inference on a BGR numpy frame."""
        h, w = frame.shape[:2]
        # D-FINE/transformers expect RGB (PIL), OpenCV reads BGR.
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(h, w)]),
            threshold=self.confidence,
        )[0]

        detections: list[Detection] = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append(Detection(
                label=self.model.config.id2label[int(label_id)],
                score=float(score),
                box=[x1 / w, y1 / h, x2 / w, y2 / h],
                frame_id=frame_id,
                timestamp=timestamp,
                video_url=video_url,
            ))

        annotated = annotate_frame(frame, detections)
        jpeg_b64 = encode_jpeg_b64(annotated)

        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            jpeg_b64=jpeg_b64,
        )