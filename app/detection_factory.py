"""Factory : construit le détecteur adapté à la requête (YOLO ou D-FINE).

"""

from typing import Protocol

from .detection import FrameResult
from .detection_dfine import DEFAULT_DFINE_MODEL
from .detection_schemas import DetectRequest


class StreamDetector(Protocol):
    """Interface commune à tout détecteur de flux (typage structurel)."""

    def stream_detections(self, video_url: str):  # -> AsyncGenerator[FrameResult, None]
        ...


_YOLO_DEFAULT_MODEL_PATH = "yolov8n.pt"


def build_detector(req: DetectRequest) -> StreamDetector:
    if req.backend == "dfine":
        from .detection_dfine import DFineStreamDetector  # import différé (torch/transformers)

        model_path = (
            DEFAULT_DFINE_MODEL
            if req.model_path == _YOLO_DEFAULT_MODEL_PATH
            else req.model_path
        )
        return DFineStreamDetector(
            model_path=model_path,
            confidence=req.confidence,
            frame_skip=req.frame_skip,
            max_frames=req.max_frames,
        )

    # backend == "yolo" (défaut)
    from .detection import YOLOStreamDetector  # import différé (ultralytics)

    return YOLOStreamDetector(
        model_path=req.model_path,
        confidence=req.confidence,
        frame_skip=req.frame_skip,
        max_frames=req.max_frames,
    )