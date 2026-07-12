"""Factory : construit le détecteur adapté à la requête (YOLO ou D-FINE).

C'est le seul point du code qui connaît l'existence des deux implémentations.
detection_router.py ne dépend que de cette factory, jamais des classes
concrètes directement — pour ajouter un 3e backend demain, on ne touche
que ce fichier.
"""

from typing import Protocol

from .detection import FrameResult
from .detection_dfine import DEFAULT_DFINE_MODEL
from .detection_schemas import DetectRequest


class StreamDetector(Protocol):
    """Interface commune à tout détecteur de flux (typage structurel)."""

    def stream_detections(self, video_url: str):  # -> AsyncGenerator[FrameResult, None]
        ...


# Le fichier yolov8n.pt par défaut de DetectRequest n'a de sens que pour
# backend="yolo" ; s'il n'a pas été explicitement changé par l'appelant et
# que backend="dfine" est demandé, on bascule sur le modèle D-FINE par défaut
# plutôt que de tenter de charger "yolov8n.pt" comme repo HuggingFace.
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