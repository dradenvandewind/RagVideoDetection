"""
License plate detection & OCR module.

Flow: vehicle crop (from YOLO detection) → YOLO plate detector → EasyOCR
Includes a lightweight IoU-based tracker to avoid re-running OCR
on the same vehicle across consecutive frames.
"""

import logging

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

logger = logging.getLogger(__name__)

VEHICLE_LABELS = {"car", "truck", "motorcycle", "bus"}


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over Union entre deux box normalisées [x1, y1, x2, y2]."""
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateTracker:
    """
    Hide license plate readings based on the vehicle's approximate position,
    to avoid running the OCR again on the same vehicle every frame.
    """

    def __init__(
        self,
        model_path: str = "models/license_plate_detector.pt",
        gpu: bool = True,
        iou_threshold: float = 0.5,
        ttl_frames: int = 30,
        plate_confidence: float = 0.5,
    ):
        logger.info("⚙️ Chargement du modèle de détection de plaques depuis %s…", model_path)
        self.plate_model = YOLO(model_path)
        self.ocr_reader = easyocr.Reader(["en"], gpu=gpu)

        self.iou_threshold = iou_threshold
        self.ttl_frames = ttl_frames
        self.plate_confidence = plate_confidence
        self._cache: list[dict] = []  # [{box, plate, last_seen}]

    def get_or_compute(
        self,
        frame: np.ndarray,
        box: list[float],
        label: str,
        frame_id: int,
    ) -> str | None:
        if label not in VEHICLE_LABELS:
            return None

        for entry in self._cache:
            if _iou(entry["box"], box) > self.iou_threshold:
                entry["box"] = box
                entry["last_seen"] = frame_id
                return entry["plate"]

        plate = self._extract_plate(frame, box)
        self._cache.append({"box": box, "plate": plate, "last_seen": frame_id})

        self._cache = [
            e for e in self._cache if frame_id - e["last_seen"] <= self.ttl_frames
        ]
        return plate

    def _extract_plate(self, frame: np.ndarray, box: list[float]) -> str | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        xi1, yi1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        xi2, yi2 = min(w, int(x2 * w)), min(h, int(y2 * h))
        vehicle_crop = frame[yi1:yi2, xi1:xi2]
        if vehicle_crop.size == 0:
            return None

        plate_results = self.plate_model(
            vehicle_crop, conf=self.plate_confidence, verbose=False
        )
        for r in plate_results:
            for pbox in r.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue
                ocr_result = self.ocr_reader.readtext(plate_crop, detail=0)
                if ocr_result:
                    return "".join(ocr_result).upper().replace(" ", "")
        return None