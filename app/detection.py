"""
Real-time YOLOv8 detection pipeline from a YouTube HLS stream.

Flow: yt-dlp → m3u8 URL → OpenCV frame-by-frame → YOLOv8 → results
"""

import asyncio
import base64
import logging
import subprocess
import json
import os

import subprocess
import json
import logging
import time
import yt_dlp


import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any

import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    score: float
    box: list[float]          # [x1, y1, x2, y2] normalized 0-1
    frame_id: int
    timestamp: float
    video_url: str

    def to_text(self) -> str:
        """Indexable text representation for ChromaDB."""
        return (
            f"Frame {self.frame_id} at {self.timestamp:.2f}s: "
            f"detected '{self.label}' with confidence {self.score:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "box": self.box,
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "video_url": self.video_url,
        }


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: list[Detection]
    jpeg_b64: str             # annotated frame encoded in base64 for WS


# ──────────────────────────────────────────────
# HLS stream URL resolver
# ──────────────────────────────────────────────
# """
# def _resolve_hls_url(youtube_url: str) -> str:
#     """
#     Resolves the HLS/m3u8 URL for a YouTube video or live stream via yt-dlp.
#     Selects the lowest resolution format to minimize
#     bandwidth (we only need frames for YOLO).
#     """
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(youtube_url, download=False)
#         # Pour un live YouTube, l'URL du manifest HLS est dans 'url'
#         url = info.get("url") or info.get("manifest_url")
#         if not url:
#             # Fallback sur les formats disponibles
#             for fmt in info.get("formats", []):
#                 if fmt.get("protocol") in ("m3u8", "m3u8_native"):
#                     return fmt["url"]
#             raise RuntimeError(f"Aucun stream HLS trouvé pour {youtube_url}")
#         return url
# """


import logging
import time
import yt_dlp

logger = logging.getLogger(__name__)

_po_token: str | None = None
_visitor_data: str | None = None
_token_fetched_at: float = 0
_TOKEN_TTL = 3600


def _is_token_expired() -> bool:
    return (time.time() - _token_fetched_at) > _TOKEN_TTL


def refresh_po_token() -> tuple[str, str]:
    """
    Génère un PO token via le plugin yt-dlp-get-pot.
    Compatible avec l'API actuelle qui exige un argument 'client'.
    """
    opts = {"quiet": True, "skip_download": True}

    with yt_dlp.YoutubeDL(opts) as ydl:
        ie = ydl.get_info_extractor("Youtube")

        # L'API du plugin attend maintenant un client explicite
        for client in ("web", "web_creator", "mweb"):
            try:
                pot_client = getattr(ie, "_GetPOTClient", None) or getattr(ie, "_pot_client", None)
                if pot_client and hasattr(pot_client, "_fetch_po_token"):
                    visitor_data, po_token = pot_client._fetch_po_token(client)
                    if po_token and visitor_data:
                        logger.info("PO token obtenu via client '%s'", client)
                        return po_token, visitor_data
            except Exception as e:
                logger.debug("Client '%s' échoué : %s", client, e)
                continue

    raise RuntimeError(
        "Plugin PO token indisponible ou aucun client fonctionnel. "
        "Vérifiez : pip install yt-dlp-get-pot bgutil-ytdlp-pot-provider "
        "et installez deno : https://deno.land"
    )


def _get_po_token_args() -> dict:
    global _po_token, _visitor_data, _token_fetched_at

    if _po_token is None or _is_token_expired():
        try:
            _po_token, _visitor_data = refresh_po_token()
            _token_fetched_at = time.time()
        except Exception as e:
            logger.warning("PO token indisponible, fallback sans token : %s", e)
            _po_token = None
            _visitor_data = None

    if _po_token and _visitor_data:
        return {
            "youtube": {
                "po_token": [f"web+{_po_token}"],
                "visitor_data": [_visitor_data],
            }
        }
    return {}


import os

COOKIES_PATH = os.getenv("YT_COOKIES_PATH", "/app/cookies.txt")

def _resolve_hls_url(youtube_url: str) -> str:
    ydl_opts = {
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["tv_embedded", "web"],
            }
        },
    }

    # Cookies si disponibles (contourne le bot check sans token)
    if os.path.isfile(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH
        logger.info("Cookies YouTube chargés depuis %s", COOKIES_PATH)
    else:
        logger.warning("Pas de cookies YT (%s), risque de bot detection", COOKIES_PATH)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    formats = info.get("formats", [])

    hls_formats = [f for f in formats if f.get("protocol") in ("m3u8", "m3u8_native")]
    if hls_formats:
        return min(hls_formats, key=lambda f: f.get("height") or 10**9)["url"]

    dash_formats = [
        f for f in formats
        if f.get("protocol") in ("https", "http_dash_segments", "dash")
    ]
    if dash_formats:
        return min(dash_formats, key=lambda f: f.get("height") or 10**9)["url"]

    url = info.get("url")
    if url:
        return url

    raise RuntimeError("No usable stream found")
# ──────────────────────────────────────────────
# YOLO detector
# ──────────────────────────────────────────────

class YOLOStreamDetector:
    """
    Open an HLS stream and perform YOLOv8 inference on every Nth frame.
    Works with both regular YouTube videos and YouTube live streams.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        frame_skip: int = 5,      # process 1 frame out of N (performance vs accuracy)
        max_frames: int = 500,    # safety limit for long videos
    ):
        logger.info("⚙️  Chargement YOLOv8 depuis %s…", model_path)
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.frame_skip = frame_skip
        self.max_frames = max_frames

    async def stream_detections(
        self,
        youtube_url: str,
    ) -> AsyncGenerator[FrameResult, None]:
        """
        Async generator: resolves the stream, reads frames, runs YOLO,
        and yields a FrameResult for each processed frame.
        """
        logger.info("🔗 Résolution du stream HLS pour %s…", youtube_url)
        hls_url = await asyncio.to_thread(_resolve_hls_url, youtube_url)
        logger.info("✅ Stream HLS : %s…", hls_url[:80])

        cap = cv2.VideoCapture(hls_url)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir le stream : {hls_url}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        processed = 0

        try:
            while processed < self.max_frames:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    logger.info("🏁 Fin du stream après %d frames traitées.", processed)
                    break

                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    continue

                timestamp = frame_idx / fps
                result = await asyncio.to_thread(self._infer, frame, frame_idx, timestamp, youtube_url)
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
        """Synchronous YOLOv8 inference on a BGR numpy frame."""
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections: list[Detection] = []
        annotated = frame.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                detections.append(Detection(
                    label=label,
                    score=score,
                    box=[x1 / w, y1 / h, x2 / w, y2 / h],
                    frame_id=frame_id,
                    timestamp=timestamp,
                    video_url=video_url,
                ))

                # Visual annotation on the frame
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {score:.2f}",
                    (int(x1), max(int(y1) - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # JPEG encoding → base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpeg_b64 = base64.b64encode(buf.tobytes()).decode()

        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            jpeg_b64=jpeg_b64,
        )
