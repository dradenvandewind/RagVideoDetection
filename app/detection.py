"""
Real-time YOLOv8 detection pipeline from a YouTube or Dailymotion HLS stream.

Flow: yt-dlp → m3u8 URL → OpenCV frame-by-frame → YOLOv8 → results
"""

import asyncio
import base64
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from turtle import color
from typing import AsyncGenerator, Any

import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Source detection (YouTube / Dailymotion)
# ──────────────────────────────────────────────

_YT_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)
_DM_REGEX = re.compile(
    r"(?:dailymotion\.com/(?:video|embed/video)/|dai\.ly/)([A-Za-z0-9]+)"
)


def _detect_source(url: str) -> tuple[str | None, str | None]:
    """Detect the platform and video id from a URL.

    Returns (source, video_id), e.g. ("youtube", "dQw4w9WgXcQ") or
    ("dailymotion", "x9rovny"). Returns (None, None) if unrecognized.
    """
    match = _YT_REGEX.search(url)
    if match:
        return "youtube", match.group(1)

    match = _DM_REGEX.search(url)
    if match:
        video_id = match.group(1).split("_")[0]
        return "dailymotion", video_id

    return None, None


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
    color: str = "indéterminé" 

    def to_text(self) -> str:
        return (
            f"Frame {self.frame_id} at {self.timestamp:.2f}s: "
            f"detected '{self.label}' ({self.color}) with confidence {self.score:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "box": self.box,
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "video_url": self.video_url,
            "color": self.color, 
        }


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: list[Detection]
    jpeg_b64: str             # annotated frame encoded in base64 for WS


# ──────────────────────────────────────────────
# Shared helpers: annotation + JPEG encoding
# Réutilisés par YOLOStreamDetector et DFineStreamDetector (detection_dfine.py)
# ──────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of the frame. detections use
    normalized [0-1] coordinates (as stored in Detection.box)."""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{det.label} {det.score:.2f} [{det.color}]",
            (int(x1), max(int(y1) - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
           (0, 255, 0),
           1,
           cv2.LINE_AA,
      )
    return annotated


def encode_jpeg_b64(frame: np.ndarray, quality: int = 70) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ──────────────────────────────────────────────
# Cookies handling (YouTube only — Dailymotion doesn't need them)
# ──────────────────────────────────────────────

COOKIES_SOURCE = os.getenv("YT_COOKIES_PATH", "/app/cookies.txt")
_writable_cookies_path: str | None = None


def _get_writable_cookies_path() -> str | None:
    """Copy cookies to /tmp (writable) once per run."""
    global _writable_cookies_path

    if not os.path.isfile(COOKIES_SOURCE):
        return None

    if _writable_cookies_path is None or not os.path.isfile(_writable_cookies_path):
        _writable_cookies_path = "/tmp/yt_cookies_writable.txt"
        shutil.copy2(COOKIES_SOURCE, _writable_cookies_path)
        logger.info("Cookies copied to %s (writable)", _writable_cookies_path)

    return _writable_cookies_path


# ──────────────────────────────────────────────
# HLS stream URL resolver
# ──────────────────────────────────────────────

# Optional SOCKS/HTTP proxy for yt-dlp requests. Left unset by default —
# only used if YT_DLP_PROXY is actually configured in the environment, so
# this doesn't break on machines/containers without that proxy running.
YT_DLP_PROXY = os.getenv("YT_DLP_PROXY")

# Which yt-dlp "player client" to impersonate for YouTube extraction.
# "tv" tends to dodge bot-detection better for live streams; override via env.
YT_PLAYER_CLIENT = os.getenv("YT_PLAYER_CLIENT", "tv")


def _resolve_hls_url(url: str) -> str:
    """
    Resolves the HLS/m3u8 (or DASH) URL for a YouTube or Dailymotion video
    or live stream via yt-dlp. Selects the lowest resolution format to
    minimize bandwidth (we only need frames for YOLO).
    """
    source, video_id = _detect_source(url)
    if not video_id:
        raise ValueError(f"URL non reconnue (ni YouTube ni Dailymotion): {url}")

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
    }

    if YT_DLP_PROXY:
        ydl_opts["proxy"] = YT_DLP_PROXY

    if source == "youtube":
        ydl_opts["extractor_args"] = {"youtube": {"player_client": [YT_PLAYER_CLIENT]}}
        cookies_path = _get_writable_cookies_path()
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
            logger.info("Cookies YouTube chargés depuis %s", cookies_path)
        else:
            logger.warning("Pas de cookies YT (%s), risque de bot detection", COOKIES_SOURCE)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        logger.error("yt-dlp extraction failed for %s (%s): %s", video_id, source, exc)
        raise RuntimeError(f"Impossible d'extraire le flux pour {url}: {exc}") from exc

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

    stream_url = info.get("url") or info.get("manifest_url")
    if stream_url:
        return stream_url

    raise RuntimeError(f"No usable stream found for {url} ({source})")

# ──────────────────────────────────────────────
# HSV color classifier
# ──────────────────────────────────────────────

# Hue ranges in OpenCV space (H: 0-179)
_HSV_COLOR_RANGES: list[tuple[str, int, int]] = [
    ("red",     0,   10),
    ("orange",  11,  25),
    ("yellow",  26,  34),
    ("green",   35,  85),
    ("cyan",    86,  95),
    ("blue",    96,  130),
    ("purple",  131, 155),
    ("pink",    156, 169),
    ("red",     170, 179),  # red wraps around
]


def _classify_hsv_color(
    frame: np.ndarray,
    box: list[float],
    sat_min: int = 60,
    val_min: int = 40,
    val_max: int = 250,
) -> str:
    """
    Determines the dominant color in a normalized bounding box [0-1].
    Ignores pixels with low saturation (gray/white/black) using S/V thresholds.
    Returns a color label or 'undetermined' if too few valid pixels.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    xi1, yi1 = int(x1 * w), int(y1 * h)
    xi2, yi2 = int(x2 * w), int(y2 * h)

    xi1, xi2 = max(0, xi1), min(w, xi2)
    yi1, yi2 = max(0, yi1), min(h, yi2)

    if xi2 <= xi1 or yi2 <= yi1:
        return "undetermined"

    crop = frame[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return "undetermined"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Mask: keep only "colored" pixels (exclude gray, white, and black)    
    mask = (s_ch >= sat_min) & (v_ch >= val_min) & (v_ch <= val_max)

    valid_hues = h_ch[mask]
    if valid_hues.size < (crop.shape[0] * crop.shape[1] * 0.1):
        # Less than 10% of valid colored pixels → gray/white/black/too dark object
        return _classify_achromatic(v_ch)

    # Histogram of hues on valid pixels
    hist, _ = np.histogram(valid_hues, bins=180, range=(0, 180))
    dominant_hue = int(np.argmax(hist))

    for name, low, high in _HSV_COLOR_RANGES:
        if low <= dominant_hue <= high:
            return name

    return "undetermined"


def _classify_achromatic(v_ch: np.ndarray) -> str:
    """Distinguishes black, white, and gray when the saturation is too low."""    
    mean_v = float(np.mean(v_ch))
    if mean_v < 60:
        return "black"
    if mean_v > 200:
        return "white"
    return "gray"



# ──────────────────────────────────────────────
# YOLO detector
# ──────────────────────────────────────────────

class YOLOStreamDetector:
    """
    Open an HLS stream and perform YOLOv8 inference on every Nth frame.
    Works with YouTube videos/live streams and Dailymotion videos.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        frame_skip: int = 5,      # process 1 frame out of N (performance vs accuracy)
        max_frames: int = 500,    # safety limit for long videos
    ):
        logger.info("⚙️  Loading YOLOv8 from %s…", model_path)
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.frame_skip = frame_skip
        self.max_frames = max_frames

    async def stream_detections(
        self,
        video_url: str,
    ) -> AsyncGenerator[FrameResult, None]:
        """
        Async generator: resolves the stream (YouTube or Dailymotion),
        reads frames, runs YOLO, and yields a FrameResult for each
        processed frame.
        """
        source, video_id = _detect_source(video_url)
        if not video_id:
            raise ValueError(f"URL non reconnue (ni YouTube ni Dailymotion): {video_url}")

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
        """Synchronous YOLOv8 inference on a BGR numpy frame."""
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections: list[Detection] = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                norm_box = [x1 / w, y1 / h, x2 / w, y2 / h]
                color = _classify_hsv_color(frame, norm_box)

                detections.append(Detection(
                    label=label,
                    score=score,
                    box=[x1 / w, y1 / h, x2 / w, y2 / h],
                    frame_id=frame_id,
                    timestamp=timestamp,
                    video_url=video_url,
                    color=color,
                ))

        annotated = annotate_frame(frame, detections)
        jpeg_b64 = encode_jpeg_b64(annotated)

        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            jpeg_b64=jpeg_b64,
        )