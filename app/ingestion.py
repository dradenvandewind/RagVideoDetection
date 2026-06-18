"""
Async video ingestion pipeline using yt-dlp.
"""

import asyncio
import logging
import os
import re
from typing import Any
import yt_dlp

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

COOKIES_PATH = os.getenv("YT_COOKIES_PATH", "/app/cookies.txt")

_YT_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    match = _YT_REGEX.search(url)
    return match.group(1) if match else None


def _fetch_transcript_with_ytdlp(url: str) -> str:
    """Download subtitles (manual or auto) via yt-dlp without Google API."""
    ydl_opts = {
        'skip_download': True,        # We don't want the video/audio, only the text
        'write_auto_html': False,
        'write_sub': True,            # Download manual subtitles
        'write_auto_sub': True,       # If no manual subtitles, use automatic captions
        'sub_langs': ['fr', 'en'],    # Prefer requested languages
        'cookiefile': COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # Retrieve available subtitles
        subtitles = info.get('subtitles') or {}
        automatic_captions = info.get('automatic_captions') or {}
        
        # Find an available language in our preference order
        chosen_lang = None
        is_auto = False
        
        for lang in ['fr', 'en']:
            if lang in subtitles:
                chosen_lang = lang
                break
            elif lang in automatic_captions:
                chosen_lang = lang
                is_auto = True
                break
                
        if not chosen_lang:
            # Fallback to the first available language
            if subtitles:
                chosen_lang = list(subtitles.keys())[0]
            elif automatic_captions:
                chosen_lang = list(automatic_captions.keys())[0]
                is_auto = True
                
        if not chosen_lang:
            raise RuntimeError(f"Aucun sous-titre trouvé pour la vidéo {url}")

        # Ask yt-dlp to download only this subtitle into memory or specific format
        # For simplicity with the yt-dlp API, we can extract the direct json3/vtt URL
        sub_info = automatic_captions[chosen_lang] if is_auto else subtitles[chosen_lang]
        
        # Find the vtt or json3 format URL
        vtt_url = next((item['url'] for item in sub_info if item.get('ext') == 'vtt'), None)
        if not vtt_url:
            vtt_url = sub_info[0]['url'] # Fallback

        # Download the subtitle content
        import requests
        response = requests.get(vtt_url)
        return _parse_srt_vtt(response.text)


def _parse_srt_vtt(raw: str) -> str:
    """Clean timestamp and style tags from SRT/VTT files."""
    # Remove timestamp lines (00:00:00.000 --> 00:00:00.000)
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", raw)
    # Remove XML/HTML tags (<c>, <b>, etc.)
    text = re.sub(r"<[^>]+>", "", text)
    # Remove line numbers (SRT)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    # Remove the WEBVTT header
    text = re.sub(r"WEBVTT.*?\n", "", text)
    return " ".join(text.split())


class VideoIngestionPipeline:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self._pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)]
        )

    async def ingest_youtube_url(
        self,
        url: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        video_id = _extract_video_id(url)
        if not video_id:
            logger.error("Invalid YouTube URL: %s", url)
            return 0

        logger.info("📥 Fetching transcript via yt-dlp for %s…", video_id)
        try:
            # On exécute la fonction bloquante yt-dlp dans un thread séparé (async)
            transcript = await asyncio.to_thread(_fetch_transcript_with_ytdlp, url)
        except Exception as exc:
            logger.error("Transcript unavailable for %s: %s", video_id, exc)
            return 0

        metadata = {
            "source": "youtube",
            "video_id": video_id,
            "url": url,
            **(extra_metadata or {}),
        }
        return await self._insert_text(transcript, metadata)

    async def ingest_text(
        self,
        raw_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if "-->" in raw_text:
            raw_text = _parse_srt_vtt(raw_text)
        return await self._insert_text(raw_text, metadata or {})

    async def _insert_text(self, text: str, metadata: dict[str, Any]) -> int:
        doc = Document(text=text, metadata=metadata)
        nodes = await asyncio.to_thread(self._pipeline.run, documents=[doc])
        logger.info("📦 %d chunks générés pour %s", len(nodes), metadata.get("source", "?"))
        await asyncio.to_thread(self.index.insert_nodes, nodes)
        logger.info("✅ %d chunks insérés dans l'index.", len(nodes))
        return len(nodes)