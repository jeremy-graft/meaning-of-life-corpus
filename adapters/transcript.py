"""Spoken-content transcripts via youtube-transcript-api.

The official Data API v3 can't download captions for videos you don't own —
captions.download needs the video owner's OAuth and 403s otherwise. To capture
what is actually *said* in the videos (a hospice nurse on camera, a commencement
speech), we fetch the manual/auto-generated transcript through
youtube-transcript-api, which reads YouTube's internal transcript endpoint.

This is the ONE source in the pipeline outside the official Data API lane (a ToS
gray area, enabled by project decision). It uses no API key, downloads no audio,
and costs no Data API quota. It fails soft: any video without an available
transcript is simply skipped.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger("collect")

# Error classes (matched by name to stay version-robust) that mean "YouTube is
# rate-limiting this IP" — worth a backoff + retry. Everything else (transcripts
# disabled, none found, video unavailable) is permanent: give up immediately.
_BLOCK_ERRORS = {"IpBlocked", "RequestBlocked", "TooManyRequests", "YouTubeRequestFailed"}


def fetch_transcript(
    video_id: str,
    languages: Optional[list[str]] = None,
    *,
    retries: int = 2,
    backoff: float = 4.0,
) -> Optional[str]:
    """Return the full transcript text for a video, or None if unavailable.

    Never raises: the library throws a wide range of errors and we treat them as
    'no transcript here, move on'. Block/rate-limit errors get a few backoff
    retries (YouTube throttles bursts from one IP); permanent errors don't.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        log.warning("youtube-transcript-api not installed; skipping transcripts")
        return None

    langs = languages or ["en"]
    attempt = 0
    while True:
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=langs)
            text = " ".join(s.text.strip() for s in fetched if s.text and s.text.strip())
            return text or None
        except Exception as e:
            name = type(e).__name__
            if name in _BLOCK_ERRORS and attempt < retries:
                wait = backoff * (2 ** attempt)
                log.info("rate-limited on %s; backing off %.0fs (retry %d/%d)",
                         video_id, wait, attempt + 1, retries)
                time.sleep(wait)
                attempt += 1
                continue
            log.info("no transcript for %s: %s", video_id, name)
            return None
