"""YouTube Data API v3 adapter.

Per genre-seed: search videos -> pull video metadata -> pull top-level + reply
comments -> (optional) pull the spoken transcript. Comments are the richest
seam; transcripts add what is *said* in the video. Everything is pseudonymized
on ingest (db.pseudonymize).

Quota: the default project quota is ~10,000 units/day. We track spend per call
and fail SOFT when the budget is exhausted (raise QuotaExhausted, caught by the
caller) rather than crashing mid-run. Approximate unit costs:
    search.list          100
    videos.list            1
    commentThreads.list    1
Transcripts (adapters/transcript.py) do NOT use the Data API, so they cost zero
quota — but they are the one source outside the official API lane.

Reads YOUTUBE_API_KEY from the environment.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from adapters import transcript as transcript_mod
from db import db as dbmod

log = logging.getLogger("collect")

# Approximate quota cost per API method (units).
COST_SEARCH = 100
COST_VIDEOS = 1
COST_COMMENTS = 1


class QuotaExhausted(Exception):
    """Raised when the next call would exceed the configured quota budget."""


class YouTubeCollector:
    def __init__(self, api_key: Optional[str] = None, quota_budget: int = 9500):
        key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not key:
            raise RuntimeError("YOUTUBE_API_KEY is not set in the environment.")
        self.yt = build("youtube", "v3", developerKey=key, cache_discovery=False)
        self.quota_budget = quota_budget
        self.spent = 0

    # -- quota -------------------------------------------------------------

    def _charge(self, cost: int, label: str) -> None:
        if self.spent + cost > self.quota_budget:
            raise QuotaExhausted(
                f"quota budget {self.quota_budget} would be exceeded by {label} "
                f"(spent {self.spent}, need {cost})"
            )
        self.spent += cost
        log.debug("quota: +%d for %s (total %d/%d)", cost, label, self.spent, self.quota_budget)

    # -- API calls ---------------------------------------------------------

    def search_videos(self, query: str, max_results: int, order: str, lang: Optional[str]) -> list[dict]:
        """Return up to max_results search results (video stubs)."""
        out: list[dict] = []
        page_token = None
        while len(out) < max_results:
            self._charge(COST_SEARCH, "search.list")
            req = self.yt.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=min(50, max_results - len(out)),
                order=order,
                relevanceLanguage=lang or None,
                pageToken=page_token,
            )
            resp = req.execute()
            out.extend(resp.get("items", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return out[:max_results]

    def get_video(self, video_id: str) -> Optional[dict]:
        self._charge(COST_VIDEOS, "videos.list")
        resp = self.yt.videos().list(part="snippet,statistics", id=video_id).execute()
        items = resp.get("items", [])
        return items[0] if items else None

    def iter_comments(self, video_id: str, max_comments: int, lang: Optional[str]) -> Iterable[dict]:
        """Yield comment dicts (top-level + inline replies), each with
        text/author/published_at/url-anchor, up to max_comments."""
        yielded = 0
        page_token = None
        while yielded < max_comments:
            self._charge(COST_COMMENTS, "commentThreads.list")
            try:
                resp = self.yt.commentThreads().list(
                    videoId=video_id,
                    part="snippet,replies",
                    maxResults=min(100, max_comments - yielded),
                    order="relevance",
                    textFormat="plainText",
                    pageToken=page_token,
                ).execute()
            except HttpError as e:
                # Comments disabled, or video not found — skip this video's comments.
                log.info("comments unavailable for %s: %s", video_id, _http_reason(e))
                return
            for thread in resp.get("items", []):
                top = thread["snippet"]["topLevelComment"]
                yield _comment_record(top, video_id)
                yielded += 1
                if yielded >= max_comments:
                    return
                for reply in thread.get("replies", {}).get("comments", []):
                    yield _comment_record(reply, video_id)
                    yielded += 1
                    if yielded >= max_comments:
                        return
            page_token = resp.get("nextPageToken")
            if not page_token:
                return


def _comment_record(comment: dict, video_id: str) -> dict:
    sn = comment["snippet"]
    return {
        "id": comment["id"],
        "author": sn.get("authorDisplayName"),
        "text": sn.get("textOriginal") or sn.get("textDisplay"),
        "published_at": sn.get("publishedAt"),
        "url": f"https://www.youtube.com/watch?v={video_id}&lc={comment['id']}",
        "raw": comment,
    }


def _http_reason(e: HttpError) -> str:
    try:
        return e.error_details[0].get("reason", str(e))  # type: ignore[attr-defined]
    except Exception:
        return str(e)


def _is_quota_error(e: HttpError) -> bool:
    return _http_reason(e) in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded")


def refetch_missing_transcripts(conn, *, delay: float = 1.5,
                                languages: Optional[list[str]] = None) -> dict:
    """Re-attempt transcripts for videos that have a description but no caption
    (i.e. the ones the first pass missed — usually IP-blocked). Throttled with a
    per-video delay to avoid re-triggering the block. Needs no API key.

    Genuinely caption-less videos (transcripts disabled) just fail soft again.
    """
    rows = conn.execute(
        "SELECT id, source_id, url, published_at, genre_seed, author_pseudonym "
        "FROM items WHERE kind='video_description' "
        "AND NOT EXISTS (SELECT 1 FROM items c WHERE c.kind='caption' "
        "                AND c.id = 'cap_' || substr(items.id, 4))"
    ).fetchall()
    stats = {"missing": len(rows), "recovered": 0}
    for i, row in enumerate(rows, 1):
        video_id = row["id"][3:]  # strip the 'vd_' prefix
        text = transcript_mod.fetch_transcript(video_id, languages)
        if text and dbmod.insert_item(
            conn,
            id=f"cap_{video_id}",
            source_id=row["source_id"],
            kind="caption",
            url=row["url"],
            author_pseudonym=row["author_pseudonym"],
            text=text,
            published_at=row["published_at"],
            lang=(languages[0] if languages else "en"),
            genre_seed=row["genre_seed"],
            raw_json={"source": "transcript", "refetched": True},
        ):
            stats["recovered"] += 1
            conn.commit()
            log.info("[%d/%d] recovered transcript for %s (%d chars)",
                     i, len(rows), video_id, len(text))
        time.sleep(delay)
    return stats


def collect_seeds(conn, seeds_cfg: dict, *, quota_budget: int = 9500,
                  api_key: Optional[str] = None) -> dict:
    """Run collection for every seed in the config, writing to `conn`.
    Returns a small stats dict. Fails soft on quota exhaustion."""
    collector = YouTubeCollector(api_key=api_key, quota_budget=quota_budget)
    defaults = seeds_cfg.get("defaults", {})
    stats = {"videos": 0, "items_new": 0, "seeds_run": 0}

    try:
        for seed in seeds_cfg.get("seeds", []):
            name = seed["name"]
            query = seed["query"]
            max_videos = seed.get("max_videos", defaults.get("max_videos", 8))
            per_video = seed.get("comments_per_video", defaults.get("comments_per_video", 60))
            lang = seed.get("relevance_language", defaults.get("relevance_language"))
            order = seed.get("order", defaults.get("order", "relevance"))
            want_desc = seed.get("collect_descriptions", defaults.get("collect_descriptions", True))
            want_comments = seed.get("collect_comments", defaults.get("collect_comments", True))
            want_transcripts = seed.get("collect_transcripts", defaults.get("collect_transcripts", True))
            transcript_langs = [lang] if lang else None

            log.info("seed '%s': %r", name, query)
            try:
                results = collector.search_videos(query, max_videos, order, lang)
            except HttpError as e:
                if _is_quota_error(e):
                    raise QuotaExhausted(f"server quota: {_http_reason(e)}")
                raise
            stats["seeds_run"] += 1

            for r in results:
                video_id = r["id"]["videoId"]
                try:
                    video = collector.get_video(video_id)
                except HttpError as e:
                    if _is_quota_error(e):
                        raise QuotaExhausted(f"server quota: {_http_reason(e)}")
                    raise
                if not video:
                    continue
                stats["videos"] += 1
                vsn = video["snippet"]
                channel_id = vsn.get("channelId", "")
                dbmod.upsert_source(
                    conn,
                    id=channel_id or f"unknown_{video_id}",
                    channel_id=channel_id,
                    channel_title=vsn.get("channelTitle", ""),
                    url=f"https://www.youtube.com/channel/{channel_id}" if channel_id else "",
                )
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                src_id = channel_id or f"unknown_{video_id}"

                if want_desc:
                    desc = vsn.get("description") or ""
                    if dbmod.insert_item(
                        conn,
                        id=f"vd_{video_id}",
                        source_id=src_id,
                        kind="video_description",
                        url=video_url,
                        author_pseudonym=dbmod.pseudonymize(channel_id),
                        text=desc,
                        published_at=vsn.get("publishedAt"),
                        lang=vsn.get("defaultAudioLanguage") or vsn.get("defaultLanguage"),
                        genre_seed=name,
                        raw_json={"title": vsn.get("title")},
                    ):
                        stats["items_new"] += 1

                if want_comments:
                    for c in collector.iter_comments(video_id, per_video, lang):
                        if not c["text"]:
                            continue
                        if dbmod.insert_item(
                            conn,
                            id=c["id"],
                            source_id=src_id,
                            kind="comment",
                            url=c["url"],
                            author_pseudonym=dbmod.pseudonymize(c["author"]),
                            text=c["text"],
                            published_at=c["published_at"],
                            lang=None,
                            genre_seed=name,
                            raw_json=None,
                        ):
                            stats["items_new"] += 1

                if want_transcripts:
                    # Spoken content — what is said in the video. Stored as one
                    # 'caption' item per video. Costs no Data API quota.
                    text = transcript_mod.fetch_transcript(video_id, transcript_langs)
                    if text and dbmod.insert_item(
                        conn,
                        id=f"cap_{video_id}",
                        source_id=src_id,
                        kind="caption",
                        url=video_url,
                        author_pseudonym=dbmod.pseudonymize(channel_id),
                        text=text,
                        published_at=vsn.get("publishedAt"),
                        lang=(lang or None),
                        genre_seed=name,
                        raw_json={"title": vsn.get("title"), "source": "transcript"},
                    ):
                        stats["items_new"] += 1

                conn.commit()
    except QuotaExhausted as e:
        log.warning("stopping early — %s", e)
        conn.commit()
        stats["quota_exhausted"] = True

    stats["quota_spent"] = collector.spent
    return stats
