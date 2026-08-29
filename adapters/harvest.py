"""Patient transcript harvesting.

THE PROBLEM THIS SOLVES. The corpus holds 1.28M comments and almost no
transcripts, which means it is full of people watching a threshold and nearly
empty of the people standing at it: the hospice nurse, the dying man, the
bereaved mother. Those are the primary voices and they are the ones missing.

WHY THE EARLIER ATTEMPT FAILED. Not the library. Both youtube-transcript-api and
yt-dlp get the same HTTP 429, so the limit is on the address, not the client. The
real mistake was behavioural: the old pass kept walking its list of 4,600 videos
after the first block, and every one of those doomed requests told YouTube to
keep the door shut. It recovered 23 transcripts and then spent an hour making
the situation worse.

WHAT THIS DOES INSTEAD. It treats the rate limit as a fact to be respected rather
than fought:

  - takes a small bite per run, not the whole backlog
  - waits a real interval between videos, jittered so the pattern is not machine-flat
  - stops the entire run after a few consecutive blocks, on the principle that
    once the door is shut, knocking harder is worse than coming back tomorrow
  - records nothing it did not get, so re-running simply continues

Run it a few times a day and the backlog clears in a few weeks, for nothing. The
alternative is paying for rotating proxies, which would take an afternoon.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Optional

from adapters import transcript as tmod
from db import db as dbmod

log = logging.getLogger("collect")

BLOCK = {"IpBlocked", "RequestBlocked", "TooManyRequests", "YouTubeRequestFailed"}


def _fetch(video_id: str, languages: Optional[list[str]]) -> tuple[Optional[str], str]:
    """Return (text, status) where status is ok, none, or blocked.

    The existing fetcher collapses "no transcript exists" and "you are blocked"
    into a single None, which is precisely the distinction a circuit breaker
    needs, so this asks the library directly.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None, "none"
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages or ["en"])
        text = " ".join(s.text.strip() for s in fetched if s.text and s.text.strip())
        return (text, "ok") if text else (None, "none")
    except Exception as e:
        return None, "blocked" if type(e).__name__ in BLOCK else "none"


def _ensure(conn) -> None:
    """Remember which videos genuinely have no transcript, so weeks of runs do not
    keep re-asking the same dead ends. A miss is only recorded from a CLEAN run,
    one that saw no blocks at all, because a rate-limited request can arrive
    wearing an error class that looks exactly like 'this video has no captions'.
    Trusting those would quietly bury thousands of recoverable videos."""
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS transcript_misses ("
        " video_id TEXT PRIMARY KEY, checked_at TEXT);"
    )
    conn.commit()


def pending(conn) -> int:
    _ensure(conn)
    return conn.execute(
        "SELECT COUNT(*) FROM items i WHERE i.kind='video_description' "
        "AND NOT EXISTS (SELECT 1 FROM items c WHERE c.kind='caption' "
        "                AND c.id = 'cap_' || substr(i.id, 4)) "
        "AND NOT EXISTS (SELECT 1 FROM transcript_misses m "
        "                WHERE m.video_id = substr(i.id, 4))"
    ).fetchone()[0]


def harvest(conn, *, limit: int = 120, delay: float = 9.0, jitter: float = 4.0,
            stop_after_blocks: int = 3, languages: Optional[list[str]] = None) -> dict:
    _ensure(conn)
    rows = conn.execute(
        "SELECT i.id, i.source_id, i.url, i.published_at, i.genre_seed, i.author_pseudonym "
        "FROM items i WHERE i.kind='video_description' "
        "AND NOT EXISTS (SELECT 1 FROM items c WHERE c.kind='caption' "
        "                AND c.id = 'cap_' || substr(i.id, 4)) "
        "AND NOT EXISTS (SELECT 1 FROM transcript_misses m "
        "                WHERE m.video_id = substr(i.id, 4)) "
        "ORDER BY RANDOM() LIMIT ?", (int(limit),)
    ).fetchall()
    misses: list[str] = []

    stats = {"attempted": 0, "recovered": 0, "no_transcript": 0,
             "blocked": 0, "stopped_early": False, "words": 0}
    streak = 0

    for row in rows:
        vid = row["id"][3:]
        stats["attempted"] += 1
        text, status = _fetch(vid, languages)

        if status == "blocked":
            stats["blocked"] += 1
            streak += 1
            log.info("blocked on %s (%d in a row)", vid, streak)
            if streak >= stop_after_blocks:
                stats["stopped_early"] = True
                log.warning("stopping: %d blocks in a row. The door is shut; "
                            "come back later rather than knocking harder.", streak)
                break
            time.sleep(delay * 4)
            continue

        streak = 0
        if status == "ok" and text:
            if dbmod.insert_item(
                conn, id=f"cap_{vid}", source_id=row["source_id"], kind="caption",
                url=row["url"], author_pseudonym=row["author_pseudonym"], text=text,
                published_at=row["published_at"],
                lang=(languages[0] if languages else "en"),
                genre_seed=row["genre_seed"],
                raw_json={"source": "transcript", "harvest": True},
            ):
                conn.commit()
                stats["recovered"] += 1
                stats["words"] += len(text.split())
                log.info("[%d] %s (%d words)", stats["recovered"], vid, len(text.split()))
        else:
            stats["no_transcript"] += 1
            misses.append(vid)

        time.sleep(delay + random.uniform(0, jitter))

    # Only believe "no transcript" if nothing was blocked during this run.
    if stats["blocked"] == 0 and misses:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT OR IGNORE INTO transcript_misses(video_id, checked_at) VALUES(?,?)",
            [(v, now) for v in misses])
        conn.commit()
        stats["recorded_as_captionless"] = len(misses)
    else:
        stats["recorded_as_captionless"] = 0
        if misses:
            stats["misses_not_trusted"] = len(misses)

    stats["still_pending"] = pending(conn)
    return stats
