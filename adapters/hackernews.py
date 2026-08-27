"""Hacker News adapter — a second crowd, deliberately unlike the first.

WHY: the YouTube corpus is mined from hospice/grief/faith seams and comes back
religion-dominated (ar 70% ... ja 11%). HN is the opposite demographic — secular,
Western, technical — and a different REGISTER (argued/reflective rather than raw
confession). If meaning really is situational, this population should name
craft_work / growth_self / purpose_cause where the hospice crowd names God. That
contrast is the point; it is not more of the same data.

API: the Algolia HN search API. Free, no auth, generous limits.

VOLUME TRICK: Algolia caps any single query at ~1,000 hits. To get past that we
slice each query into YEAR windows via numericFilters on created_at_i — ~19 years
of HN, so ~19x the reach per query. That is what makes this a massive-scale source
rather than a 1,000-comment toy.
"""
from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from db import db as dbmod

log = logging.getLogger("collect")

API = "https://hn.algolia.com/api/v1/search_by_date"
UA = "meaning-of-life-corpus/0.1 (research; contact jerrrr92@gmail.com)"
SOURCE_ID = "hackernews"
HITS = 100
MAX_PAGE = 10          # Algolia hard-caps ~1000 results per query window
YEARS = range(2007, 2027)


def _get(params: dict, retries: int = 4) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


_TAG = re.compile(r"<[^>]+>")


def _clean(t: str) -> str:
    """HN comment_text is HTML-escaped with <p>/<i>/<a> markup."""
    t = t.replace("<p>", "\n\n")
    t = _TAG.sub("", t)
    return html.unescape(t).strip()


def collect_hn(conn, cfg: dict, delay: float = 0.15, min_chars: int = 60) -> dict:
    """Harvest HN comments per seam. Idempotent: re-runs skip existing ids."""
    dbmod.upsert_source(conn, id=SOURCE_ID, channel_id=SOURCE_ID,
                        channel_title="Hacker News", url="https://news.ycombinator.com")
    stats = {"queries": 0, "items_new": 0, "by_seam": {}}
    for seam in cfg.get("seams", []):
        name = seam["name"]
        for q in seam.get("queries", []):
            stats["queries"] += 1
            for yr in YEARS:
                start = int(datetime(yr, 1, 1, tzinfo=timezone.utc).timestamp())
                end = int(datetime(yr + 1, 1, 1, tzinfo=timezone.utc).timestamp())
                for page in range(MAX_PAGE):
                    try:
                        d = _get({"tags": "comment", "query": q, "hitsPerPage": HITS,
                                  "page": page,
                                  "numericFilters": f"created_at_i>{start},created_at_i<{end}"})
                    except Exception as e:
                        log.warning("hn fetch failed %r %s p%d: %s", q, yr, page, type(e).__name__)
                        break
                    hits = d.get("hits", [])
                    if not hits:
                        break
                    for h in hits:
                        text = _clean(h.get("comment_text") or "")
                        if len(text) < min_chars:
                            continue
                        oid = h.get("objectID")
                        if dbmod.insert_item(
                            conn,
                            id=f"hn_{oid}",
                            source_id=SOURCE_ID,
                            kind="comment",
                            url=f"https://news.ycombinator.com/item?id={oid}",
                            author_pseudonym=dbmod.pseudonymize(h.get("author")),
                            text=text,
                            published_at=h.get("created_at"),
                            lang="en",
                            genre_seed=name,
                            raw_json={"story": h.get("story_title"), "src": "hn"},
                        ):
                            stats["items_new"] += 1
                            stats["by_seam"][name] = stats["by_seam"].get(name, 0) + 1
                    conn.commit()
                    time.sleep(delay)
                    if page + 1 >= d.get("nbPages", 0):
                        break
            log.info("hn seam '%s' q=%r -> running total %d", name, q, stats["items_new"])
    return stats
