"""Wikipedia 'Themes'/'Analysis' adapter — the *interpreted* meaning of a work.

A quote FROM a film/novel is often the opposite of what the work MEANS (villains,
unreliable narrators, irony — Gekko says "greed is good"; the film indicts him).
So instead of extracting quotes, we pull the interpretive **Themes / Analysis /
Interpretation** section of a work's Wikipedia article — a critical synthesis of
what the work actually says. CC BY-SA, free, no key (MediaWiki Action API).

The crowd layer (YouTube) lives in `items`; the canon layer lives in a separate
`works` table so collection here never touches the corpus being re-tagged.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("collect")

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia grants higher rate limits to a policy-compliant UA that carries contact info.
UA = "meaning-of-life-corpus/0.1 (research; contact jerrrr92@gmail.com) python-urllib"
# Section headings that hold interpretation (not plot/cast/production).
SECTION_RE = re.compile(r"theme|analys|interpretat|symbolism|meaning|motif|philosoph", re.I)


def _get(params: dict, retries: int = 5) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 429 = rate-limited by a burst; 5xx = transient. Back off and retry.
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 2 ** attempt  # 1,2,4,8,16s
                log.info("wiki HTTP %s, backing off %ds", e.code, wait)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _clean(wt: str) -> str:
    """Strip wikitext to readable prose (good enough for a model to read)."""
    wt = re.sub(r"<ref[^>]*?/>", "", wt)
    wt = re.sub(r"<ref[^>]*?>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    for _ in range(3):  # collapse nested templates a few levels
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", wt)   # [[link|text]] -> text
    wt = re.sub(r"\[\[([^\]]*)\]\]", r"\1", wt)            # [[link]] -> link
    wt = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", wt)   # [url label] -> label
    wt = re.sub(r"'''?", "", wt)                           # bold / italic
    wt = re.sub(r"<[^>]+>", "", wt)                        # any stray html
    wt = re.sub(r"^\s*=+.*?=+\s*$", "", wt, flags=re.M)    # sub-headings
    wt = re.sub(r"[ \t]+", " ", wt)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def fetch_themes(title: str) -> Optional[dict]:
    """Return {title, text, url} of a work's interpretive sections, or None."""
    data = _get({"action": "parse", "page": title, "prop": "sections", "redirects": 1})
    if "error" in data:
        log.info("no page for %r (%s)", title, data["error"].get("code"))
        return None
    real = data["parse"]["title"]
    secs = [s for s in data["parse"]["sections"] if SECTION_RE.search(s["line"])]
    if not secs:
        log.info("no interpretive section for %r", real)
        return None
    parts = []
    for s in secs:
        d2 = _get({"action": "parse", "page": real, "section": s["index"],
                   "prop": "wikitext", "redirects": 1})
        wt = d2.get("parse", {}).get("wikitext", {}).get("*", "")
        cleaned = _clean(wt)
        if len(cleaned) > 40:
            parts.append(f"== {s['line']} ==\n{cleaned}")
    text = "\n\n".join(parts).strip()
    if len(text) < 80:
        return None
    return {
        "title": real,
        "text": text,
        "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(real.replace(" ", "_")),
    }


def category_members(category: str, limit: int = 500) -> list[str]:
    """Article titles (namespace 0) in a Wikipedia category. Paginated."""
    out: list[str] = []
    cont: Optional[str] = None
    while True:
        params = {"action": "query", "list": "categorymembers",
                  "cmtitle": f"Category:{category}", "cmlimit": "500", "cmtype": "page"}
        if cont:
            params["cmcontinue"] = cont
        d = _get(params)
        for m in d.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 0:
                out.append(m["title"])
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont or len(out) >= limit:
            break
    return out[:limit]


def subcategories(category: str, limit: int = 60) -> list[str]:
    """Direct sub-category names (no 'Category:' prefix) of a category."""
    out: list[str] = []
    cont: Optional[str] = None
    while True:
        params = {"action": "query", "list": "categorymembers",
                  "cmtitle": f"Category:{category}", "cmlimit": "500", "cmtype": "subcat"}
        if cont:
            params["cmcontinue"] = cont
        d = _get(params)
        for m in d.get("query", {}).get("categorymembers", []):
            out.append(m["title"].split("Category:", 1)[-1])
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont or len(out) >= limit:
            break
    return out[:limit]


# Sub-categories whose names signal topic-drift away from meaning-grappling works.
_SUBCAT_DRIFT_RE = re.compile(
    r"\b(actor|actress|award|adaptation|soundtrack|by year|by decade|stub|template|"
    r"screenwriter|director|writers|publisher|magazine|franchise|serial killer|video game)",
    re.I,
)


def _expand_categories(categories_cfg: dict, recurse: int, subcat_cap: int) -> list[tuple[str, str]]:
    """Build the (medium, category) work-list. With recurse>=1, each seed category
    also contributes one level of its sub-categories (drift-filtered, deduped)."""
    work: list[tuple[str, str]] = []
    seen: set[str] = set()
    for medium, cats in categories_cfg.items():
        for cat in cats or []:
            wanted = [cat]
            if recurse >= 1:
                try:
                    subs = subcategories(cat, limit=subcat_cap)
                except Exception as e:
                    log.warning("subcat fetch failed %r: %s", cat, type(e).__name__)
                    subs = []
                wanted += [s for s in subs if not _SUBCAT_DRIFT_RE.search(s)]
            for c in wanted:
                if c in seen:  # a subcat can hang under several seeds — harvest once
                    continue
                seen.add(c)
                work.append((medium, c))
    return work


# Titles that are clearly not works — lists, indices, franchise hubs.
_SKIP_RE = re.compile(r"^(List of|Index of|Outline of)\b|\(disambiguation\)", re.I)


def discover_works(conn, categories_cfg: dict, per_cat: int = 150, delay: float = 0.2,
                   recurse: int = 0, subcat_cap: int = 60) -> dict:
    """Harvest works from Wikipedia categories, keeping only those with a real
    Themes/Analysis section. Resumable: already-collected titles are skipped.
    recurse>=1 also descends one level into each category's sub-categories."""
    ensure_works_table(conn)
    have = {row[0] for row in conn.execute("SELECT title FROM works")}
    seen: set[str] = set()
    stats = {"candidates": 0, "collected": 0, "no_themes": 0, "skipped": 0,
             "by_medium": {}, "dead_categories": []}
    work_cats = _expand_categories(categories_cfg, recurse, subcat_cap)
    log.info("harvesting %d categories (recurse=%d)", len(work_cats), recurse)
    for medium, cat in work_cats:
        try:
            titles = category_members(cat, limit=per_cat)
        except Exception as e:
            log.warning("category fetch failed %r: %s", cat, type(e).__name__)
            continue
        if not titles:
            stats["dead_categories"].append(cat)
            continue
        log.info("category %s [%s]: %d candidate pages", cat, medium, len(titles))
        for title in titles:
            if title in seen or _SKIP_RE.search(title):
                continue
            seen.add(title)
            if title in have:
                stats["skipped"] += 1
                continue
            stats["candidates"] += 1
            time.sleep(delay)  # polite throttle on EVERY probe, not just hits
            try:
                r = fetch_themes(title)
            except Exception as e:
                log.warning("themes fail %r: %s", title, type(e).__name__)
                r = None
            if not r:
                stats["no_themes"] += 1
                continue
            conn.execute(
                "INSERT OR REPLACE INTO works(id, title, medium, seed, themes_text, url, collected_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (r["title"], r["title"], medium, cat, r["text"], r["url"],
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            have.add(r["title"])
            stats["collected"] += 1
            stats["by_medium"][medium] = stats["by_medium"].get(medium, 0) + 1
            log.info("  + [%s] %s (%d chars)", medium, r["title"], len(r["text"]))
    return stats


def ensure_works_table(conn) -> None:
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS works ("
        " id TEXT PRIMARY KEY, title TEXT, medium TEXT, seed TEXT,"
        " themes_text TEXT, url TEXT, collected_at TEXT);"
    )
    conn.commit()


def collect_works(conn, works_cfg: dict, delay: float = 1.0) -> dict:
    """works_cfg = {'film': [titles...], 'novel': [titles...]}. Stored in `works`."""
    ensure_works_table(conn)
    have = {row[0] for row in conn.execute("SELECT title FROM works")}
    stats = {"attempted": 0, "collected": 0, "no_themes": 0, "skipped": 0}
    for medium, titles in works_cfg.items():
        for title in titles or []:
            if title in have:  # already have themes for this exact seed; don't re-hammer
                stats["skipped"] += 1
                continue
            stats["attempted"] += 1
            try:
                r = fetch_themes(title)
            except Exception as e:
                log.warning("fetch failed for %r: %s", title, type(e).__name__)
                r = None
            if not r:
                stats["no_themes"] += 1
                continue
            conn.execute(
                "INSERT OR REPLACE INTO works(id, title, medium, seed, themes_text, url, collected_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (r["title"], r["title"], medium, medium, r["text"], r["url"],
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            stats["collected"] += 1
            log.info("themes: %s (%d chars)", r["title"], len(r["text"]))
            time.sleep(delay)
    return stats
