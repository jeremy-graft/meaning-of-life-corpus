"""Reddit adapter — the richest confession corpus on the internet.

WHY IT'S THE BIGGEST VEIN: Reddit is anonymous, long-form, and organised into
exactly the communities this project mines by hand elsewhere — r/GriefSupport,
r/stopdrinking, r/exchristian, r/cancer, r/widowers, r/childfree. On YouTube we
must *infer* the seam from the video genre; on Reddit the seam IS the subreddit,
declared by the poster. That makes genre_seed far less noisy here than anywhere
else in the corpus.

SETUP (one-time, ~2 minutes, free — see README section printed by `collect-reddit`):
  1. https://www.reddit.com/prefs/apps -> "create another app..."
  2. choose type: **script**;  redirect uri: http://localhost:8080
  3. put the two values in .env (NOT in chat):
         REDDIT_CLIENT_ID=...
         REDDIT_CLIENT_SECRET=...
We use app-only OAuth (client_credentials) — read-only, no user account access,
no password needed.

VOLUME: Reddit caps any single listing at ~1,000 items. We beat that the same way
we beat Algolia — by slicing: subreddit x sort(new/top) x time-window(all/year/
month) x listing pagination. Many subreddits x several slices each = deep reach.
Comments (not just posts) are fetched per submission — that's where confession
actually lives.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from db import db as dbmod

log = logging.getLogger("collect")

UA = "script:meaning-of-life-corpus:0.1 (by /u/research_corpus)"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH = "https://oauth.reddit.com"

SETUP_HELP = """
Reddit credentials are not set. One-time setup (free, ~2 min):
  1. Go to https://www.reddit.com/prefs/apps  ->  "create another app..."
  2. Type: script      Redirect uri: http://localhost:8080
  3. Add to D:\\Claude Code\\meaning-of-life-corpus\\.env  (keep them out of chat):
         REDDIT_CLIENT_ID=<the string under the app name>
         REDDIT_CLIENT_SECRET=<the "secret" field>
Then re-run:  python cli.py collect-reddit
"""


class RedditAuthMissing(Exception):
    pass


class Reddit:
    def __init__(self):
        cid = os.environ.get("REDDIT_CLIENT_ID")
        sec = os.environ.get("REDDIT_CLIENT_SECRET")
        if not cid or not sec:
            raise RedditAuthMissing(SETUP_HELP)
        self.token = self._auth(cid, sec)
        self.calls = 0

    def _auth(self, cid: str, sec: str) -> str:
        body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
        basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
        req = urllib.request.Request(TOKEN_URL, data=body, headers={
            "Authorization": f"Basic {basic}", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))["access_token"]

    def get(self, path: str, params: dict, retries: int = 4) -> dict:
        url = f"{OAUTH}{path}?" + urllib.parse.urlencode(params)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={
                    "Authorization": f"bearer {self.token}", "User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    self.calls += 1
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")


def _walk_comments(node, out, depth=0):
    """Flatten a reddit comment tree (skip 'more' stubs)."""
    if not isinstance(node, dict):
        return
    for ch in (node.get("data", {}).get("children") or []):
        if ch.get("kind") != "t1":
            continue
        d = ch.get("data", {})
        if d.get("body") and d["body"] not in ("[deleted]", "[removed]"):
            out.append(d)
        rep = d.get("replies")
        if isinstance(rep, dict) and depth < 6:
            _walk_comments(rep, out, depth + 1)


def collect_reddit(conn, cfg: dict, delay: float = 1.1, min_chars: int = 80,
                   max_posts_per_slice: int = 300, comments_per_post: int = 0) -> dict:
    """Harvest posts (+optionally comment trees) from seam-named subreddits."""
    r = Reddit()
    stats = {"calls": 0, "items_new": 0, "by_seam": {}}
    for seam in cfg.get("seams", []):
        name = seam["name"]
        for sub in seam.get("subreddits", []):
            sid = f"reddit_{sub}"
            dbmod.upsert_source(conn, id=sid, channel_id=sid, channel_title=f"r/{sub}",
                                url=f"https://reddit.com/r/{sub}")
            for sort, tparam in cfg.get("slices", [["new", None], ["top", "all"],
                                                    ["top", "year"]]):
                after, got = None, 0
                while got < max_posts_per_slice:
                    p = {"limit": 100}
                    if after:
                        p["after"] = after
                    if tparam:
                        p["t"] = tparam
                    try:
                        d = r.get(f"/r/{sub}/{sort}", p)
                    except Exception as e:
                        log.warning("reddit %s/%s failed: %s", sub, sort, type(e).__name__)
                        break
                    kids = (d.get("data") or {}).get("children") or []
                    if not kids:
                        break
                    for ch in kids:
                        pd = ch.get("data") or {}
                        got += 1
                        body = (pd.get("selftext") or "").strip()
                        if body in ("[deleted]", "[removed]"):
                            body = ""
                        text = ((pd.get("title") or "") + "\n\n" + body).strip()
                        if len(text) >= min_chars and dbmod.insert_item(
                            conn, id=f"rd_{pd.get('id')}", source_id=sid, kind="comment",
                            url="https://reddit.com" + (pd.get("permalink") or ""),
                            author_pseudonym=dbmod.pseudonymize(pd.get("author")),
                            text=text,
                            published_at=None, lang="en", genre_seed=name,
                            raw_json={"src": "reddit", "sub": sub, "type": "post",
                                      "score": pd.get("score")},
                        ):
                            stats["items_new"] += 1
                            stats["by_seam"][name] = stats["by_seam"].get(name, 0) + 1
                        # comment tree — where the confession actually is
                        if comments_per_post and pd.get("id"):
                            try:
                                cd = r.get(f"/comments/{pd['id']}",
                                           {"limit": comments_per_post, "depth": 6})
                                flat: list = []
                                if isinstance(cd, list) and len(cd) > 1:
                                    _walk_comments(cd[1], flat)
                                for cm in flat[:comments_per_post]:
                                    if len(cm.get("body", "")) < min_chars:
                                        continue
                                    if dbmod.insert_item(
                                        conn, id=f"rdc_{cm.get('id')}", source_id=sid,
                                        kind="comment",
                                        url="https://reddit.com" + (cm.get("permalink") or ""),
                                        author_pseudonym=dbmod.pseudonymize(cm.get("author")),
                                        text=cm["body"], published_at=None, lang="en",
                                        genre_seed=name,
                                        raw_json={"src": "reddit", "sub": sub,
                                                  "type": "comment", "score": cm.get("score")},
                                    ):
                                        stats["items_new"] += 1
                                        stats["by_seam"][name] = stats["by_seam"].get(name, 0) + 1
                                time.sleep(delay)
                            except Exception as e:
                                log.warning("comments %s: %s", pd.get("id"), type(e).__name__)
                    conn.commit()
                    after = (d.get("data") or {}).get("after")
                    time.sleep(delay)
                    if not after:
                        break
                log.info("reddit r/%s %s/%s -> %d new total", sub, sort, tparam or "-",
                         stats["items_new"])
    stats["calls"] = r.calls
    return stats
