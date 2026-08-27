"""Stack Exchange adapter — a third register: ARGUED meaning-discourse.

The corpus so far has two voices:
  YouTube  — confessed (raw_confession under hospice/grief videos)
  HN       — reflected (secular, technical, discursive)
Stack Exchange adds a third: people EXPLICITLY reasoning about meaning, with
citations and rebuttals, on philosophy / buddhism / christianity / islam /
judaism / hermeneutics. Nobody stumbles into saying what life means here — they
argue it on purpose. That is a genuinely different speech act, and it is the
natural counterweight to seam-mined confession.

API: api.stackexchange.com v2.3.
  - WITHOUT a key: 300 requests/day per IP (x100 items = ~30k items/day)
  - WITH a free key (register at stackapps.com/apps/oauth/register, no OAuth
    flow needed, just the form): 10,000/day — a 33x lift.
Set STACKEXCHANGE_KEY in .env to use one. The adapter fails soft on quota.

Answers are the payload: a question is a prompt, an answer is a claim. We store
both, tagged in raw_json.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from db import db as dbmod

log = logging.getLogger("collect")

API = "https://api.stackexchange.com/2.3"
UA = "meaning-of-life-corpus/0.1 (research; contact jerrrr92@gmail.com)"
PAGESIZE = 100
_TAG = re.compile(r"<[^>]+>")


class SEQuotaExhausted(Exception):
    pass


class SE:
    def __init__(self, budget: int = 250):
        self.key = os.environ.get("STACKEXCHANGE_KEY") or None
        self.budget = budget          # our own request cap (stay under the daily limit)
        self.spent = 0
        self.remaining = None

    def get(self, path: str, params: dict, retries: int = 3) -> dict:
        if self.spent >= self.budget:
            raise SEQuotaExhausted(f"local request budget {self.budget} reached")
        p = {**params, "pagesize": PAGESIZE}
        if self.key:
            p["key"] = self.key
        url = f"{API}/{path}?" + urllib.parse.urlencode(p)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                           "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        import gzip
                        raw = gzip.decompress(raw)
                    d = json.loads(raw.decode("utf-8"))
                self.spent += 1
                self.remaining = d.get("quota_remaining", self.remaining)
                if d.get("backoff"):                 # SE tells us to slow down
                    time.sleep(d["backoff"] + 1)
                if self.remaining is not None and self.remaining <= 2:
                    raise SEQuotaExhausted(f"server quota_remaining={self.remaining}")
                return d
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")


def _clean(t: str) -> str:
    t = (t or "").replace("</p>", "\n\n").replace("<br>", "\n")
    return html.unescape(_TAG.sub("", t)).strip()


def collect_se(conn, cfg: dict, budget: int = 250, min_chars: int = 80,
               delay: float = 0.3) -> dict:
    """Search each site for each seam query; store question bodies + answers."""
    se = SE(budget=budget)
    sites = cfg.get("sites", [])
    stats = {"requests": 0, "items_new": 0, "by_site": {}, "by_seam": {}}
    for site in sites:
        sid = f"se_{site}"
        dbmod.upsert_source(conn, id=sid, channel_id=sid,
                            channel_title=f"{site}.stackexchange.com",
                            url=f"https://{site}.stackexchange.com")
    try:
        for site in sites:
            sid = f"se_{site}"
            for seam in cfg.get("seams", []):
                name = seam["name"]
                for q in seam.get("queries", []):
                    qids: list[int] = []
                    for page in range(1, cfg.get("max_pages", 3) + 1):
                        d = se.get("search/advanced", {
                            "site": site, "q": q, "page": page, "sort": "votes",
                            "order": "desc", "filter": "withbody"})
                        for it in d.get("items", []):
                            qid = it.get("question_id")
                            if qid:
                                qids.append(qid)
                            body = _clean(it.get("body"))
                            if len(body) < min_chars:
                                continue
                            if dbmod.insert_item(
                                conn, id=f"se_{site}_q{qid}", source_id=sid, kind="comment",
                                url=it.get("link"),
                                author_pseudonym=dbmod.pseudonymize(
                                    str((it.get("owner") or {}).get("account_id") or "")),
                                text=(it.get("title", "") + "\n\n" + body).strip(),
                                published_at=None, lang="en", genre_seed=name,
                                raw_json={"se_type": "question", "site": site,
                                          "score": it.get("score")},
                            ):
                                stats["items_new"] += 1
                                stats["by_seam"][name] = stats["by_seam"].get(name, 0) + 1
                                stats["by_site"][site] = stats["by_site"].get(site, 0) + 1
                        conn.commit()
                        time.sleep(delay)
                        if not d.get("has_more"):
                            break
                    # answers for those questions — where the actual claims live
                    for k in range(0, len(qids), 100):
                        ids = ";".join(str(i) for i in qids[k:k + 100])
                        if not ids:
                            break
                        d = se.get(f"questions/{ids}/answers", {
                            "site": site, "sort": "votes", "order": "desc",
                            "filter": "withbody"})
                        for a in d.get("items", []):
                            body = _clean(a.get("body"))
                            if len(body) < min_chars:
                                continue
                            if dbmod.insert_item(
                                conn, id=f"se_{site}_a{a.get('answer_id')}", source_id=sid,
                                kind="comment",
                                url=f"https://{site}.stackexchange.com/a/{a.get('answer_id')}",
                                author_pseudonym=dbmod.pseudonymize(
                                    str((a.get("owner") or {}).get("account_id") or "")),
                                text=body, published_at=None, lang="en", genre_seed=name,
                                raw_json={"se_type": "answer", "site": site,
                                          "score": a.get("score")},
                            ):
                                stats["items_new"] += 1
                                stats["by_seam"][name] = stats["by_seam"].get(name, 0) + 1
                                stats["by_site"][site] = stats["by_site"].get(site, 0) + 1
                        conn.commit()
                        time.sleep(delay)
                    log.info("se %s/%s q=%r -> %d new (req %d, quota_left %s)",
                             site, name, q, stats["items_new"], se.spent, se.remaining)
    except SEQuotaExhausted as e:
        log.warning("stopping early — %s", e)
        stats["quota_exhausted"] = True
    stats["requests"] = se.spent
    stats["quota_remaining"] = se.remaining
    return stats
