"""Scaled multilingual gold: 3,000 random comments per language.

WHY 3,000: at a ~7% assert rate, 290 comments/language yields only ~20 asserts —
source shares then carry ±20pp, which is a rumour, not a finding. 3,000 yields
~200 asserts => ±7pp, the point where the cross-cultural map becomes usable.

Pure random within each language (no stratification), so every number here is
prevalence-safe. Excludes ids already labelled in _gold/ and _goldml/ so no agent
work is repeated.

Chunks are 750 (not 200): an agent re-sends its whole context on every tool call,
so per-item cost is dominated by round-trips, not by the items. Bigger chunks
amortise that overhead — the 30-item chunk in the last run cost 1,767 tokens/item
versus ~600 for the 200-item chunks.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from db import db as dbmod

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_gold3"
CHUNK = 750
SEED = 20260703
PER_LANG = 3000
LANGS = ["es", "pt", "hi", "ar", "id", "ja", "tr"]


def already_labelled() -> set[str]:
    have: set[str] = set()
    for d in (ROOT / "_gold", ROOT / "_goldml"):
        for f in d.glob("out_*.json") if d.exists() else []:
            try:
                have |= {r["id"] for r in json.loads(f.read_text(encoding="utf-8"))}
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return have


def main() -> None:
    rng = random.Random(SEED)
    skip = already_labelled()
    conn = dbmod.get_conn()
    picked: list[dict] = []
    for lg in LANGS:
        pool = conn.execute(
            "SELECT id, genre_seed, lang, text FROM items WHERE lang=? AND kind='comment' "
            "AND text IS NOT NULL AND length(text)>40 ORDER BY id", (lg,)).fetchall()
        pool = [r for r in pool if r[0] not in skip]
        take = rng.sample(pool, min(PER_LANG, len(pool)))
        for r in take:
            picked.append({"id": r[0], "seed": r[1], "lang": r[2],
                           "text": re.sub(r"\s+", " ", r[3]).strip()[:400]})
        print(f"  {lg}: pool {len(pool):,} -> took {len(take):,}")
    conn.close()

    rng.shuffle(picked)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()
    n = 0
    for k in range(0, len(picked), CHUNK):
        blind = [{"id": i["id"], "text": i["text"]} for i in picked[k:k + CHUNK]]
        (OUT / f"chunk_{k // CHUNK:03d}.json").write_text(
            json.dumps(blind, ensure_ascii=False), encoding="utf-8")
        n += 1
    (OUT / "_langs.json").write_text(
        json.dumps({i["id"]: i["lang"] for i in picked}, ensure_ascii=False), encoding="utf-8")
    (OUT / "_frames.json").write_text(
        json.dumps({i["id"]: "random" for i in picked}, ensure_ascii=False), encoding="utf-8")
    print(f"\ngold v2: {len(picked):,} items -> {n} chunks of {CHUNK}")


if __name__ == "__main__":
    main()
