"""Multilingual gold sample — the cross-lingual reality check.

Two things this answers at once:
  1. Does ANY tagger work on the ~2/3 of the corpus that is now non-English?
     Every quality number we have (7B asserts-F1 0.64) was measured on English.
     qwen2.5:7b is strong in en/zh and unproven in hi/ar/tr — that 0.64 may not
     transfer, and the whole cross-cultural map depends on whether it does.
  2. Feeds the fine-tune the data it is starving for, WITH cross-lingual coverage.

Sampling: pure random WITHIN each language, equal n per language. The new items
carry no 7B labels, so there is nothing to stratify by — which is a bonus: every
draw here is unbiased, so per-language prevalence is directly estimable (unlike
the English `stratified` frame, which can never be used for prevalence).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from db import db as dbmod

OUT = Path(__file__).resolve().parent.parent / "_goldml"
CHUNK = 200
SEED = 20260702
PER_LANG = 290


def main() -> None:
    rng = random.Random(SEED)
    conn = dbmod.get_conn()
    # ONLY comments, and only the exact seam-target languages. video_description
    # rows carry YouTube's own locale codes instead (es-419, pt-BR, en-US...), so a
    # naive `lang != 'en'` filter silently lets ENGLISH descriptions into a
    # "multilingual" sample — and mixes two different meanings of the column.
    langs = ["es", "pt", "hi", "ar", "id", "ja", "tr"]
    picked: list[dict] = []
    for lg in langs:
        pool = conn.execute(
            "SELECT id, genre_seed, lang, text FROM items WHERE lang=? AND kind='comment' "
            "AND text IS NOT NULL AND length(text)>40 ORDER BY id", (lg,)).fetchall()
        for r in rng.sample(pool, min(PER_LANG, len(pool))):
            picked.append({"id": r[0], "seed": r[1], "lang": r[2],
                           "text": re.sub(r"\s+", " ", r[3]).strip()[:400]})
    conn.close()

    rng.shuffle(picked)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()
    n = 0
    for k in range(0, len(picked), CHUNK):
        blind = [{"id": i["id"], "seed": i["seed"], "text": i["text"]} for i in picked[k:k + CHUNK]]
        (OUT / f"chunk_{k // CHUNK:03d}.json").write_text(
            json.dumps(blind, ensure_ascii=False, indent=0), encoding="utf-8")
        n += 1
    # every draw is random => frame is uniformly "random" (prevalence-safe)
    (OUT / "_frames.json").write_text(
        json.dumps({i["id"]: "random" for i in picked}, ensure_ascii=False), encoding="utf-8")
    (OUT / "_langs.json").write_text(
        json.dumps({i["id"]: i["lang"] for i in picked}, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"multilingual gold: {len(picked)} items -> {n} chunks")
    print("by lang:", dict(Counter(i["lang"] for i in picked)))


if __name__ == "__main__":
    main()
