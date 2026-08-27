"""Build the gold-label sample that the distilled classifier learns from.

TWO sampling frames, deliberately kept separate — conflating them is the classic
way to get a confident wrong answer:

  frame="stratified"  drawn across the 7B's classes so the RARE classes (denies,
                      make_your_own) get enough examples to train on. NOT a random
                      sample — must never be used to estimate prevalence.
  frame="random"      a pure random draw from the whole corpus. Unbiased, so this
                      is the ONLY frame allowed for prevalence ("what % actually
                      deny?") and for honest held-out evaluation.

The 7B's own labels are used ONLY as a sampling aid to find rare classes — they
are never shown to the labelling agent (that would anchor it to the errors we're
trying to correct).

Writes _gold/chunk_NNN.json for agents to label. Deterministic (fixed seed).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from db import db as dbmod

OUT = Path(__file__).resolve().parent.parent / "_gold"
CHUNK = 200
SEED = 20260702

# 7B class -> how many to draw for the stratified (training) frame.
STRATA = {"none": 300, "asserts": 400, "denies": 250, "make_your_own": 250, "doesnt_know": 100}
RANDOM_N = 800  # the unbiased frame


def main() -> None:
    rng = random.Random(SEED)
    conn = dbmod.get_conn()
    picked: dict[str, dict] = {}

    def rows_for(stance: str):
        return conn.execute(
            "SELECT i.id, i.genre_seed, i.lang, i.text FROM items i JOIN enrichment e ON e.item_id=i.id "
            "WHERE e.model='qwen2.5:7b#g3' AND e.meaning_stance=? AND i.text IS NOT NULL "
            "AND length(i.text)>40 ORDER BY i.id", (stance,)).fetchall()

    for stance, n in STRATA.items():
        pool = rows_for(stance)
        for r in rng.sample(pool, min(n, len(pool))):
            picked[r[0]] = {"id": r[0], "seed": r[1], "lang": r[2] or "en",
                            "text": re.sub(r"\s+", " ", r[3]).strip()[:400], "frame": "stratified"}

    allrows = conn.execute(
        "SELECT i.id, i.genre_seed, i.lang, i.text FROM items i JOIN enrichment e ON e.item_id=i.id "
        "WHERE e.model='qwen2.5:7b#g3' AND i.text IS NOT NULL AND length(i.text)>40 ORDER BY i.id"
    ).fetchall()
    for r in rng.sample(allrows, min(RANDOM_N, len(allrows))):
        if r[0] in picked:          # already drawn by a stratum — the random frame wins,
            picked[r[0]]["frame"] = "random"   # since it's the one we need unbiased
            continue
        picked[r[0]] = {"id": r[0], "seed": r[1], "lang": r[2] or "en",
                        "text": re.sub(r"\s+", " ", r[3]).strip()[:400], "frame": "random"}
    conn.close()

    items = list(picked.values())
    rng.shuffle(items)   # so no chunk is all-one-class (agents shouldn't infer a prior)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()
    n = 0
    for k in range(0, len(items), CHUNK):
        # the agent never sees `frame` or the 7B's label — only the raw text
        blind = [{"id": i["id"], "seed": i["seed"], "text": i["text"]} for i in items[k:k + CHUNK]]
        (OUT / f"chunk_{k // CHUNK:03d}.json").write_text(
            json.dumps(blind, ensure_ascii=False, indent=0), encoding="utf-8")
        n += 1
    (OUT / "_frames.json").write_text(
        json.dumps({i["id"]: i["frame"] for i in items}, ensure_ascii=False), encoding="utf-8")
    frames = {}
    for i in items:
        frames[i["frame"]] = frames.get(i["frame"], 0) + 1
    print(f"gold sample: {len(items)} items -> {n} chunks   frames={frames}")


if __name__ == "__main__":
    main()
