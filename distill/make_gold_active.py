"""Build assert-ENRICHED gold chunks from the ranker's P(asserts) scores.

Active learning: label the tagger's TOP-ranked items per language instead of
random ones. Random draws are ~5-10% asserts; the top band should be far richer,
so each Max credit buys several times more of the asserts that actually carry the
cross-cultural map and that the tagger is starving for in training.

NOT prevalence-safe by construction (that's the point) — prevalence stays with
the random frames. These labels are for TRAINING + the source map only.

Budget knob: PER_LANG. Kept modest by default ("some, not all" of the Max quota).
Chunks are JSONL (one record per line) so the Read tool paginates naturally
instead of forcing agents into dozens of sub-reads.
"""
from __future__ import annotations

import json
from pathlib import Path

from db import db as dbmod

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "_emb" / "pscores.json"
OUT = ROOT / "_goldact"
LANGS = ["es", "pt", "hi", "ar", "id", "ja", "tr"]
PER_LANG = 500          # modest: 7 langs -> 3,500 items -> 7 chunks -> ~7 agents
CHUNK = 500
MIN_SCORE = 0.30        # don't bother below this — the band won't be assert-rich


def already_labelled() -> set[str]:
    have: set[str] = set()
    for d in ("_gold", "_goldml", "_gold3", "_goldact"):
        p = ROOT / d
        for f in (p.glob("out_*.json") if p.exists() else []):
            try:
                have |= {r["id"] for r in json.loads(f.read_text(encoding="utf-8"))}
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return have


def main() -> None:
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    skip = already_labelled()
    conn = dbmod.get_conn()
    lang = {r[0]: r[1] for r in conn.execute(
        "SELECT id, lang FROM items WHERE kind='comment' AND lang IS NOT NULL")}
    picked: list[dict] = []
    for lg in LANGS:
        ranked = sorted(
            ((i, s) for i, s in scores.items()
             if lang.get(i) == lg and i not in skip and s >= MIN_SCORE),
            key=lambda x: -x[1])[:PER_LANG]
        ids = [i for i, _ in ranked]
        rows = {}
        for k in range(0, len(ids), 500):
            ch = ids[k:k + 500]
            for r in conn.execute(
                    f"SELECT id, genre_seed, text FROM items WHERE id IN ({','.join('?'*len(ch))})", ch):
                rows[r[0]] = r
        for i, s in ranked:
            r = rows.get(i)
            if r:
                picked.append({"id": i, "lang": lg, "score": s,
                               "text": " ".join((r[2] or "").split())[:400]})
        print(f"  {lg}: {len(ranked)} items, score range "
              f"{ranked[-1][1] if ranked else 0:.2f}-{ranked[0][1] if ranked else 0:.2f}")
    conn.close()

    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()
    n = 0
    for k in range(0, len(picked), CHUNK):
        block = picked[k:k + CHUNK]
        # JSONL: one {id,text} per line
        lines = [json.dumps({"id": i["id"], "text": i["text"]}, ensure_ascii=False) for i in block]
        (OUT / f"chunk_{k // CHUNK:03d}.jsonl").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    (OUT / "_langs.json").write_text(
        json.dumps({i["id"]: i["lang"] for i in picked}, ensure_ascii=False), encoding="utf-8")
    print(f"\nactive gold: {len(picked):,} items -> {n} JSONL chunks of {CHUNK}")


if __name__ == "__main__":
    main()
