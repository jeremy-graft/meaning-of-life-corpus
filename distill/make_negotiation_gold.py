"""Sample reply pairs for coding what strangers do with someone else's meaning.

THE DESIGN. The parent side is already settled. Every parent here is a comment a
careful pass judged as asserting a meaning, with its sources recorded, so this
round only has to code the REPLY. That turns a vague question (how do people talk
about meaning together) into a precise one: when a person says God, what comes
back, and is it different from what comes back when they say family?

Both sides of each pair are carried into the chunk file, because a response can
only be read against what it is responding to.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from db import db as dbmod

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_nego"
CHUNK = 190          # agents hit the 64k output cap above roughly 250
SEED = 73
GOLD_DIRS = ["_gold", "_goldml", "_gold3", "_goldact"]


def load_gold() -> dict:
    g: dict[str, dict] = {}
    for d in GOLD_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        langs = {}
        lp = p / "_langs.json"
        if lp.exists():
            langs = json.loads(lp.read_text(encoding="utf-8"))
        for f in sorted(p.glob("out_*.json")):
            try:
                recs = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for r in recs:
                if r.get("stance") == "asserts" and r.get("sources"):
                    g[r["id"]] = {"sources": r["sources"],
                                  "lang": langs.get(r["id"], "en")}
    return g


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def main() -> None:
    gold = load_gold()
    conn = dbmod.get_conn()
    ids = list(gold)

    # pull the parents' text and every reply they received
    parents: dict[str, str] = {}
    replies: list[tuple[str, str, str]] = []
    for k in range(0, len(ids), 300):
        ch = ids[k:k + 300]
        ph = ",".join("?" * len(ch))
        for i, t in conn.execute(
                f"SELECT id, text FROM items WHERE id IN ({ph})", ch):
            parents[i] = clean(t)
        for pid, rid, rtext in conn.execute(
                f"SELECT substr(r.id,1,instr(r.id,'.')-1), r.id, r.text FROM items r "
                f"WHERE r.kind='comment' AND instr(r.id,'.')>0 "
                f"AND substr(r.id,1,instr(r.id,'.')-1) IN ({ph})", ch):
            replies.append((pid, rid, clean(rtext)))

    rows = []
    for pid, rid, rtext in replies:
        ptext = parents.get(pid, "")
        if len(ptext) < 25 or len(rtext) < 10:
            continue
        if re.search(r"https?://|subscribe to my|check out my channel", rtext, re.I):
            continue
        g = gold[pid]
        rows.append({
            "id": rid,
            "lang": g["lang"],
            "said": ptext[:700],          # the meaning claim, already judged
            "sources": g["sources"],      # what it was judged to be
            "reply": rtext[:700],         # the thing to code
        })

    rng = random.Random(SEED)
    rng.shuffle(rows)

    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json*"):
        f.unlink()

    n = 0
    for k in range(0, len(rows), CHUNK):
        part = rows[k:k + CHUNK]
        lines = [json.dumps({"id": r["id"], "said": r["said"], "reply": r["reply"]},
                            ensure_ascii=False) for r in part]
        (OUT / f"chunk_{k // CHUNK:02d}.jsonl").write_text("\n".join(lines),
                                                           encoding="utf-8")
        n += 1

    meta = {r["id"]: {"lang": r["lang"], "sources": r["sources"]} for r in rows}
    (OUT / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding="utf-8")

    print(f"negotiation gold: {len(rows):,} reply pairs -> {n} chunks of {CHUNK}")
    print("  by language:", dict(Counter(r["lang"] for r in rows).most_common()))
    print("  parent said:", dict(Counter(s for r in rows
                                         for s in r["sources"]).most_common(6)))


if __name__ == "__main__":
    main()
