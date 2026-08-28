"""Package the judgements as a distributable dataset.

Distributes IDs and labels, never the comment text. That is the ordinary practice
for annotation corpora built on someone else's platform: the labels are ours to
give away, the words are not. Anyone who wants the text fetches it themselves from
the YouTube API using the comment id, which also means a deleted comment stays
deleted rather than living on in a file we handed out.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from db import db as dbmod

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset"

# folder -> what that round was for
ROUNDS = {
    "_gold":    "meaning, English, first round",
    "_goldml":  "meaning, seven languages, pilot",
    "_gold3":   "meaning, seven languages, scaled",
    "_goldact": "meaning, seven languages, classifier-enriched sample",
    "_hope":    "coping in irreversible situations",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    conn = dbmod.get_conn()

    rows = []
    for folder, purpose in ROUNDS.items():
        d = ROOT / folder
        if not d.exists():
            continue
        langs = {}
        meta = {}
        if (d / "_langs.json").exists():
            langs = json.loads((d / "_langs.json").read_text(encoding="utf-8"))
        if (d / "_meta.json").exists():
            meta = json.loads((d / "_meta.json").read_text(encoding="utf-8"))
        frames = {}
        if (d / "_frames.json").exists():
            frames = json.loads((d / "_frames.json").read_text(encoding="utf-8"))

        for f in sorted(d.glob("out_*.json")):
            try:
                recs = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for r in recs:
                i = r.get("id")
                if not i:
                    continue
                lang = langs.get(i) or (meta.get(i, {}) or {}).get("lang") or "en"
                row = {
                    "id": i,
                    "round": folder.lstrip("_"),
                    "purpose": purpose,
                    "lang": lang,
                    "frame": frames.get(i, "random"),
                }
                if "stance" in r:
                    row["stance"] = r["stance"]
                    row["sources"] = r.get("sources") or []
                    row["register"] = r.get("register")
                if "strategies" in r:
                    row["strategies"] = r.get("strategies") or []
                    row["conceals"] = bool(r.get("conceals"))
                rows.append(row)

    # attach the situation each comment came from; that is ours, not the platform's
    ids = [r["id"] for r in rows]
    seam = {}
    for k in range(0, len(ids), 400):
        ch = ids[k:k + 400]
        q = f"SELECT id, genre_seed FROM items WHERE id IN ({','.join('?' * len(ch))})"
        for i, s in conn.execute(q, ch):
            seam[i] = s
    conn.close()
    for r in rows:
        r["situation"] = seam.get(r["id"])

    path = OUT / "atlas-labels-v1.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_round = Counter(r["round"] for r in rows)
    by_lang = Counter(r["lang"] for r in rows)
    stance = Counter(r.get("stance") for r in rows if r.get("stance"))
    strat = Counter(s for r in rows for s in r.get("strategies", []))

    print(f"wrote {path}  ({len(rows):,} labels, {path.stat().st_size/1024:.0f} KB)")
    print("  by round:", dict(by_round))
    print("  by language:", dict(by_lang.most_common()))
    print("  stance labels:", dict(stance))
    print("  coping labels:", sum(strat.values()))
    return len(rows), by_round, by_lang


if __name__ == "__main__":
    main()
