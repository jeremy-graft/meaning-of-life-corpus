"""Score every item by P(asserts) with the fine-tuned tagger. The ranker for
active learning — NOT a labeller.

The tagger is only ~0.4 F1, too weak to LABEL the corpus (systematic bias doesn't
average out over 303k). But a weak classifier is still a decent RANKER: items it
scores high are enriched for real asserts. So instead of labelling random items
(~7% assert => ~50 asserts per 750-item agent), we label its top-ranked items
(hopefully ~40%+ => ~300 asserts per agent) — ~6x more asserts per Max credit.

Prevalence stays with the RANDOM frames (already settled). These ranked labels
feed training + the per-language source map, where more asserts is pure upside.

Writes _emb/pscores.json  {id: P(asserts)}.  Free, GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from db import db as dbmod

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "_emb" / "stance_ft_v2"
OUT = ROOT / "_emb" / "pscores.json"
MAX_LEN = 128
BATCH = 128


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    labels = json.loads((MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
    ai = labels.index("asserts")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()

    conn = dbmod.get_conn()
    rows = conn.execute(
        "SELECT id, text FROM items WHERE kind='comment' AND text IS NOT NULL "
        "AND length(text)>40 ORDER BY id").fetchall()
    conn.close()
    print(f"scoring {len(rows):,} comments on {dev}")

    scores: dict[str, float] = {}
    with torch.no_grad():
        for k in range(0, len(rows), BATCH):
            chunk = rows[k:k + BATCH]
            enc = tok([r[1][:500] for r in chunk], truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(dev)
            p = torch.softmax(model(**enc).logits, -1)[:, ai].cpu().tolist()
            for r, s in zip(chunk, p):
                scores[r[0]] = round(float(s), 4)
            if k % (BATCH * 100) == 0:
                print(f"  {k:,}/{len(rows):,}")
    OUT.write_text(json.dumps(scores), encoding="utf-8")
    hi = sum(1 for s in scores.values() if s >= 0.5)
    print(f"wrote {len(scores):,} scores -> {OUT}\n  P(asserts)>=0.5: {hi:,} ({100*hi/len(scores):.1f}%)")


if __name__ == "__main__":
    main()
