"""Embed every item once, cache to disk. The engine room of the distillation.

WHY THIS EXISTS: the 7B ran at ~0.55 items/sec — ~2 days per 110k pass, ~3 weeks
per 1M, and every schema change cost another full pass. An embedding model on the
same laptop does hundreds/sec, and once embedded, re-training a classifier on new
labels takes SECONDS. That's what makes a 1M-item corpus re-taggable instead of
frozen.

MULTILINGUAL ON PURPOSE: the corpus is now ~2/3 non-English (es/pt/ar/id/tr/ja/hi).
An English-only embedder would silently destroy exactly the cross-cultural
comparison the multilingual collection was built for. This model maps
semantically-similar text from 50+ languages into ONE shared space — which also
means a classifier trained on English gold partially transfers to Japanese
(imperfectly; multilingual gold still helps, see distill/make_gold.py).

Cache is incremental: re-running only embeds items it hasn't seen.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from db import db as dbmod

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path(__file__).resolve().parent.parent / "_emb"
MAX_CHARS = 500
BATCH = 256


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()[:MAX_CHARS]


def load_cache() -> tuple[list[str], np.ndarray]:
    ids_p, vec_p = CACHE / "ids.json", CACHE / "vecs.npy"
    if ids_p.exists() and vec_p.exists():
        return json.loads(ids_p.read_text(encoding="utf-8")), np.load(vec_p)
    return [], np.zeros((0, 384), dtype=np.float32)


def embed_all(limit: int | None = None) -> dict:
    from sentence_transformers import SentenceTransformer

    CACHE.mkdir(exist_ok=True)
    have_ids, have_vecs = load_cache()
    have = set(have_ids)

    conn = dbmod.get_conn()
    rows = conn.execute(
        "SELECT id, text FROM items WHERE text IS NOT NULL AND TRIM(text) != '' ORDER BY id"
    ).fetchall()
    conn.close()
    todo = [(r[0], clean(r[1])) for r in rows if r[0] not in have]
    todo = [(i, t) for i, t in todo if t]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"embedded": 0, "cached": len(have_ids), "dim": int(have_vecs.shape[1]) if len(have_ids) else 0}

    print(f"loading {MODEL} …")
    model = SentenceTransformer(MODEL)
    print(f"embedding {len(todo):,} new items (cached: {len(have_ids):,})")
    vecs = model.encode([t for _, t in todo], batch_size=BATCH, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)

    all_ids = have_ids + [i for i, _ in todo]
    all_vecs = np.vstack([have_vecs, vecs.astype(np.float32)]) if len(have_ids) else vecs.astype(np.float32)
    (CACHE / "ids.json").write_text(json.dumps(all_ids), encoding="utf-8")
    np.save(CACHE / "vecs.npy", all_vecs)
    return {"embedded": len(todo), "cached_total": len(all_ids), "dim": int(all_vecs.shape[1])}


if __name__ == "__main__":
    print(json.dumps(embed_all(), indent=2))
