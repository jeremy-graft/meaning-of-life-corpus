"""Run the 7B over exactly the multilingual-gold ids, so we can compare it to
Claude's labels on the SAME items.

This is the cross-lingual reality check. Every 7B quality number we have
(asserts F1 0.64, none-precision 95.8%) was measured on English only. qwen2.5:7b
is strong in en/zh and unproven in hi/ar/tr — if it collapses on those, the whole
cross-cultural map needs a different tagger, and we'd rather know now.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from db import db as dbmod
from enrich import extract

GOLDML = Path(__file__).resolve().parent.parent / "_goldml"
WORKERS = 3


def main() -> None:
    ids: list[str] = []
    for f in sorted(GOLDML.glob("chunk_*.json")):
        ids += [r["id"] for r in json.loads(f.read_text(encoding="utf-8"))]

    conn = dbmod.get_conn()
    dbmod.init_db(conn)
    schema = extract.build_schema(extract._taxonomy())
    model = extract.LOCAL_MODEL
    store = f"{model}#{extract.LOCAL_TAG_VERSION}"
    done = {r[0] for r in conn.execute("SELECT item_id FROM enrichment WHERE model=?", (store,))}
    todo = [i for i in ids if i not in done]
    print(f"7B to tag: {len(todo):,} of {len(ids):,} multilingual gold items")

    rows = {}
    for k in range(0, len(todo), 500):
        chunk = todo[k:k + 500]
        q = f"SELECT * FROM items WHERE id IN ({','.join('?' * len(chunk))})"
        for r in conn.execute(q, chunk):
            rows[r["id"]] = r

    def work(iid):
        r = rows.get(iid)
        if r is None:
            return iid, None
        return iid, extract.enrich_item_local(model, r, schema)

    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for iid, data in pool.map(work, todo):
            if data is None:
                continue
            dbmod.upsert_enrichment(
                conn, item_id=iid, meaning_sources=data["meaning_sources"],
                meaning_stance=data["meaning_stance"], life_stage=data["life_stage"],
                age_band_guess=data.get("age_band_guess"),
                sincerity_register=data["sincerity_register"], themes=data["themes"],
                essence=data["essence"], pull_quote=data["pull_quote"], model=store)
            conn.commit()
            n += 1
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}")
    conn.close()
    print(f"7B tagged {n:,}")


if __name__ == "__main__":
    main()
