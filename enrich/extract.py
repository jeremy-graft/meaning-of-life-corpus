"""Enrichment — two interchangeable backends, one schema.

For each un-enriched item, one model call returns STRICT JSON conforming to a
json_schema built from the editable taxonomy. Parsing is defensive: any failure
logs and skips, never crashing the batch. Enrichment is cached by item id — an
already-enriched item is skipped unless --force.

Two backends, same prompt + schema:
  - API (default): Claude via the Anthropic API (claude-haiku-4-5 for volume,
    claude-opus-4-8 for a curated re-pass). Costs money per item.
  - local (--local): a model served by Ollama on this machine (qwen2.5:7b).
    Free per item — the volume workhorse. Reserve the API for the curated pass.

This is the brief's own two-tier design ("fast model for volume; stronger
curated re-pass") with the volume tier swapped to a free local model.
The `enrichment.model` column records which model tagged each item.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import anthropic
import yaml

from db import db as dbmod

log = logging.getLogger("enrich")

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.yaml")

# API tiers: fast model for volume; opus for a curated re-pass on starred items.
DEFAULT_MODEL = "claude-haiku-4-5"
CURATED_MODEL = "claude-opus-4-8"
# Free local workhorse (served by Ollama). Strong JSON adherence for its size.
LOCAL_MODEL = "qwen2.5:7b"
# Local model: keep the context SMALL for speed (a big window tanks throughput
# on the laptop GPU), and cap input chars tight enough that nothing can overflow
# it. 6k chars (~1.5k tokens) is plenty to judge any comment; long transcripts
# are truncated. Output is capped so a looping generation can't hang the run.
LOCAL_MAX_CHARS = 6000
LOCAL_NUM_CTX = 3072
LOCAL_NUM_PREDICT = 500  # hard output cap — stops runaway/looping generations

PULL_QUOTE_MAX_WORDS = 14

SYSTEM_PROMPT = (
    "You are a careful close-reader building an art corpus about what gives "
    "human life meaning. You are NOT a census-taker: do not flatten or average. "
    "Read one fragment at a time and preserve its frame.\n\n"
    "Your single most important judgment is `sincerity_register` — the gap "
    "between the mask someone performs and the face that bleeds through. Take it "
    "seriously:\n"
    "  raw_confession  — unguarded; saying it appears to cost the speaker something\n"
    "  reflective      — considered, thinking out loud, not performing\n"
    "  advice_giving   — turned outward into a lesson for an audience\n"
    "  performed_brand — shaped for an audience or the algorithm\n"
    "  joke_deflection — deflects real feeling with humor\n\n"
    "Infer life_stage and an optional age_band only from internal evidence in "
    "the text; use 'unknown' / null when you cannot tell. Choose meaning_sources "
    "ONLY from the provided taxonomy; themes are free lowercase tags. The essence "
    "is one sentence in your own words. The pull_quote is verbatim and UNDER 15 "
    "words — it exists for provenance, not display.\n"
    "Return ONLY the structured object. No prose, no markdown."
)


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_schema(taxonomy: dict) -> dict:
    """A strict json_schema derived from the (editable) taxonomy: enum-constrain
    the controlled fields, leave themes free, allow null for the age band."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "meaning_sources": {
                "type": "array",
                "items": {"type": "string", "enum": taxonomy["meaning_sources"]},
            },
            "themes": {"type": "array", "items": {"type": "string"}},
            "life_stage": {"type": "string", "enum": taxonomy["life_stage"]},
            "age_band_guess": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "sincerity_register": {"type": "string", "enum": taxonomy["sincerity_register"]},
            "essence": {"type": "string"},
            "pull_quote": {"type": "string"},
        },
        "required": [
            "meaning_sources", "themes", "life_stage", "age_band_guess",
            "sincerity_register", "essence", "pull_quote",
        ],
    }


def _user_prompt(item: Any, max_chars: Optional[int] = None) -> str:
    kind = item["kind"]
    where = {
        "comment": "a YouTube comment",
        "video_description": "a YouTube video description",
        "caption": "a YouTube caption track",
    }.get(kind, "a fragment")
    text = item["text"] or ""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return (
        f"This is {where}, harvested from the genre-seed '{item['genre_seed']}'.\n\n"
        f"--- fragment ---\n{text}\n--- end ---"
    )


def _truncate_quote(quote: str, max_words: int = PULL_QUOTE_MAX_WORDS) -> str:
    words = quote.split()
    if len(words) <= max_words:
        return quote
    return " ".join(words[:max_words]) + "…"


_TAXONOMY: Optional[dict] = None


def _taxonomy() -> dict:
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = load_taxonomy()
    return _TAXONOMY


def is_low_signal(text: Optional[str]) -> bool:
    """Cheap pre-filter: true for items not worth a model call — too short to
    carry a meaning-statement, or mostly emoji/symbols ('first!', '❤️❤️'). Used
    only when --prefilter is on; it saves throughput, not money (local is free)."""
    t = (text or "").strip()
    if len(re.findall(r"[A-Za-z']+", t)) < 4:
        return True
    letters = sum(c.isalpha() for c in t)
    return letters < 0.4 * max(len(t), 1)


def _normalise(data: dict, model: str) -> dict:
    """Coerce a raw model response into a clean, taxonomy-valid record. The
    schema constrains both backends, but small local models occasionally drift,
    so we validate against the taxonomy and drop anything off-vocabulary."""
    tax = _taxonomy()
    ms_allowed = set(tax["meaning_sources"])
    stage_allowed = set(tax["life_stage"])
    reg_allowed = set(tax["sincerity_register"])
    age = data.get("age_band_guess")
    return {
        "meaning_sources": [t for t in (data.get("meaning_sources") or []) if t in ms_allowed],
        "themes": [str(t).lower().strip() for t in (data.get("themes") or []) if str(t).strip()][:8],
        "life_stage": data.get("life_stage") if data.get("life_stage") in stage_allowed else "unknown",
        "sincerity_register": data.get("sincerity_register") if data.get("sincerity_register") in reg_allowed else "reflective",
        "age_band_guess": str(age) if age not in (None, "", "null") else None,
        "essence": str(data.get("essence") or "").strip(),
        "pull_quote": _truncate_quote(str(data.get("pull_quote") or "")),
        "model": model,
    }


def enrich_item(client, model: str, item: Any, schema: dict) -> Optional[dict]:
    """One Claude API call. Returns a validated dict, or None on any failure."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(item)}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.APIError as e:
        log.warning("API error on item %s: %s", item["id"], e)
        return None

    try:
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, json.JSONDecodeError) as e:
        log.warning("unparseable response for item %s: %s", item["id"], e)
        return None

    return _normalise(data, model)


def enrich_item_local(model: str, item: Any, schema: dict) -> Optional[dict]:
    """One local (Ollama) call with the same schema. Free per item. Ollama
    enforces the json_schema via its structured-output `format` field."""
    try:
        import ollama
    except ImportError:
        log.error("ollama package not installed — run: uv sync")
        return None
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(item, max_chars=LOCAL_MAX_CHARS)},
            ],
            format=schema,
            # Cap input chars + output tokens so a long item can't overflow the
            # context and wedge the grammar decoder (the bug that hung the resume).
            options={"temperature": 0, "num_ctx": LOCAL_NUM_CTX, "num_predict": LOCAL_NUM_PREDICT},
        )
        data = json.loads(resp["message"]["content"])
    except Exception as e:  # connection refused, model not pulled, bad JSON, ...
        log.warning("local enrich failed for %s: %s", item["id"], type(e).__name__)
        return None

    return _normalise(data, model)


def run(conn, *, model: Optional[str] = None, limit: Optional[int] = None,
        force: bool = False, starred_only: bool = False,
        local: bool = False, prefilter: bool = False, workers: int = 1) -> dict:
    """Enrich un-enriched items (or, with force, re-enrich). Returns stats.

    backend: local Ollama model (free) when local=True, else the Anthropic API.
    workers>1 (local only) runs that many model calls concurrently — the GPU
    batches them — while all DB writes stay on this (main) thread.
    """
    schema = build_schema(_taxonomy())

    if local:
        model = model or LOCAL_MODEL
        def enrich(item):
            return enrich_item_local(model, item, schema)
    else:
        client = anthropic.Anthropic()
        model = model or DEFAULT_MODEL
        workers = 1  # API path stays sequential (its own rate limits)
        def enrich(item):
            return enrich_item(client, model, item, schema)

    if force or starred_only:
        # Re-pass: pull items (optionally only starred) regardless of enrichment.
        sql = (
            "SELECT i.* FROM items i "
            "LEFT JOIN curation c ON c.item_id = i.id "
            "WHERE i.text IS NOT NULL AND TRIM(i.text) != '' "
        )
        if starred_only:
            sql += "AND COALESCE(c.starred,0)=1 "
        sql += "ORDER BY i.collected_at"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        items = conn.execute(sql).fetchall()
    else:
        items = dbmod.get_unenriched_items(conn, limit=limit)

    kept = [it for it in items if not (prefilter and is_low_signal(it["text"]))]
    stats = {"attempted": 0, "enriched": 0, "skipped": 0,
             "filtered": len(items) - len(kept), "model": model, "workers": workers}
    total = len(kept)

    def _write(item, data) -> None:
        if data is None:
            stats["skipped"] += 1
            return
        dbmod.upsert_enrichment(
            conn, item_id=item["id"],
            meaning_sources=data["meaning_sources"], life_stage=data["life_stage"],
            age_band_guess=data.get("age_band_guess"), sincerity_register=data["sincerity_register"],
            themes=data["themes"], essence=data["essence"], pull_quote=data["pull_quote"], model=model,
        )
        conn.commit()
        stats["enriched"] += 1
        log.info("[%d/%d] %s -> %s / %s", stats["enriched"] + stats["skipped"], total,
                 item["id"], data["sincerity_register"], ",".join(data["meaning_sources"]) or "-")

    if workers > 1:
        import itertools
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cursor = iter(kept)
            while True:
                chunk = list(itertools.islice(cursor, workers * 4))
                if not chunk:
                    break
                for item, data in zip(chunk, pool.map(enrich, chunk)):
                    stats["attempted"] += 1
                    _write(item, data)
    else:
        for item in kept:
            stats["attempted"] += 1
            _write(item, enrich(item))

    return stats
