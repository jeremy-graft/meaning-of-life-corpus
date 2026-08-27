"""Works enrichment — the *canon* layer's counterpart to crowd enrichment.

A crowd comment states (at most) what gives ITS author life meaning. A work of
art ARGUES: it affirms some sources of meaning, indicts others as hollow, and
often leaves the question deliberately open. So a work gets three axes over the
SAME meaning-source taxonomy the crowd uses —

    affirms     — what the work presents as genuinely giving life meaning
    rejects     — what it exposes as false / hollow / a trap
    unresolved  — raised and deliberately left ambiguous

plus a single `stance` (its overall posture toward meaning) and a short essence.

Input is the noisy Wikipedia Themes/Analysis prose (which also carries
prose-quality criticism and reception history); the model is told to read
THROUGH that noise and report only what the work says about how to live.

Free local model (Ollama), same backend as crowd enrichment. Only ~dozens of
works, so we can afford a big context window and read the whole interpretation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from enrich import extract  # reuse taxonomy loader + local model id

log = logging.getLogger("enrich")

# Works are few but their interpretations are long — give the model room to read.
WORKS_MAX_CHARS = 12000
WORKS_NUM_CTX = 8192
WORKS_NUM_PREDICT = 1024
WORKS_TEMPERATURE = 0.3
WORKS_TAG_VERSION = "w2"

# The work's overall posture toward the question of meaning.
STANCE = [
    "affirming",    # presents a positive source of meaning as real
    "critical",     # indicts false sources; meaning is elsewhere / withheld
    "absurdist",    # no inherent meaning; you make/choose your own
    "tragic",       # meaning exists but is lost, too late, or costly
    "ironic",       # undercuts its own apparent answer
    "ambiguous",    # deliberately leaves the question open
    "nihilist",     # denies meaning outright
]

PULL_THEME_MAX_WORDS = 14

WORKS_SYSTEM_PROMPT = (
    "You are a close-reader mapping what works of art SAY about the meaning of "
    "human life, for an art corpus. You are given the interpretive 'Themes / "
    "Analysis' prose about ONE film or novel.\n\n"
    "IMPORTANT — the prose is noisy. It often mixes three things:\n"
    "  1. what the work says about how to live / what makes life meaningful  <- USE THIS\n"
    "  2. literary-quality criticism (is the prose good, is it well-made)     <- IGNORE\n"
    "  3. reception history (how critics/audiences reacted, sales, awards)    <- IGNORE\n"
    "Report ONLY category 1.\n\n"
    "A work does not just state a meaning — it ARGUES. Map its argument onto the "
    "meaning-source taxonomy along three axes:\n"
    "  affirms    — sources the work presents as genuinely giving life meaning\n"
    "  rejects    — sources the work ACTIVELY EXPOSES as false, hollow, or a trap\n"
    "  unresolved — sources it raises but deliberately leaves ambiguous\n\n"
    "CRITICAL GATE — do NOT over-fill these arrays. A source is tagged ONLY if the "
    "interpretation actually discusses the work's stance toward it. If the work is "
    "simply SILENT about a source, it goes in NONE of the arrays. Silence is not "
    "rejection. Most works actively reject only 0-2 sources; if you list more than 3 "
    "rejects you are almost certainly mistaking silence for indictment — cut it back. "
    "For `rejects`, prefer the FALSE-IDOLS vocabulary (consumerism_materialism, "
    "status_prestige, power_domination, conformity, control_rationalism, "
    "escapism_distraction, vanity_image, hedonism_numbing, achievement_striving); you "
    "MAY also reject a normally-positive source if the work argues it is hollow. Only "
    "tag what the work ACTIVELY indicts — silence tags nothing.\n"
    "Examples of the gate:\n"
    "  A film about a man learning to help others before he dies -> "
    "affirms:[service_others, growth_self]  rejects:[]  (it doesn't attack anything, it's silent)\n"
    "  A novel indicting empty social success, urging spiritual awakening -> "
    "affirms:[religion_transcendence]  rejects:[craft_work]  (only if it names status/work as hollow)\n"
    "  A film with no clear meaning-argument, mostly about its own style -> "
    "states_meaning:false, all arrays empty\n"
    "A source belongs to at most ONE axis. Do not force an 'affirms' — many works affirm nothing.\n\n"
    "stance = the work's single overall posture. Use 'ambiguous' ONLY when the work "
    "genuinely refuses to resolve; if it leans, pick the leaning stance (affirming, "
    "critical, absurdist, tragic, ironic, nihilist).\n"
    "essence = one or two sentences, your own words: what this work says about the "
    "meaning of life.\n"
    "pull_theme = a crisp claim ABOUT the work, UNDER 15 words. NOT a quote from "
    "the work (quotes mislead — villains and ironists speak in works).\n"
    "Return ONLY the structured object. No prose, no markdown."
)


def build_works_schema(taxonomy: dict) -> dict:
    pos = taxonomy["meaning_sources"]
    # A work can reject a false idol OR argue a normally-positive source is hollow.
    reject_vocab = list(taxonomy["false_idols"]) + list(pos)
    def axis(cap, vocab):
        return {"type": "array", "maxItems": cap, "items": {"type": "string", "enum": vocab}}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "states_meaning": {"type": "boolean"},
            "affirms": axis(4, pos),
            "rejects": axis(3, reject_vocab),  # a work indicts few things; a long list = mistaking silence for attack
            "unresolved": axis(3, pos),
            "stance": {"type": "string", "enum": STANCE},
            "essence": {"type": "string"},
            "pull_theme": {"type": "string"},
        },
        "required": ["states_meaning", "affirms", "rejects", "unresolved",
                     "stance", "essence", "pull_theme"],
    }


def _user_prompt(work: Any) -> str:
    text = work["themes_text"] or ""
    if len(text) > WORKS_MAX_CHARS:
        text = text[:WORKS_MAX_CHARS] + " …[truncated]"
    medium = work["medium"]
    return (
        f"This is the interpretive 'Themes / Analysis' prose about the {medium} "
        f"\"{work['title']}\".\n\n--- interpretation ---\n{text}\n--- end ---"
    )


def _truncate(s: str, max_words: int) -> str:
    w = s.split()
    return s if len(w) <= max_words else " ".join(w[:max_words]) + "…"


def _normalise(data: dict, taxonomy: dict, model: str) -> dict:
    pos = set(taxonomy["meaning_sources"])
    reject_ok = pos | set(taxonomy["false_idols"])
    def _axis(key, allowed):
        return list(dict.fromkeys(t for t in (data.get(key) or []) if t in allowed))
    affirms = _axis("affirms", pos)
    rejects = _axis("rejects", reject_ok)
    unresolved = _axis("unresolved", pos)
    # Enforce the "each source at most one axis" rule: affirms > rejects > unresolved.
    rejects = [t for t in rejects if t not in affirms]
    unresolved = [t for t in unresolved if t not in affirms and t not in rejects]
    stance = data.get("stance") if data.get("stance") in set(STANCE) else "ambiguous"
    return {
        "states_meaning": bool(data.get("states_meaning")),
        "affirms": affirms, "rejects": rejects, "unresolved": unresolved,
        "stance": stance,
        "essence": str(data.get("essence") or "").strip(),
        "pull_theme": _truncate(str(data.get("pull_theme") or "").strip(), PULL_THEME_MAX_WORDS),
        "model": model,
    }


def _enrich_local(model: str, work: Any, schema: dict, taxonomy: dict) -> Optional[dict]:
    try:
        import ollama
    except ImportError:
        log.error("ollama package not installed — run: uv sync")
        return None
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": WORKS_SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(work)},
            ],
            format=schema,
            options={"temperature": WORKS_TEMPERATURE, "num_ctx": WORKS_NUM_CTX,
                     "num_predict": WORKS_NUM_PREDICT},
        )
        data = json.loads(resp["message"]["content"])
    except Exception as e:
        log.warning("works enrich failed for %r: %s", work["title"], type(e).__name__)
        return None
    return _normalise(data, taxonomy, model)


def ensure_table(conn) -> None:
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS work_enrichment ("
        " work_id TEXT PRIMARY KEY, states_meaning INT, stance TEXT,"
        " affirms_json TEXT, rejects_json TEXT, unresolved_json TEXT,"
        " essence TEXT, pull_theme TEXT, model TEXT, enriched_at TEXT);"
    )
    conn.commit()


def run(conn, *, model: Optional[str] = None, retag: bool = False,
        limit: Optional[int] = None) -> dict:
    """Enrich each collected work. Resumable: with retag, re-does only works not
    already on the current method version; otherwise does the un-enriched ones."""
    from datetime import datetime, timezone

    taxonomy = extract._taxonomy()
    schema = build_works_schema(taxonomy)
    model = model or extract.LOCAL_MODEL
    store_model = f"{model}#{WORKS_TAG_VERSION}"
    ensure_table(conn)

    if retag:
        sql = ("SELECT w.* FROM works w LEFT JOIN work_enrichment e ON e.work_id = w.id "
               "WHERE e.work_id IS NULL OR e.model != ? ORDER BY w.title")
        rows = conn.execute(sql, (store_model,)).fetchall()
    else:
        sql = ("SELECT w.* FROM works w LEFT JOIN work_enrichment e ON e.work_id = w.id "
               "WHERE e.work_id IS NULL ORDER BY w.title")
        rows = conn.execute(sql).fetchall()
    if limit is not None:
        rows = rows[:limit]

    stats = {"attempted": 0, "enriched": 0, "skipped": 0, "model": store_model}
    for w in rows:
        stats["attempted"] += 1
        data = _enrich_local(model, w, schema, taxonomy)
        if data is None:
            stats["skipped"] += 1
            continue
        conn.execute(
            "INSERT OR REPLACE INTO work_enrichment(work_id, states_meaning, stance,"
            " affirms_json, rejects_json, unresolved_json, essence, pull_theme, model, enriched_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (w["id"], int(data["states_meaning"]), data["stance"],
             json.dumps(data["affirms"]), json.dumps(data["rejects"]),
             json.dumps(data["unresolved"]), data["essence"], data["pull_theme"],
             store_model, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        stats["enriched"] += 1
        log.info("[%d/%d] %s -> %s  A:%s R:%s", stats["enriched"] + stats["skipped"],
                 len(rows), w["title"], data["stance"],
                 ",".join(data["affirms"]) or "-", ",".join(data["rejects"]) or "-")
    return stats
