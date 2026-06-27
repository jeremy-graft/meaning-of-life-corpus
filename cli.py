"""Meaning-of-life corpus — CLI.

Pipeline:  collect (YouTube) -> store (SQLite) -> enrich (Claude) -> query/export/curate

    python cli.py initdb
    python cli.py collect --seeds config/seeds.yaml
    python cli.py enrich --limit 200
    python cli.py query --register raw_confession --tag family_children
    python cli.py export --starred-only --format md --out exports/starred.md
    python cli.py curate star <item_id>

Keys come from the environment (.env is auto-loaded): YOUTUBE_API_KEY, ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv

from db import db as dbmod

# The corpus is full of non-ASCII (curly quotes, em-dashes, ★). Force UTF-8 on
# stdout/stderr so printing never crashes on a legacy Windows cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

load_dotenv()

app = typer.Typer(add_completion=False, help=__doc__)
curate_app = typer.Typer(help="Mark starred / add notes / cluster.")
app.add_typer(curate_app, name="curate")

DB_OPT = typer.Option(str(dbmod.DEFAULT_DB_PATH), "--db", help="Path to the SQLite database.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _conn(db: str):
    conn = dbmod.get_conn(db)
    dbmod.init_db(conn)  # idempotent; safe to call every time
    return conn


@app.command()
def initdb(db: str = DB_OPT):
    """Create the database and tables (idempotent)."""
    conn = _conn(db)
    typer.echo(f"Initialized {db}")
    conn.close()


@app.command()
def collect(
    seeds: str = typer.Option("config/seeds.yaml", "--seeds", help="Seeds config (genre-seeds)."),
    db: str = DB_OPT,
    quota: int = typer.Option(9500, "--quota", help="Quota budget (units) to spend this run."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fetch raw items from YouTube into the database."""
    _setup_logging(verbose)
    from adapters import youtube  # imported lazily so other verbs don't need googleapiclient

    cfg = yaml.safe_load(Path(seeds).read_text(encoding="utf-8"))
    conn = _conn(db)
    stats = youtube.collect_seeds(conn, cfg, quota_budget=quota)
    conn.close()
    typer.echo(json.dumps(stats, indent=2))


@app.command("refetch-transcripts")
def refetch_transcripts(
    db: str = DB_OPT,
    delay: float = typer.Option(1.5, "--delay", help="Seconds between videos (politeness throttle)."),
    lang: str = typer.Option("en", "--lang", help="Preferred transcript language."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Re-attempt transcripts for videos missed on the first pass (e.g. IP-blocked)."""
    _setup_logging(verbose)
    from adapters import youtube

    conn = _conn(db)
    stats = youtube.refetch_missing_transcripts(conn, delay=delay, languages=[lang])
    conn.close()
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def enrich(
    db: str = DB_OPT,
    limit: Optional[int] = typer.Option(None, "--limit", help="Max items to enrich this run."),
    model: Optional[str] = typer.Option(None, "--model", help="Override the model id."),
    local: bool = typer.Option(False, "--local", help="Use the free local model (Ollama) instead of the API."),
    prefilter: bool = typer.Option(False, "--prefilter", help="Skip low-signal items (too short / emoji-only)."),
    workers: int = typer.Option(3, "--workers", help="Concurrent local model calls (local only; GPU batches them)."),
    force: bool = typer.Option(False, "--force", help="Re-enrich already-enriched items."),
    starred_only: bool = typer.Option(False, "--starred-only", help="Curated re-pass over starred items only."),
    curated: bool = typer.Option(False, "--curated", help="Use the stronger curated model (API)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Enrich un-enriched items (cached by item id). Local model by default-free; API for curated."""
    _setup_logging(verbose)
    from enrich import extract

    if local:
        chosen = model or extract.LOCAL_MODEL
    else:
        chosen = model or (extract.CURATED_MODEL if (curated or starred_only) else extract.DEFAULT_MODEL)
    conn = _conn(db)
    stats = extract.run(conn, model=chosen, limit=limit, force=force,
                        starred_only=starred_only, local=local, prefilter=prefilter, workers=workers)
    conn.close()
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def query(
    db: str = DB_OPT,
    tag: Optional[str] = typer.Option(None, help="Meaning-source or theme tag."),
    life_stage: Optional[str] = typer.Option(None, help="teen|young_adult|midlife|older|unknown"),
    register: Optional[str] = typer.Option(None, help="sincerity_register enum value."),
    genre_seed: Optional[str] = typer.Option(None, help="Filter by genre-seed name."),
    starred: bool = typer.Option(False, "--starred", help="Only starred items."),
    limit: int = typer.Option(25, help="Max rows."),
):
    """Pull fragments by tag, life-stage, register, or seed."""
    conn = _conn(db)
    rows = dbmod.query_items(
        conn, tag=tag, life_stage=life_stage, register=register,
        genre_seed=genre_seed, starred=(True if starred else None), limit=limit,
    )
    conn.close()
    if not rows:
        typer.echo("No matches.")
        raise typer.Exit()
    for r in rows:
        star = "★" if r["starred"] else " "
        sources = ", ".join(json.loads(r["meaning_sources_json"] or "[]"))
        typer.echo(
            f"{star} [{r['id']}] {r['life_stage']}/{r['sincerity_register']}  ({r['genre_seed']})\n"
            f"    sources: {sources or '-'}\n"
            f"    essence: {r['essence']}\n"
            f"    quote:   “{r['pull_quote']}”\n"
        )


@app.command()
def export(
    db: str = DB_OPT,
    starred_only: bool = typer.Option(False, "--starred-only", help="Export only starred items."),
    fmt: str = typer.Option("json", "--format", help="json | md"),
    out: Optional[str] = typer.Option(None, "--out", help="Output file (default: stdout)."),
    limit: int = typer.Option(1000, help="Max items."),
):
    """Export the (starred) set for curation. Favors essence/paraphrase over verbatim text."""
    conn = _conn(db)
    rows = dbmod.query_items(conn, starred=(True if starred_only else None), limit=limit)
    conn.close()

    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "genre_seed": r["genre_seed"],
            "kind": r["kind"],
            "url": r["url"],  # provenance only
            "life_stage": r["life_stage"],
            "age_band_guess": r["age_band_guess"],
            "sincerity_register": r["sincerity_register"],
            "meaning_sources": json.loads(r["meaning_sources_json"] or "[]"),
            "themes": json.loads(r["themes_json"] or "[]"),
            "essence": r["essence"],          # art-facing: paraphrase
            "pull_quote": r["pull_quote"],    # provenance only, < 15 words
            "cluster": r["cluster"],
            "notes": r["notes"],
        })

    if fmt == "json":
        text = json.dumps(records, indent=2, ensure_ascii=False)
    elif fmt == "md":
        lines = ["# Meaning-of-life corpus — curated export\n"]
        for rec in records:
            lines.append(f"## {rec['sincerity_register']} · {rec['life_stage']} · _{rec['genre_seed']}_")
            lines.append(f"\n{rec['essence']}\n")
            lines.append(f"> “{rec['pull_quote']}”")
            lines.append(f"\n`{', '.join(rec['meaning_sources']) or '-'}`"
                         f"{(' · ' + ', '.join(rec['themes'])) if rec['themes'] else ''}")
            lines.append(f"\n<sub>provenance: {rec['url']}</sub>\n")
        text = "\n".join(lines)
    else:
        typer.echo(f"Unknown format: {fmt}", err=True)
        raise typer.Exit(code=1)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {len(records)} items to {out}")
    else:
        typer.echo(text)


@app.command()
def stats(db: str = DB_OPT):
    """Show corpus counts."""
    conn = _conn(db)
    typer.echo(json.dumps(dbmod.counts(conn), indent=2))
    conn.close()


@curate_app.command("star")
def curate_star(item_id: str, db: str = DB_OPT):
    """Star an item."""
    conn = _conn(db)
    dbmod.set_star(conn, item_id, True)
    conn.commit(); conn.close()
    typer.echo(f"Starred {item_id}")


@curate_app.command("unstar")
def curate_unstar(item_id: str, db: str = DB_OPT):
    """Unstar an item."""
    conn = _conn(db)
    dbmod.set_star(conn, item_id, False)
    conn.commit(); conn.close()
    typer.echo(f"Unstarred {item_id}")


@curate_app.command("note")
def curate_note(item_id: str, note: str, db: str = DB_OPT):
    """Attach a note to an item."""
    conn = _conn(db)
    dbmod.set_note(conn, item_id, note)
    conn.commit(); conn.close()
    typer.echo(f"Noted {item_id}")


@curate_app.command("cluster")
def curate_cluster(item_id: str, cluster: str, db: str = DB_OPT):
    """Assign an item to a cluster."""
    conn = _conn(db)
    dbmod.set_cluster(conn, item_id, cluster)
    conn.commit(); conn.close()
    typer.echo(f"Clustered {item_id} -> {cluster}")


if __name__ == "__main__":
    app()
