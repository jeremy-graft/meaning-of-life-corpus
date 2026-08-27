"""SQLite access layer for the meaning-of-life corpus.

One database, four tables (see schema.sql). All writes are idempotent:
re-running `collect` never duplicates an item, and never clobbers enrichment
that already exists. Author handles are hashed on the way in and the raw handle
is never stored — see pseudonymize().
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "corpus.db"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def pseudonymize(handle: Optional[str]) -> str:
    """Hash a raw author handle into a stable pseudonym. The raw handle is
    never persisted anywhere — only this digest and the source url are kept."""
    if not handle:
        return "anon"
    digest = hashlib.sha256(handle.encode("utf-8")).hexdigest()
    return f"yt_{digest[:12]}"


def get_conn(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # wait, don't error, on brief write contention
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent, additive migrations for DBs created before a column existed.
    SQLite has no ADD COLUMN IF NOT EXISTS, so we check pragma first."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(enrichment)")}
    if "meaning_stance" not in cols:
        conn.execute("ALTER TABLE enrichment ADD COLUMN meaning_stance TEXT")


# --- writes -----------------------------------------------------------------

def upsert_source(conn, *, id: str, channel_id: str, channel_title: str, url: str) -> None:
    conn.execute(
        "INSERT INTO sources(id, channel_id, channel_title, url) VALUES(?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET channel_title=excluded.channel_title, url=excluded.url",
        (id, channel_id, channel_title, url),
    )


def insert_item(
    conn,
    *,
    id: str,
    source_id: Optional[str],
    kind: str,
    url: Optional[str],
    author_pseudonym: str,
    text: Optional[str],
    published_at: Optional[str],
    lang: Optional[str],
    genre_seed: str,
    raw_json: Any = None,
) -> bool:
    """Insert one item. Returns True if it was newly inserted, False if it
    already existed (INSERT OR IGNORE on the primary key)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO items"
        "(id, source_id, kind, url, author_pseudonym, text, published_at, lang, genre_seed, raw_json, collected_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            id, source_id, kind, url, author_pseudonym, text, published_at, lang, genre_seed,
            json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
            utcnow(),
        ),
    )
    return cur.rowcount > 0


def upsert_enrichment(
    conn,
    *,
    item_id: str,
    meaning_sources: list[str],
    meaning_stance: str,
    life_stage: str,
    age_band_guess: Optional[str],
    sincerity_register: str,
    themes: list[str],
    essence: str,
    pull_quote: str,
    model: str,
) -> None:
    conn.execute(
        "INSERT INTO enrichment"
        "(item_id, meaning_sources_json, meaning_stance, life_stage, age_band_guess, sincerity_register, "
        " themes_json, essence, pull_quote, model, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(item_id) DO UPDATE SET "
        "meaning_sources_json=excluded.meaning_sources_json, meaning_stance=excluded.meaning_stance, "
        "life_stage=excluded.life_stage, "
        "age_band_guess=excluded.age_band_guess, sincerity_register=excluded.sincerity_register, "
        "themes_json=excluded.themes_json, essence=excluded.essence, pull_quote=excluded.pull_quote, "
        "model=excluded.model, created_at=excluded.created_at",
        (
            item_id, json.dumps(meaning_sources, ensure_ascii=False), meaning_stance, life_stage, age_band_guess,
            sincerity_register, json.dumps(themes, ensure_ascii=False), essence, pull_quote, model, utcnow(),
        ),
    )


def ensure_curation(conn, item_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO curation(item_id, starred) VALUES(?, 0)", (item_id,))


def set_star(conn, item_id: str, starred: bool = True) -> None:
    ensure_curation(conn, item_id)
    conn.execute("UPDATE curation SET starred=? WHERE item_id=?", (1 if starred else 0, item_id))


def set_note(conn, item_id: str, note: str) -> None:
    ensure_curation(conn, item_id)
    conn.execute("UPDATE curation SET notes=? WHERE item_id=?", (note, item_id))


def set_cluster(conn, item_id: str, cluster: str) -> None:
    ensure_curation(conn, item_id)
    conn.execute("UPDATE curation SET cluster=? WHERE item_id=?", (cluster, item_id))


# --- reads ------------------------------------------------------------------

def get_unenriched_items(conn, limit: Optional[int] = None) -> list[sqlite3.Row]:
    """Items with non-empty text that have no enrichment row yet."""
    sql = (
        "SELECT i.* FROM items i LEFT JOIN enrichment e ON e.item_id = i.id "
        "WHERE e.item_id IS NULL AND i.text IS NOT NULL AND TRIM(i.text) != '' "
        "ORDER BY i.collected_at"
    )
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (int(limit),)).fetchall()
    return conn.execute(sql).fetchall()


def query_items(
    conn,
    *,
    tag: Optional[str] = None,
    life_stage: Optional[str] = None,
    register: Optional[str] = None,
    genre_seed: Optional[str] = None,
    starred: Optional[bool] = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """Join items + enrichment + curation, filtered by frame fields. `tag`
    matches either a meaning_source or a free theme (via json_each)."""
    where: list[str] = []
    params: list[Any] = []
    if tag:
        where.append(
            "(EXISTS (SELECT 1 FROM json_each(e.meaning_sources_json) WHERE value = ?) "
            " OR EXISTS (SELECT 1 FROM json_each(e.themes_json) WHERE value = ?))"
        )
        params += [tag, tag]
    if life_stage:
        where.append("e.life_stage = ?")
        params.append(life_stage)
    if register:
        where.append("e.sincerity_register = ?")
        params.append(register)
    if genre_seed:
        where.append("i.genre_seed = ?")
        params.append(genre_seed)
    if starred is not None:
        where.append("COALESCE(c.starred, 0) = ?")
        params.append(1 if starred else 0)

    sql = (
        "SELECT i.id, i.kind, i.url, i.genre_seed, i.author_pseudonym, i.text, "
        "e.meaning_sources_json, e.themes_json, e.life_stage, e.age_band_guess, "
        "e.sincerity_register, e.essence, e.pull_quote, "
        "COALESCE(c.starred,0) AS starred, c.notes, c.cluster "
        "FROM items i JOIN enrichment e ON e.item_id = i.id "
        "LEFT JOIN curation c ON c.item_id = i.id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY starred DESC, i.collected_at DESC LIMIT ?"
    params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def counts(conn) -> dict[str, int]:
    def one(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    def maybe(sql: str) -> int:
        try:
            return one(sql)
        except sqlite3.OperationalError:  # table not created yet
            return 0

    return {
        "sources": one("SELECT COUNT(*) FROM sources"),
        "items": one("SELECT COUNT(*) FROM items"),
        "comments": one("SELECT COUNT(*) FROM items WHERE kind='comment'"),
        "descriptions": one("SELECT COUNT(*) FROM items WHERE kind='video_description'"),
        "captions": one("SELECT COUNT(*) FROM items WHERE kind='caption'"),
        "enriched": one("SELECT COUNT(*) FROM enrichment"),
        "starred": one("SELECT COUNT(*) FROM curation WHERE starred=1"),
        "works": maybe("SELECT COUNT(*) FROM works"),
        "works_enriched": maybe("SELECT COUNT(*) FROM work_enrichment"),
    }
