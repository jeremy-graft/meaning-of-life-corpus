-- One SQLite database for the whole corpus.
-- Four tables: sources -> items -> enrichment, plus curation.
-- The frame (genre_seed, sincerity_register, life_stage) is first-class, not metadata.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id            TEXT PRIMARY KEY,   -- channel_id (the thing that published)
    channel_id    TEXT,
    channel_title TEXT,
    url           TEXT
);

CREATE TABLE IF NOT EXISTS items (
    id               TEXT PRIMARY KEY,   -- stable per-item id (see adapters/youtube.py)
    source_id        TEXT REFERENCES sources(id),
    kind             TEXT NOT NULL CHECK (kind IN ('video_description','caption','comment')),
    url              TEXT,
    author_pseudonym TEXT,               -- hashed handle; raw handle is NEVER stored
    text             TEXT,
    published_at     TEXT,
    lang             TEXT,
    genre_seed       TEXT,               -- which seam this came from (config/seeds.yaml)
    raw_json         TEXT,
    collected_at     TEXT
);

CREATE TABLE IF NOT EXISTS enrichment (
    item_id              TEXT PRIMARY KEY REFERENCES items(id),
    meaning_sources_json TEXT,           -- JSON array of taxonomy tags
    life_stage           TEXT,
    age_band_guess       TEXT,
    sincerity_register   TEXT,           -- single enum (the mask-vs-face field)
    themes_json          TEXT,           -- JSON array of free tags
    essence              TEXT,           -- one sentence, model's words
    pull_quote           TEXT,           -- verbatim, < 15 words, provenance only
    model                TEXT,
    created_at           TEXT
);

CREATE TABLE IF NOT EXISTS curation (
    item_id TEXT PRIMARY KEY REFERENCES items(id),
    starred INTEGER NOT NULL DEFAULT 0,
    notes   TEXT,
    cluster TEXT
);

CREATE INDEX IF NOT EXISTS idx_items_genre     ON items(genre_seed);
CREATE INDEX IF NOT EXISTS idx_items_kind      ON items(kind);
CREATE INDEX IF NOT EXISTS idx_enrich_register ON enrichment(sincerity_register);
CREATE INDEX IF NOT EXISTS idx_enrich_stage    ON enrichment(life_stage);
CREATE INDEX IF NOT EXISTS idx_curation_star   ON curation(starred);
