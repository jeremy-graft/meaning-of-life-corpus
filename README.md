# Meaning-of-life corpus

A curated, queryable corpus of **fragments where people articulate what gives
life meaning**, harvested from **YouTube only**. This is an art project, not a
census — the point is *divergence and contradiction* (the 14-year-old next to
the 70-year-old; the gap between what someone performs and what bleeds
through), so the system **preserves mediation as data** rather than averaging it
into the usual flattened word cloud.

Two non-negotiables baked into the design:

1. **Seam-mining, not keyword soup.** We do not search "meaning of life" and
   scrape. We harvest the *genres* where people involuntarily say what life
   means — hospice reflections, deathbed regrets, eulogies, sobriety
   milestones, letters to a younger self (see [`config/seeds.yaml`](config/seeds.yaml)).
2. **Capture the frame.** Every item stores its `genre_seed` and a model-judged
   `sincerity_register` (raw_confession … performed_brand) as first-class
   fields — they are what let the art find the mask-vs-face contrast.

```
collect (YouTube) ──▶ store (SQLite) ──▶ enrich (local model, free, or Claude API) ──▶ query / export / curate
```

Enrichment has **two interchangeable backends** behind one schema: a free local
model (Ollama) for bulk volume, and the Claude API for a curated re-pass on the
subset you star. Same prompt, same strict-JSON taxonomy, same database.

## Layout

```
adapters/youtube.py     YouTube Data API v3 adapter (quota-aware, fails soft)
adapters/transcript.py  spoken-content transcripts (youtube-transcript-api, throttled)
db/schema.sql, db.py    one SQLite database: sources → items → enrichment, + curation
enrich/extract.py       enrichment → strict JSON; local + API backends; taxonomy.yaml (editable)
config/seeds.yaml       the genre-seeds (editable)
cli.py                  initdb / collect / refetch-transcripts / enrich / query / export / curate / stats
```

Items come in three `kind`s: `comment` (the richest, most unguarded seam),
`video_description`, and `caption` (the spoken transcript — what is *said* in
the video).

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
[`pyproject.toml`](pyproject.toml).

```bash
uv sync                       # creates .venv and installs deps
cp .env.example .env          # then fill in the keys
```

Keys (read from the environment; `.env` is auto-loaded):

- `YOUTUBE_API_KEY` — Google Cloud Console → enable **YouTube Data API v3** →
  create an API key (free, ~5 min). **Required** for `collect`.
- `ANTHROPIC_API_KEY` — https://console.anthropic.com. Only needed for the
  **API** enrichment backend (the curated re-pass). Not needed for `--local`.

**For free local enrichment** install [Ollama](https://ollama.com) and pull the
workhorse model once:

```bash
winget install Ollama.Ollama      # or download from ollama.com
ollama pull qwen2.5:7b            # ~4.7 GB; runs on an 8 GB GPU
```

## Usage

```bash
uv run cli.py initdb
uv run cli.py collect --seeds config/seeds.yaml      # fetch comments + descriptions into corpus.db
uv run cli.py refetch-transcripts                    # throttled spoken-content pass (separate; see below)
uv run cli.py stats

uv run cli.py enrich --local                         # FREE bulk enrichment via Ollama (qwen2.5:7b)
uv run cli.py enrich --local --prefilter             # ...skipping low-signal items (too short / emoji)
uv run cli.py enrich --starred-only --curated        # curated API re-pass over starred items (claude-opus-4-8)

uv run cli.py query --register raw_confession --tag family_children
uv run cli.py curate star <item_id>
uv run cli.py export --starred-only --format md --out exports/starred.md
```

The intended flow: **collect a lot (free) → `enrich --local` everything (free) →
star what hits → `enrich --starred-only --curated` for the premium pass.** The
Anthropic API is only ever spent on the fragments you've already chosen.

Get the end-to-end pipeline producing a real corpus first, then refine seeds and
taxonomy — `config/seeds.yaml` and `enrich/taxonomy.yaml` are meant to be edited.

## How the pieces work

**Collection** (`adapters/youtube.py`). Per genre-seed: `search.list` → for each
video `videos.list` → `commentThreads.list` (top-level + inline replies — the
unguarded answers live in comments). Quota spend is tracked per call against a
budget (`--quota`, default 9500 of the ~10k/day free quota) and the run **fails
soft** when exhausted rather than crashing mid-collection.

**Storage** (`db/`). One SQLite database, four tables. All writes are idempotent:
re-running `collect` never duplicates an item (INSERT OR IGNORE on the id), and
never clobbers existing enrichment.

**Enrichment** (`enrich/extract.py`). One model call per item returns **strict
JSON** — a `json_schema` built dynamically from `taxonomy.yaml` (so the
controlled vocabulary stays editable while output stays valid), enforced via the
API's `output_config.format` or Ollama's structured-output `format`. Two
backends share the prompt and schema:

- **`--local`** — a model served by [Ollama](https://ollama.com) on your machine
  (default `qwen2.5:7b`). **Free per item** — the volume workhorse. Output is
  validated against the taxonomy (small models occasionally drift). Needs Ollama
  running and the model pulled; no API key.
- **API (default)** — Claude (`claude-haiku-4-5`, or `claude-opus-4-8` with
  `--curated`). Costs per item; reserve it for the curated re-pass.

Parsing is defensive: any failure logs and skips, never crashing the batch.
Enrichment is **cached by item id** (skipped unless `--force`), and the
`enrichment.model` column records which model tagged each item. `--prefilter`
skips obviously low-signal items (too short, emoji-only) to save throughput.

## Privacy & ethics (enforced in code)

- **Pseudonymized on ingest.** Author handles are SHA-256 hashed into
  `author_pseudonym` (`db.pseudonymize`); the raw handle is never stored. The
  source `url` is kept for provenance only.
- **No verbatim republishing at length.** `pull_quote` is capped under 15 words
  and exists for provenance; exports favor the paraphrased `essence`.
- **API-only, within ToS.** YouTube Data API v3 only — no scraping fallbacks.

## A note on transcripts (spoken content)

What's *said* in the videos matters — a hospice nurse on camera, a commencement
speech — so transcripts are captured and stored as `caption` items.

There's a deliberate tradeoff. The official Data API **cannot** download captions
for videos you don't own: `captions.download` requires the channel owner's OAuth
and 403s otherwise. So transcripts come from
[`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/),
which reads YouTube's internal transcript endpoint — **the one source in this
pipeline outside the official API lane** (a ToS gray area, enabled by choice).
No API key, no audio download, no Data API quota.

**Transcripts are fetched in a *separate, throttled* pass, not during `collect`.**
Fetching them inline during a big collection sends a burst that YouTube
IP-blocks almost immediately. So `collect_transcripts` defaults to **false**, and
`uv run cli.py refetch-transcripts` does a throttled (delay + backoff) pass over
every video that has a description but no caption — idempotent, so re-run it
after a cooldown to pick up more. Even so, a single IP gets rate-limited; for
full coverage at scale, configure a rotating proxy (the library supports it).

Each transcript is stored whole; one `essence`/`pull_quote` per talk. We
evaluated chunking long transcripts into passages and found it unnecessary — the
single essence faithfully captures single-speaker talks (which most are).
