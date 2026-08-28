# Atlas of Meanings: labels v1

19,464 judgements about how people talk about meaning and endurance, across eight
languages, made on public YouTube comments gathered from moments where the question
tends to ask itself: a hospice channel, a sobriety anniversary, a faith coming apart,
a first child, a terminal diagnosis.

## What is here, and what is not

`atlas-labels-v1.jsonl`, one JSON object per line. **The comment text is not
included.** Each row carries the platform's comment id, so anyone can fetch the text
themselves through the YouTube API. This is the ordinary practice for annotation
corpora built on someone else's platform, and it has a second virtue: a comment its
author later deletes stays deleted, rather than living on in a file we handed out.

## Fields

| field | meaning |
|---|---|
| `id` | YouTube comment id, for rehydration |
| `round` | which labelling round; see below |
| `purpose` | what that round was measuring |
| `lang` | language the seam was searched in, not per-comment detection |
| `situation` | the seam: what had happened to the speaker |
| `frame` | `random` (unbiased, safe for prevalence) or `stratified` / `active` (enriched for rare classes, **never** use these for prevalence) |
| `stance` | `asserts`, `none`, `doesnt_know`, `denies`, `make_your_own` |
| `sources` | up to three of fifteen positive sources of meaning, only when `stance` is `asserts` |
| `register` | `raw_confession`, `reflective`, `advice_giving`, `performed_brand`, `joke_deflection` |
| `strategies` | coping strategies, in the `hope` round only |
| `conceals` | whether the speaker describes hiding their feelings from others |

## The frame field matters more than anything else here

Three of the five rounds are deliberately **not** representative. They oversample
rare classes so a classifier has something to learn from. Only rows marked
`frame: random` estimate how common anything actually is. Mixing the two produces
confident nonsense, which is a mistake this project made and had to correct
publicly.

## Rounds

| round | n | what it was for |
|---|---|---|
| `gold` | 2,086 | meaning, English, first round |
| `goldml` | 2,030 | meaning, seven languages, pilot |
| `gold3` | 9,750 | meaning, seven languages, scaled |
| `goldact` | 3,198 | meaning, classifier-enriched sample |
| `hope` | 2,400 | coping in irreversible situations |

Languages are close to balanced: Arabic 2,518, Indonesian 2,494, Hindi 2,492,
Spanish 2,465, Portuguese 2,432, Turkish 2,415, English 2,386, Japanese 2,262.

## Taxonomies

Meaning uses fifteen positive sources plus a separate stance field, in
`enrich/taxonomy.yaml`. Coping uses **Carver's Brief COPE (1997)**, the standard
instrument, plus five categories added for irreversible loss, of which the most
important is **continuing bonds** (Klass, Silverman and Nickman, 1996). See
`enrich/hope_taxonomy.yaml`.

## Who made these judgements

Not people. Large language models, working one comment at a time against written
codebooks, with a human designing the instrument, auditing samples and forcing the
corrections. The labels are good but they are not a trained human coder, and no
inter-rater reliability against human coders has been established. Treat them as a
strong first pass, not as ground truth.

Known weaknesses, measured rather than guessed:

- `denies` and `make_your_own` are nearly empty classes. In 800 randomly drawn
  comments, zero were genuine expressions of meaninglessness. An earlier automated
  pass claimed a fifth of the corpus denied that life has meaning; that was wrong.
- `humour` in the coping round is 0.3%. A keyword approach put it far higher by
  counting people who enjoyed a video.
- Religion detection by keyword reaches F1 0.83 against these labels, with a
  consistent seven point overcount.

## Ethics

Comments are public. Author names were hashed at collection and the originals were
never stored, so nothing here links a judgement to a person. No text is
distributed. Several of these situations are among the most painful a person can
be in, and the material should be handled accordingly.

## Citation

Atlas of Meanings, labels v1. https://atlasofmeanings.com
