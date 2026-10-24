# Retrieval evaluation harness

Measures whether indexing a note's `name + description` as the retrieval key
(one vector per note) beats the service's current behaviour of chunking the
note body at `CHUNK_SIZE`/`CHUNK_OVERLAP` (several vectors per note) — and
whether holding **both** beats either.

That question is the one `nescio-ai`'s ADR 0005 answers from the *shape* of the
corpus rather than from measurement, and which its own "to re-verify" section
flags as unmeasured. This harness is what settles it.

## The three strategies

| Strategy | Units per note | What it is |
| :--- | :--- | :--- |
| `chunk` | several | `chunk_text(body)`, exactly as `app/api/v1/ingest.py` does it |
| `summary` | one | `name` + `\n` + `description` — ADR 0005's proposal |
| `hybrid` | several + one | both of the above in **one** index, ranked together |

`hybrid` is a *composite*: it embeds nothing of its own, and is built by
unioning the two primitive indexes (`eval/retrieval.py::union_index`). That is
a hard requirement, not a convenience — the embedder is one serial HTTP call
per unit, so a third pass over a real corpus would be ~2,800 calls to
recompute vectors already in memory. It is scored as **one shared cosine
ranking**, not a score fusion: both unit types come from the same model under
the same metric, so there is no scale mismatch for a fusion to correct, and one
ordering is what "store both in one table" would mean for `/search`.

## This is not a test suite

`tests/` is hermetic, deterministic, and fakes the embedder. This harness
**requires the real embedding model**, because a hash-derived vector has no
semantics and so cannot tell you which strategy retrieves better *meaning* —
which is the entire question. It is therefore a hand-run tool, it is not in CI,
and it imports nothing from `tests/`. The long version is in
`eval/__init__.py`; read it before changing anything here.

## Running it

```bash
# Synthetic fixture corpus, real model. Needs a reachable embedding backend.
uv run python -m eval

# Prove the plumbing with no model available. Numbers are NOT a finding.
uv run python -m eval --embedder stub

# A real brain directory. Keep its query set out of the repo.
uv run python -m eval --corpus /path/to/brain --queries eval/private/brain-queries.yaml

# A hand-maintained corpus needs BOTH frontmatter reads -- see below.
uv run python -m eval --corpus /path/to/brain \
    --queries eval/private/brain-queries.yaml --frontmatter strict

uv run python -m eval --help
```

Reports land in `eval-out/` as `<timestamp>-<corpus>-<embedder>.md` and
`.json`. That directory is gitignored.

Prerequisites for a `real` run: an embedding backend reachable at the
configured `OLLAMA_URL` serving `OLLAMA_MODEL` at `EMBEDDING_DIMENSION`
dimensions, or `EMBEDDING_BACKEND=local` with the `local-embeddings` extra
installed. The harness needs **no** database — it holds vectors in memory and
scans them exactly — but `app.config` requires `DATABASE_URL` to be set at
import time, so `eval/appenv.py` supplies a placeholder pointing at a closed
port.

## Privacy

This repository is **public**. The real brain contains third-party client
project names and internal detail. `.gitignore` excludes `eval-out/`,
`eval/private/`, `eval/corpora/real/` and `eval/**/*.local.yaml`; keep real
corpora and real query sets inside those paths. Nothing derived from a private
brain may be committed — including a report that quotes note names.

## Layout

| Path | What it is |
| :--- | :--- |
| `__main__.py` | CLI entry point (`python -m eval`) |
| `appenv.py` | Env prep that must precede any `app.` import |
| `corpus.py` | Markdown + YAML-frontmatter note loading, corpus stats |
| `queries.py` | Query-set loading and validation |
| `strategies.py` | The two indexing strategies behind one interface |
| `embedders.py` | The real embedder, and the clearly-labelled stub |
| `retrieval.py` | In-memory index and exact similarity scan |
| `vectors.py` | Cosine similarity, stdlib only |
| `metrics.py` | `recall@k`, `MRR@10`, and the one judgement call |
| `report.py` | Markdown and JSON output |
| `corpora/synthetic/` | Committed fixture corpus + its query set |

## `--frontmatter recover` vs `strict`

A hand-maintained corpus has hand-written YAML, and YAML is unforgiving about
two things authors do constantly: an unquoted `:` inside a plain scalar, and an
unescaped `#`, which starts a comment and silently eats the rest of the line.
In the real brain this damaged the `description` of **116 of 440 notes** — 93
blocks that do not parse at all, 23 that parse but lose most of the value.

That is not a neutral detail. Reading those descriptions as absent strips
`summary` of its retrieval key on a quarter of the corpus and credits the loss
to indexing granularity, which it has nothing to do with. Reading them as
authored assumes a quoting fix that has not happened.

So it is a flag and **both are measured**, rather than one being argued for:

- `strict` — `yaml.safe_load` alone. What a production ingest on a stock YAML
  parser sees today.
- `recover` (default) — additionally restores authored text for keys YAML
  dropped or truncated. What the human wrote. The merge rule is narrow: take
  the line-based value only when it is strictly *longer* than the parsed one,
  which happens exactly when YAML dropped characters, and never when the author
  used a block scalar the line reader cannot see.

Every report states which mode produced it, and `notes_with_recovered_frontmatter`
counts how much of the description coverage rests on the fallback.

## Adding query cases

Edit `eval/corpora/synthetic/queries.yaml`. The file's header comment explains
the five fields and, more importantly, how to keep the mix of case kinds
honest — a set built only from note titles hands `summary` a win it has not
earned, and a set built only from buried details does the same favour to
`chunk`.

`kind` is **required** on every case and must be `topic`, `buried` or
`oblique`; the loader rejects a set without it. That is what makes a run
interpretable: every report breaks recall and MRR down **per class** as well as
overall, because the overall figure is just those classes weighted by the case
mix. Change the mix and the headline winner can change with no retrieval
behaviour changing at all — so the by-kind table is the finding and the
aggregate is a summary of it.

## What the numbers do not include

- **No HNSW.** The scan here is exact, so the figures are an upper bound.
  Production retrieves through an approximate, post-filtered pgvector index
  that loses recall of its own; this harness does not measure that loss.
- **`k` counts units, not notes** — matching `/search`'s `top_k`, which returns
  that many chunk rows. `eval/metrics.py` explains why and reports
  `distinct_notes_at_k` alongside so slot dilution is visible.
- **Hand-labelled relevance.** The numbers are only as good as the query set.
- **The aggregate is a fact about the case mix** as much as about retrieval.
  Read the by-kind table; it is the part that survives someone choosing a
  different mix.
- **`hybrid` is one ranking, not a fusion.** The figures say nothing about what
  a rank-fusion (RRF or similar) over the two routes would do.
- **One model at one moment.** Re-run after any change to `OLLAMA_MODEL` or
  `EMBEDDING_DIMENSION`.
