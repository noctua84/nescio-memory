# Retrieval evaluation harness

Measures whether indexing a note's `name + description` as the retrieval key
(one vector per note) beats the service's current behaviour of chunking the
note body at `CHUNK_SIZE`/`CHUNK_OVERLAP` (several vectors per note).

That question is the one `nescio-ai`'s ADR 0005 answers from the *shape* of the
corpus rather than from measurement, and which its own "to re-verify" section
flags as unmeasured. This harness is what settles it.

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

## Adding query cases

Edit `eval/corpora/synthetic/queries.yaml`. The file's header comment explains
the four fields and, more importantly, how to keep the mix of case kinds
honest — a set built only from note titles hands `summary` a win it has not
earned, and a set built only from buried details does the same favour to
`chunk`.

## What the numbers do not include

- **No HNSW.** The scan here is exact, so the figures are an upper bound.
  Production retrieves through an approximate, post-filtered pgvector index
  that loses recall of its own; this harness does not measure that loss.
- **`k` counts units, not notes** — matching `/search`'s `top_k`, which returns
  that many chunk rows. `eval/metrics.py` explains why and reports
  `distinct_notes_at_k` alongside so slot dilution is visible.
- **Hand-labelled relevance.** The numbers are only as good as the query set.
- **One model at one moment.** Re-run after any change to `OLLAMA_MODEL` or
  `EMBEDDING_DIMENSION`.
