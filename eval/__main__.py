"""CLI: `uv run python -m eval [options]`.

Run it by hand when you want a measurement. It is not a test and must never be
wired into CI -- see eval/__init__.py.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import eval.appenv  # noqa: F401  -- must precede any app. import; see appenv

from app.config import settings
from app.core.errors import EmbeddingBackendError
from eval.corpus import CorpusError, corpus_shape, load_corpus
from eval.embedders import build_embedder
from eval.metrics import DEFAULT_KS, aggregate, score_query
from eval.queries import QuerySetError, load_query_set
from eval.report import write_reports
from eval.retrieval import build_index, search
from eval.strategies import MIN_CONTENT_CHARS, STRATEGIES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "eval" / "corpora" / "synthetic"
DEFAULT_OUT = REPO_ROOT / "eval-out"
QUERY_SET_FILENAME = "queries.yaml"

# Enough units to cover the largest k. Fixed rather than exposed as a flag:
# MRR@10 and recall@10 are the deepest metrics, so retrieving further would
# cost time and change nothing.
RETRIEVE_LIMIT = max(DEFAULT_KS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description=(
            "Measure whether indexing a note's name+description beats chunking "
            "its body, using the service's own embedder. Hand-run measurement "
            "tool, not a test -- needs a reachable embedding backend."
        ),
        epilog=(
            "Real-corpus runs: outputs and any real query set are gitignored on "
            "purpose. Keep a real query set under eval/private/ and write "
            "reports to the default eval-out/. Nothing derived from a private "
            "brain may be committed -- this repository is public."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        metavar="PATH",
        help=(
            "directory of markdown notes to index, searched recursively "
            "(default: the committed synthetic fixture corpus)"
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"query set YAML (default: <corpus>/{QUERY_SET_FILENAME})",
    )
    parser.add_argument(
        "--embedder",
        choices=("real", "stub"),
        default="real",
        help=(
            "'real' calls app.core.embeddings and measures the configured "
            "model; 'stub' is a lexical hashing vectorizer that only proves "
            "the plumbing and whose numbers are not a finding (default: real)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        metavar="DIR",
        help="where to write the .md and .json report (default: eval-out/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="index only the first N notes, for a quick smoke run",
    )
    return parser


def resolve_query_set(corpus: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidate = corpus / QUERY_SET_FILENAME
    if candidate.is_file():
        return candidate
    raise QuerySetError(
        f"no {QUERY_SET_FILENAME} in {corpus} -- pass --queries explicitly. For a "
        f"real brain directory, keep the query set outside it (eval/private/ is "
        f"gitignored) rather than writing into the brain."
    )


def run(arguments: argparse.Namespace) -> dict:
    started = datetime.now(timezone.utc)
    notes = load_corpus(arguments.corpus)
    if arguments.limit is not None:
        notes = notes[: arguments.limit]
    query_path = resolve_query_set(arguments.corpus, arguments.queries)
    queries = load_query_set(query_path, notes)

    embedder = build_embedder(arguments.embedder)
    if not embedder.is_real:
        print(
            "WARNING: --embedder stub is a lexical hashing vectorizer, not a "
            "model.\n         It proves the plumbing only. Its numbers are not "
            "a finding.",
            file=sys.stderr,
        )

    indexes = {}
    scores = {}
    for name, strategy in STRATEGIES.items():
        index = build_index(notes, strategy, embedder)
        per_query = []
        for query in queries:
            hits = search(index, embedder(query.text), RETRIEVE_LIMIT)
            per_query.append(
                score_query(
                    query.query_id,
                    [hit.note_id for hit in hits],
                    query.relevant,
                    DEFAULT_KS,
                )
            )
        indexes[name] = index.stats()
        scores[name] = aggregate(name, per_query, DEFAULT_KS)

    return {
        "started_at": started.isoformat(timespec="seconds"),
        "slug": (
            f"{started.strftime('%Y%m%d-%H%M%S')}-{arguments.corpus.name}"
            f"-{embedder.label}"
        ),
        "embedder": {
            "label": embedder.label,
            "is_real": embedder.is_real,
            "describe": embedder.describe(),
        },
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_content_chars": MIN_CONTENT_CHARS,
            "embedding_dimension": settings.embedding_dimension,
            "strategies": {
                name: strategy.describe() for name, strategy in STRATEGIES.items()
            },
        },
        "corpus": {
            "path": str(arguments.corpus),
            "shape": corpus_shape(notes),
        },
        "query_set": {
            "path": str(query_path),
            "queries": len(queries),
            "entries": [
                {
                    "id": query.query_id,
                    "query": query.text,
                    "expect_notes": list(query.expect_notes),
                    "why": query.why,
                }
                for query in queries
            ],
        },
        "ks": list(DEFAULT_KS),
        "index": indexes,
        "scores": scores,
    }


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, and the report contains characters
    # outside it. Reconfigure before anything is printed.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    arguments = build_parser().parse_args(argv)
    try:
        result = run(arguments)
    except (CorpusError, QuerySetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EmbeddingBackendError as exc:
        # Reported, never worked around. A run that substituted anything for a
        # failed embedding call would write a plausible report built on missing
        # data, which is the worst outcome available to a measurement tool.
        print(
            f"error: the embedding backend failed, so no measurement was made.\n"
            f"       {exc}\n"
            f"       backend={settings.embedding_backend} "
            f"url={settings.ollama_url} model={settings.ollama_model}\n"
            f"       Start the backend and re-run. To check this harness's "
            f"plumbing without a\n"
            f"       model, use --embedder stub -- but its numbers are not a "
            f"finding.",
            file=sys.stderr,
        )
        return 3

    markdown_path, json_path = write_reports(result, arguments.out)

    chunk = result["scores"]["chunk"]
    summary = result["scores"]["summary"]
    print()
    print(f"corpus   {result['corpus']['shape']['notes']} notes, "
          f"{result['query_set']['queries']} queries")
    print(f"embedder {result['embedder']['label']} -- "
          f"{result['embedder']['describe']}")
    print()
    print(f"{'metric':<10}{'chunk':>9}{'summary':>9}{'delta':>9}")
    for k in DEFAULT_KS:
        delta = summary.recall_at_k[k] - chunk.recall_at_k[k]
        print(f"{'recall@' + str(k):<10}{chunk.recall_at_k[k]:>9.3f}"
              f"{summary.recall_at_k[k]:>9.3f}{delta:>+9.3f}")
    print(f"{'MRR@10':<10}{chunk.mrr_at_10:>9.3f}{summary.mrr_at_10:>9.3f}"
          f"{summary.mrr_at_10 - chunk.mrr_at_10:>+9.3f}")
    print()
    print(f"report   {markdown_path}")
    print(f"json     {json_path}")
    if not result["embedder"]["is_real"]:
        print()
        print("REMINDER: stub embedder -- these numbers are not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
