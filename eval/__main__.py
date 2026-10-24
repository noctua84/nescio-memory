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
from eval.corpus import (
    FRONTMATTER_MODES,
    CorpusError,
    corpus_shape,
    load_corpus,
)
from eval.embedders import build_embedder
from eval.metrics import DEFAULT_KS, aggregate, aggregate_by_kind, score_query
from eval.queries import QUERY_KINDS, QuerySetError, kind_mix, load_query_set
from eval.report import ReportError, write_reports
from eval.retrieval import build_index, search, union_index
from eval.strategies import (
    COMPOSITES,
    MIN_CONTENT_CHARS,
    REPORT_ORDER,
    STRATEGIES,
)

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
        # --corpus is deliberately NOT confined to the repository. Pointing it
        # at an arbitrary directory outside this checkout is the entire point
        # of the tool -- the real use is a brain living somewhere else
        # entirely, e.g. `--corpus /path/to/brain` (see the epilog above and
        # eval/README.md). Do not "harden" this one; it would break the
        # harness for its actual job. --out and --queries below carry the
        # confinement instead, because they are the primitives that write and
        # read on this operator-supplied path, not merely index its contents.
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
        "--frontmatter",
        choices=FRONTMATTER_MODES,
        default="recover",
        help=(
            "how to read a note's name/description. 'strict' is yaml.safe_load "
            "alone -- a block that will not parse contributes nothing, and a "
            "value YAML eats as a '#' comment stays eaten, which is what a "
            "production ingest on a stock YAML parser would see. 'recover' "
            "additionally reads the authored text for keys YAML dropped or "
            "truncated, which is what the human wrote. The choice moves the "
            "result on a hand-maintained corpus, so run BOTH rather than "
            "arguing it (default: recover)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="index only the first N notes, for a quick smoke run",
    )
    return parser


def path_is_confined(path: Path, *roots: Path) -> bool:
    """True if `path`, once resolved, is one of `roots` or sits inside one.

    Containment is checked on RESOLVED paths, not the raw argument. Two
    reasons, both load-bearing:

      * `Path.resolve()` is non-strict by default, so a destination that does
        not exist yet (the common case for --out, which its caller is about
        to create) still resolves to a real, comparable path.
      * A `..` segment, a symlink, or an absolute path that happens to type
        inside `root` as a string all have to be caught, and only the
        resolved filesystem location can be trusted to catch all three. A
        string check for ".." (or any other textual heuristic) misses a
        symlink entirely and is the kind of "fix" that looks right in review
        and is wrong in production.
    """
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def confine_out_dir(out_dir: Path) -> None:
    """Refuse an --out that resolves outside the repository.

    write_reports() creates `out_dir` (mkdir(parents=True, exist_ok=True))
    and then writes two files into it with no further checks -- an arbitrary
    file write if `out_dir` is left unconstrained, and this harness is driven
    by an agent assembling command lines, so a prompt-injected --out is not
    hypothetical. Confining it costs nothing: DEFAULT_OUT is already in-repo,
    and nobody has a legitimate reason to write a report outside this
    checkout. Unlike --corpus, this argument is safe to lock down.
    """
    if not path_is_confined(out_dir, REPO_ROOT):
        raise ReportError(
            f"--out must resolve inside the repository ({REPO_ROOT}); "
            f"{out_dir} resolves to {out_dir.resolve()}, outside it"
        )


def resolve_query_set(corpus: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit
    else:
        candidate = corpus / QUERY_SET_FILENAME
        if not candidate.is_file():
            raise QuerySetError(
                f"no {QUERY_SET_FILENAME} in {corpus} -- pass --queries explicitly. "
                f"For a real brain directory, keep the query set outside it "
                f"(eval/private/ is gitignored) rather than writing into the brain."
            )
        path = candidate

    # --queries has exactly two legitimate homes: inside this repository (the
    # gitignored eval/private/ is where a real query set belongs -- see the
    # epilog above) or inside the --corpus directory the operator already
    # named (<corpus>/queries.yaml, the default resolved just above).
    # load_query_set() both reads this file and parses it as YAML, echoing
    # path and content fragments into its error messages on a bad parse --
    # an arbitrary-file-read primitive if this path is left unconstrained, and
    # this harness is driven by an agent assembling command lines while
    # reading a private corpus, so a prompt-injected --queries is not
    # hypothetical either.
    if not path_is_confined(path, REPO_ROOT, corpus):
        raise QuerySetError(
            f"--queries must resolve inside the repository ({REPO_ROOT}) or "
            f"inside --corpus ({corpus}); {path} resolves to {path.resolve()}, "
            f"outside both"
        )
    return path


def run(arguments: argparse.Namespace) -> dict:
    started = datetime.now(timezone.utc)
    notes = load_corpus(arguments.corpus, arguments.frontmatter)
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

    kinds = {query.query_id: query.kind for query in queries}

    # Primitives first, because the composites are unions of their indexes and
    # must not trigger a second embedding pass. built[] keeps the Index objects
    # (vectors included) alive for exactly that reason.
    built = {
        name: build_index(notes, strategy, embedder)
        for name, strategy in STRATEGIES.items()
    }
    for name, composite in COMPOSITES.items():
        print(
            f"[{name}] unioning {' + '.join(composite.parts)} -- no embedding",
            file=sys.stderr,
            flush=True,
        )
        built[name] = union_index(
            name, [built[part] for part in composite.parts]
        )

    indexes = {}
    scores = {}
    class_scores = {}
    first_hit_routes = {}
    for name in REPORT_ORDER:
        index = built[name]
        per_query = []
        routes: dict[str, str | None] = {}
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
            # Which unit type reached the note first. Meaningless for a
            # primitive (always itself) and the whole point for `hybrid`: it is
            # the only way to see whether the union is actually using both
            # routes or whether one of them is dead weight.
            routes[query.query_id] = next(
                (hit.unit_kind for hit in hits if hit.note_id in query.relevant),
                None,
            )
        indexes[name] = index.stats()
        scores[name] = aggregate(name, per_query, DEFAULT_KS)
        class_scores[name] = aggregate_by_kind(name, per_query, kinds, DEFAULT_KS)
        first_hit_routes[name] = routes

    return {
        "started_at": started.isoformat(timespec="seconds"),
        "slug": (
            f"{started.strftime('%Y%m%d-%H%M%S')}-{arguments.corpus.name}"
            f"-{embedder.label}-{arguments.frontmatter}"
        ),
        "embedder": {
            "label": embedder.label,
            "is_real": embedder.is_real,
            "describe": embedder.describe(),
            # Disclosed rather than swallowed: a run that needed retries was
            # measured against a backend that wobbled, and a reader deciding
            # how much to trust a small delta should know that.
            "calls": getattr(embedder, "calls", 0),
            "cache_hits": getattr(embedder, "cache_hits", 0),
            "retries": getattr(embedder, "retries", 0),
        },
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_content_chars": MIN_CONTENT_CHARS,
            "embedding_dimension": settings.embedding_dimension,
            "frontmatter_mode": arguments.frontmatter,
            "strategies": {
                name: (STRATEGIES.get(name) or COMPOSITES[name]).describe()
                for name in REPORT_ORDER
            },
        },
        "corpus": {
            "path": str(arguments.corpus),
            "shape": corpus_shape(notes),
        },
        "query_set": {
            "path": str(query_path),
            "queries": len(queries),
            "kind_mix": kind_mix(queries),
            "entries": [
                {
                    "id": query.query_id,
                    "query": query.text,
                    "kind": query.kind,
                    "expect_notes": list(query.expect_notes),
                    "why": query.why,
                }
                for query in queries
            ],
        },
        "ks": list(DEFAULT_KS),
        "strategy_order": list(REPORT_ORDER),
        "index": indexes,
        "scores": scores,
        "class_scores": class_scores,
        "first_hit_routes": first_hit_routes,
    }


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, and the report contains characters
    # outside it. Reconfigure before anything is printed.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    arguments = build_parser().parse_args(argv)
    try:
        # Checked before run() does any work (including the embedding pass),
        # so a bad --out fails fast rather than after several minutes against
        # a real model.
        confine_out_dir(arguments.out)
        result = run(arguments)
    except (CorpusError, QuerySetError, ReportError) as exc:
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

    order = result["strategy_order"]
    mix = result["query_set"]["kind_mix"]
    print()
    print(f"corpus   {result['corpus']['shape']['notes']} notes, "
          f"{result['query_set']['queries']} queries "
          f"({', '.join(f'{kind} {count}' for kind, count in mix.items())})")
    print(f"embedder {result['embedder']['label']} -- "
          f"{result['embedder']['describe']}")
    print()
    header = f"{'metric':<10}" + "".join(f"{name:>9}" for name in order)
    print(header)
    for k in DEFAULT_KS:
        row = f"{'recall@' + str(k):<10}"
        row += "".join(
            f"{result['scores'][name].recall_at_k[k]:>9.3f}" for name in order
        )
        print(row)
    print(f"{'MRR@10':<10}" + "".join(
        f"{result['scores'][name].mrr_at_10:>9.3f}" for name in order
    ))

    # The per-class block, printed rather than left to the report file, because
    # the aggregate above is a weighted average of these with the case mix as
    # the weights -- reading it alone is how a case mix gets mistaken for a
    # finding.
    print()
    print("recall@5 / MRR@10 by query kind")
    print(f"{'kind':<10}{'n':>4}" + "".join(f"{name:>17}" for name in order))
    for kind in QUERY_KINDS:
        present = [
            name for name in order if kind in result["class_scores"][name]
        ]
        if not present:
            continue
        count = result["class_scores"][present[0]][kind].queries
        row = f"{kind:<10}{count:>4}"
        for name in order:
            slice_ = result["class_scores"][name].get(kind)
            if slice_ is None:
                row += f"{'--':>17}"
            else:
                row += f"{slice_.recall_at_k[5]:>10.3f}/{slice_.mrr_at_10:>6.3f}"
        print(row)
    print()
    print(f"report   {markdown_path}")
    print(f"json     {json_path}")
    if not result["embedder"]["is_real"]:
        print()
        print("REMINDER: stub embedder -- these numbers are not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
