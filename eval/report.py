"""Rendering a run as a markdown report and a JSON record.

Numeric columns are padded to a fixed width and a fixed number of decimals, so
the figures line up under a tabular-numerals font instead of drifting with the
digit count. Deltas always carry an explicit sign for the same reason.
"""
import json
from pathlib import Path

from eval.metrics import StrategyScore

# One limitation list, rendered into every report. A reader who sees only the
# output file -- which is how these get circulated -- must see the caveats too,
# not have to find them in a module docstring.
LIMITATIONS = [
    "No database. Vectors are held in memory and scanned exactly, so these "
    "numbers are an UPPER BOUND: production retrieves through a pgvector HNSW "
    "index that is approximate and post-filtered by client_name, and loses "
    "recall of its own. This harness does not measure that loss.",
    "k counts retrieved UNITS, not distinct notes, matching /search's top_k. "
    "The `chunk` strategy therefore pays for spending result slots on several "
    "chunks of one note. See eval/metrics.py for why that is deliberate.",
    "Query relevance is hand-labelled. The numbers are only as good as the "
    "query set, and a query set that mostly restates note titles will flatter "
    "the `summary` strategy.",
    "A single embedding model at one point in time. Re-run after any change to "
    "OLLAMA_MODEL or EMBEDDING_DIMENSION; the result does not transfer.",
]

STUB_BANNER = """> [!WARNING]
> **These numbers are meaningless as a finding.**
>
> This run used the `stub` embedder: a lexical hashing vectorizer that scores
> literal word overlap and has no semantic capability whatsoever. It exists
> only to prove this harness's plumbing end to end where no Ollama server is
> reachable. It cannot distinguish the two indexing strategies on the thing
> they actually differ on, because it cannot match a paraphrase to prose.
>
> Do not quote any figure below. Re-run with `--embedder real` against the
> configured model before drawing any conclusion.
"""


def _row(cells: list[str], widths: list[int], aligns: str) -> str:
    padded = [
        cell.ljust(width) if align == "l" else cell.rjust(width)
        for cell, width, align in zip(cells, widths, aligns)
    ]
    return "| " + " | ".join(padded) + " |"


def _divider(widths: list[int], aligns: str) -> str:
    parts = [
        ":" + "-" * (width - 1) if align == "l" else "-" * (width - 1) + ":"
        for width, align in zip(widths, aligns)
    ]
    return "| " + " | ".join(parts) + " |"


def _table(header: list[str], rows: list[list[str]], aligns: str | None = None) -> str:
    """Render a markdown table.

    `aligns` is one character per column, "l" or "r"; it defaults to a text
    first column and right-aligned numerics after it. Numeric columns are
    right-aligned and padded to a fixed width so the digits line up under a
    tabular-numerals font rather than drifting with the digit count.
    """
    aligns = aligns or ("l" + "r" * (len(header) - 1))
    widths = [
        max(len(header[column]), *(len(row[column]) for row in rows))
        for column in range(len(header))
    ]
    widths = [max(width, 6) for width in widths]
    lines = [_row(header, widths, aligns), _divider(widths, aligns)]
    lines.extend(_row(row, widths, aligns) for row in rows)
    return "\n".join(lines)


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _verdict(chunk: StrategyScore, summary: StrategyScore, is_real: bool) -> str:
    if not is_real:
        return (
            "**No verdict.** This run used the stub embedder; it measures word "
            "overlap, not meaning, so it cannot answer the question."
        )
    recall_delta = summary.recall_at_k[5] - chunk.recall_at_k[5]
    mrr_delta = summary.mrr_at_10 - chunk.mrr_at_10
    # A single threshold, named rather than implied. 0.05 on a query set of a
    # few dozen cases is roughly one or two queries changing outcome, which is
    # not a result; it is stated as inconclusive rather than rounded into one.
    if abs(recall_delta) < 0.05 and abs(mrr_delta) < 0.05:
        leaning = (
            "**Inconclusive.** Both recall@5 and MRR@10 differ by less than "
            "0.05, which on this query set is a query or two changing outcome. "
            "Add cases before concluding."
        )
    elif recall_delta > 0 and mrr_delta > 0:
        leaning = (
            f"**`summary` wins** on both headline metrics "
            f"(recall@5 {_signed(recall_delta)}, MRR@10 {_signed(mrr_delta)})."
        )
    elif recall_delta < 0 and mrr_delta < 0:
        leaning = (
            f"**`chunk` wins** on both headline metrics "
            f"(recall@5 {_signed(recall_delta)}, MRR@10 {_signed(mrr_delta)})."
        )
    else:
        leaning = (
            f"**Split.** recall@5 {_signed(recall_delta)} but MRR@10 "
            f"{_signed(mrr_delta)}; the two metrics disagree, so read the "
            f"per-query table rather than the headline."
        )
    return leaning + " Deltas are `summary` minus `chunk`."


def build_markdown(run: dict) -> str:
    ks = run["ks"]
    chunk = run["scores"]["chunk"]
    summary = run["scores"]["summary"]
    is_real = run["embedder"]["is_real"]

    out: list[str] = ["# Retrieval evaluation: `chunk` vs `summary`", ""]
    if not is_real:
        out += [STUB_BANNER, ""]

    out += [
        "Hand-run measurement, not a test. See `eval/__init__.py` for why this "
        "is not in `tests/`.",
        "",
        "## Run",
        "",
        f"- **when**: {run['started_at']}",
        f"- **embedder**: `{run['embedder']['label']}` -- "
        f"{run['embedder']['describe']}",
        f"- **corpus**: `{run['corpus']['path']}`",
        f"- **query set**: `{run['query_set']['path']}` "
        f"({run['query_set']['queries']} queries)",
        f"- **chunking config**: CHUNK_SIZE={run['config']['chunk_size']}, "
        f"CHUNK_OVERLAP={run['config']['chunk_overlap']}, "
        f"min chunk chars={run['config']['min_content_chars']}",
        "",
        "## Corpus shape",
        "",
    ]
    shape = run["corpus"]["shape"]
    out.append(
        _table(
            ["property", "value"],
            [[key.replace("_", " "), f"{value:g}"] for key, value in shape.items()],
        )
    )

    out += ["", "## Index shape", ""]
    chunk_stats = run["index"]["chunk"]
    summary_stats = run["index"]["summary"]
    out.append(
        _table(
            ["property", "chunk", "summary"],
            [
                ["retrievable units", str(chunk_stats["units"]), str(summary_stats["units"])],
                ["notes indexed", str(chunk_stats["notes_indexed"]), str(summary_stats["notes_indexed"])],
                ["units per note", f"{chunk_stats['units_per_note']:.2f}", f"{summary_stats['units_per_note']:.2f}"],
                ["mean unit chars", f"{chunk_stats['mean_unit_chars']:.1f}", f"{summary_stats['mean_unit_chars']:.1f}"],
                [
                    "notes w/o description (name-only fallback)",
                    "n/a",
                    str(summary_stats["notes_without_description"]),
                ],
                [
                    "notes with no unit at all",
                    str(len(chunk_stats["notes_without_units"])),
                    str(len(summary_stats["notes_without_units"])),
                ],
            ],
        )
    )

    out += ["", "## Results", ""]
    metric_rows = []
    for k in ks:
        metric_rows.append(
            [
                f"recall@{k}",
                f"{chunk.recall_at_k[k]:.3f}",
                f"{summary.recall_at_k[k]:.3f}",
                _signed(summary.recall_at_k[k] - chunk.recall_at_k[k]),
            ]
        )
    metric_rows.append(
        [
            "MRR@10",
            f"{chunk.mrr_at_10:.3f}",
            f"{summary.mrr_at_10:.3f}",
            _signed(summary.mrr_at_10 - chunk.mrr_at_10),
        ]
    )
    out.append(_table(["metric", "chunk", "summary", "delta"], metric_rows))

    out += [
        "",
        "### Diagnostic: distinct notes among the top k units",
        "",
        "`summary` is one unit per note, so this is k by construction. For "
        "`chunk` it shows how many result slots are duplicate notes -- the "
        "mechanism behind any recall gap above.",
        "",
    ]
    out.append(
        _table(
            ["k", "chunk", "summary"],
            [
                [
                    str(k),
                    f"{chunk.mean_distinct_notes_at_k[k]:.2f}",
                    f"{summary.mean_distinct_notes_at_k[k]:.2f}",
                ]
                for k in ks
            ],
        )
    )

    out += ["", "## Verdict", "", _verdict(chunk, summary, is_real), ""]

    out += ["## Per query (rank of first relevant note, 1-based in unit space)", ""]
    per_chunk = {score.query_id: score for score in chunk.per_query}
    per_summary = {score.query_id: score for score in summary.per_query}
    rows = []
    for query in run["query_set"]["entries"]:
        query_id = query["id"]
        chunk_rank = per_chunk[query_id].first_relevant_rank
        summary_rank = per_summary[query_id].first_relevant_rank
        text = query["query"]
        rows.append(
            [
                query_id,
                text if len(text) <= 58 else text[:55] + "...",
                str(chunk_rank) if chunk_rank else "miss",
                str(summary_rank) if summary_rank else "miss",
            ]
        )
    out.append(_table(["id", "query", "chunk", "summary"], rows, aligns="llrr"))
    out += ["", "`miss` means no expected note appeared in the top 10.", ""]

    out += ["## What this does not measure", ""]
    out.extend(f"{position}. {item}" for position, item in enumerate(LIMITATIONS, 1))
    out.append("")
    return "\n".join(out)


def _score_to_dict(score: StrategyScore) -> dict:
    return {
        "strategy": score.strategy_name,
        "queries": score.queries,
        "recall_at_k": {str(k): value for k, value in score.recall_at_k.items()},
        "mrr_at_10": score.mrr_at_10,
        "mean_distinct_notes_at_k": {
            str(k): value for k, value in score.mean_distinct_notes_at_k.items()
        },
        "per_query": [
            {
                "id": query.query_id,
                "recall_at_k": {
                    str(k): value for k, value in query.recall_at_k.items()
                },
                "reciprocal_rank": query.reciprocal_rank,
                "first_relevant_rank": query.first_relevant_rank,
                "distinct_notes_at_k": {
                    str(k): value for k, value in query.distinct_notes_at_k.items()
                },
                "retrieved_note_ids": query.retrieved_note_ids,
            }
            for query in score.per_query
        ],
    }


def build_json(run: dict) -> str:
    payload = {
        "started_at": run["started_at"],
        "embedder": run["embedder"],
        "config": run["config"],
        "corpus": run["corpus"],
        "query_set": {
            "path": run["query_set"]["path"],
            "queries": run["query_set"]["queries"],
            "entries": run["query_set"]["entries"],
        },
        "ks": list(run["ks"]),
        "index": run["index"],
        "scores": {
            name: _score_to_dict(score) for name, score in run["scores"].items()
        },
        "limitations": LIMITATIONS,
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def write_reports(run: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = run["slug"]
    markdown_path = out_dir / f"{stamp}.md"
    json_path = out_dir / f"{stamp}.json"
    markdown_path.write_text(build_markdown(run), encoding="utf-8")
    json_path.write_text(build_json(run), encoding="utf-8")
    return markdown_path, json_path
