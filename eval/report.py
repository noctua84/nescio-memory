"""Rendering a run as a markdown report and a JSON record.

Numeric columns are padded to a fixed width and a fixed number of decimals, so
the figures line up under a tabular-numerals font instead of drifting with the
digit count. Deltas always carry an explicit sign for the same reason.
"""
import json
from pathlib import Path

from eval.metrics import StrategyScore
from eval.queries import QUERY_KINDS

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
    "The OVERALL row is a weighted average of the per-class rows with the case "
    "mix as the weights, so on its own it is a fact about the query set as "
    "much as about retrieval. Read the by-kind table; it is the part that "
    "survives a change of mix.",
    "`hybrid` is measured as one shared cosine ranking over both unit types, "
    "which is what storing both in one table would mean for /search. It is "
    "not a score fusion, and says nothing about what one would do.",
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


# A single threshold, named rather than implied. 0.05 on a query set of a few
# dozen cases is roughly one or two queries changing outcome, which is not a
# result; it is stated as inconclusive rather than rounded into one.
MATERIAL_DELTA = 0.05


def _leader(scores: dict[str, StrategyScore], order: list[str], metric) -> list[str]:
    """Every strategy within MATERIAL_DELTA of the best on `metric`.

    A list rather than a winner, because "best by 0.004" is not a finding and
    naming it as one is how a measurement turns into a decision it cannot
    support.
    """
    best = max(metric(scores[name]) for name in order)
    return [
        name for name in order if best - metric(scores[name]) < MATERIAL_DELTA
    ]


def _verdict(run: dict) -> str:
    if not run["embedder"]["is_real"]:
        return (
            "**No verdict.** This run used the stub embedder; it measures word "
            "overlap, not meaning, so it cannot answer the question."
        )
    order = list(run["strategy_order"])
    scores = run["scores"]
    by_recall = _leader(scores, order, lambda score: score.recall_at_k[5])
    by_mrr = _leader(scores, order, lambda score: score.mrr_at_10)
    agreed = [name for name in by_recall if name in by_mrr]

    lines: list[str] = []
    if not agreed:
        lines.append(
            f"**Split.** recall@5 favours {_names(by_recall)} and MRR@10 "
            f"favours {_names(by_mrr)}. The two metrics disagree, so the "
            f"headline is not a result -- read the by-kind table."
        )
    elif len(agreed) == len(order):
        lines.append(
            f"**Inconclusive overall.** All {len(order)} strategies sit within "
            f"{MATERIAL_DELTA:.2f} of each other on both recall@5 and MRR@10, "
            f"which on this query set is a query or two changing outcome."
        )
    elif len(agreed) == 1:
        lines.append(
            f"**`{agreed[0]}` leads** on both recall@5 and MRR@10 by more than "
            f"{MATERIAL_DELTA:.2f}."
        )
    else:
        lines.append(
            f"**{_names(agreed)} tie** at the top on both metrics, within "
            f"{MATERIAL_DELTA:.2f} of each other and ahead of the rest."
        )

    # Per-class, stated in the verdict and not only in its own table. A verdict
    # that quoted the aggregate alone would be reporting the case mix.
    class_scores = run["class_scores"]
    for kind in QUERY_KINDS:
        if kind not in class_scores[order[0]]:
            continue
        slices = {name: class_scores[name][kind] for name in order}
        winners = _leader(slices, order, lambda score: score.mrr_at_10)
        count = slices[order[0]].queries
        detail = ", ".join(
            f"`{name}` {slices[name].mrr_at_10:.3f}" for name in order
        )
        lines.append(
            f"- **{kind}** (n={count}): MRR@10 {detail} -- "
            + (
                "no separation."
                if len(winners) == len(order)
                else f"{_names(winners)} ahead."
            )
        )
    return "\n".join(lines)


def _names(names: list[str]) -> str:
    quoted = [f"`{name}`" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def build_markdown(run: dict) -> str:
    ks = run["ks"]
    order = list(run["strategy_order"])
    scores = run["scores"]
    class_scores = run["class_scores"]
    is_real = run["embedder"]["is_real"]

    out: list[str] = [
        "# Retrieval evaluation: " + " vs ".join(f"`{name}`" for name in order),
        "",
    ]
    if not is_real:
        out += [STUB_BANNER, ""]

    mix = run["query_set"]["kind_mix"]
    total = sum(mix.values()) or 1
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
        "- **case mix**: "
        + ", ".join(
            f"{kind} {count} ({count / total:.0%})" for kind, count in mix.items()
        ),
        f"- **chunking config**: CHUNK_SIZE={run['config']['chunk_size']}, "
        f"CHUNK_OVERLAP={run['config']['chunk_overlap']}, "
        f"min chunk chars={run['config']['min_content_chars']}",
        f"- **embedding dimension**: {run['config']['embedding_dimension']}",
        f"- **frontmatter mode**: `{run['config']['frontmatter_mode']}`",
        "",
        "## Strategies",
        "",
    ]
    out.append(
        _table(
            ["strategy", "rule"],
            [[f"`{name}`", run["config"]["strategies"][name]] for name in order],
            aligns="ll",
        )
    )

    out += ["", "## Corpus shape", ""]
    shape = run["corpus"]["shape"]
    out.append(
        _table(
            ["property", "value"],
            [[key.replace("_", " "), f"{value:g}"] for key, value in shape.items()],
        )
    )
    recovered = shape.get("notes_with_recovered_frontmatter", 0)
    if run["config"]["frontmatter_mode"] == "strict":
        out += [
            "",
            "> [!NOTE]",
            "> `--frontmatter strict`: a block that does not parse as YAML "
            "contributed nothing, and a description YAML ate as a `#` comment "
            "stayed eaten. This is what a production ingest on a stock YAML "
            "parser sees, so `summary` is measured here on the keys the "
            "corpus can actually hand it rather than on what its authors "
            "wrote. Compare against the `recover` run of the same query set.",
        ]
    elif recovered:
        out += [
            "",
            "> [!NOTE]",
            f"> `--frontmatter recover`: {recovered} of {shape['notes']} notes "
            f"needed authored `name`/`description` text restored, because "
            f"their frontmatter either does not parse as YAML (an unquoted "
            f"`:` in a plain scalar) or parses but loses the value to a `#` "
            f"comment. `summary` is measured here on what the human wrote. "
            f"Treating a fixable quoting bug as a missing description would "
            f"instead credit that loss to indexing granularity -- compare "
            f"against the `strict` run to see the size of the difference.",
        ]

    out += ["", "## Index shape", ""]
    stats = {name: run["index"][name] for name in order}
    out.append(
        _table(
            ["property"] + [f"`{name}`" for name in order],
            [
                ["retrievable units"]
                + [str(stats[name]["units"]) for name in order],
                ["notes indexed"]
                + [str(stats[name]["notes_indexed"]) for name in order],
                ["units per note"]
                + [f"{stats[name]['units_per_note']:.2f}" for name in order],
                ["mean unit chars"]
                + [f"{stats[name]['mean_unit_chars']:.1f}" for name in order],
                ["notes w/o description (name-only summary unit)"]
                + [
                    str(stats[name]["notes_without_description"]) or "0"
                    for name in order
                ],
                ["notes with no unit at all"]
                + [str(len(stats[name]["notes_without_units"])) for name in order],
            ],
        )
    )

    out += ["", "## Results: overall", "", _metric_table(scores, order, ks), ""]
    out += [
        "> [!IMPORTANT]",
        "> This table is a weighted average of the next one, with the case mix "
        "as the weights. Change the mix and the winner here can change with no "
        "retrieval behaviour changing at all. The by-kind table is the finding; "
        "this one is a summary of it.",
        "",
        "## Results: by query kind",
        "",
        "`topic` restates the note's subject in other words. `buried` asks for "
        "a fact that appears once in a body and is absent from the name and "
        "description. `oblique` describes a symptom without naming the "
        "mechanism.",
        "",
    ]
    for kind in QUERY_KINDS:
        if kind not in class_scores[order[0]]:
            continue
        slices = {name: class_scores[name][kind] for name in order}
        count = slices[order[0]].queries
        out += [
            f"### `{kind}` (n={count})",
            "",
            _metric_table(slices, order, ks),
            "",
        ]

    out += [
        "### Diagnostic: distinct notes among the top k units",
        "",
        "`summary` is one unit per note, so this is k by construction. For "
        "`chunk` and `hybrid` it shows how many result slots are duplicate "
        "notes -- the mechanism behind any recall gap above.",
        "",
    ]
    out.append(
        _table(
            ["k"] + [f"`{name}`" for name in order],
            [
                [str(k)]
                + [f"{scores[name].mean_distinct_notes_at_k[k]:.2f}" for name in order]
                for k in ks
            ],
        )
    )

    routes = run.get("first_hit_routes", {})
    if "hybrid" in routes:
        counted: dict[str, int] = {}
        for route in routes["hybrid"].values():
            counted[route or "miss"] = counted.get(route or "miss", 0) + 1
        out += [
            "",
            "### Diagnostic: which route reached the note first in `hybrid`",
            "",
            "If one route never wins, the union is carrying dead weight and "
            "the simpler single-type index is the better design.",
            "",
            _table(
                ["route", "queries"],
                [
                    [key, str(value)]
                    for key, value in sorted(
                        counted.items(), key=lambda row: -row[1]
                    )
                ],
            ),
        ]

    out += ["", "## Verdict", "", _verdict(run), ""]

    out += ["## Per query (rank of first relevant note, 1-based in unit space)", ""]
    per_strategy = {
        name: {score.query_id: score for score in scores[name].per_query}
        for name in order
    }
    rows = []
    for query in run["query_set"]["entries"]:
        query_id = query["id"]
        text = query["query"]
        row = [
            query_id,
            query["kind"],
            text if len(text) <= 52 else text[:49] + "...",
        ]
        for name in order:
            rank = per_strategy[name][query_id].first_relevant_rank
            row.append(str(rank) if rank else "miss")
        rows.append(row)
    out.append(
        _table(
            ["id", "kind", "query"] + [f"`{name}`" for name in order],
            rows,
            aligns="lll" + "r" * len(order),
        )
    )
    out += ["", "`miss` means no expected note appeared in the top 10.", ""]

    out += ["## What this does not measure", ""]
    out.extend(f"{position}. {item}" for position, item in enumerate(LIMITATIONS, 1))
    out.append("")
    return "\n".join(out)


def _metric_table(
    scores: dict[str, StrategyScore], order: list[str], ks: list[int]
) -> str:
    """recall@k and MRR@10, one column per strategy.

    The delta column the two-strategy version carried is gone: with three
    strategies there is no single baseline to subtract, and picking one would
    quietly privilege it. The leader is named in the verdict instead.
    """
    rows = [
        [f"recall@{k}"] + [f"{scores[name].recall_at_k[k]:.3f}" for name in order]
        for k in ks
    ]
    rows.append(["MRR@10"] + [f"{scores[name].mrr_at_10:.3f}" for name in order])
    return _table(["metric"] + [f"`{name}`" for name in order], rows)


def _score_to_dict(score: StrategyScore) -> dict:
    return {
        "strategy": score.strategy_name,
        "kind": score.kind or "overall",
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
            "kind_mix": run["query_set"]["kind_mix"],
            "entries": run["query_set"]["entries"],
        },
        "ks": list(run["ks"]),
        "strategy_order": list(run["strategy_order"]),
        "index": run["index"],
        "scores": {
            name: _score_to_dict(score) for name, score in run["scores"].items()
        },
        # The per-class slices, which are the interpretable part: `scores`
        # above is these weighted by the case mix. A consumer comparing two
        # runs should compare here, because two query sets over the same corpus
        # differ in mix even when both are honest.
        "class_scores": {
            name: {
                kind: _score_to_dict(score) for kind, score in by_kind.items()
            }
            for name, by_kind in run["class_scores"].items()
        },
        "first_hit_routes": run["first_hit_routes"],
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
