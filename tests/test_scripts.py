"""The pure parts of scripts/reingest.py and scripts/eval_queries.py.

No server, no database, no network: every test here builds its own corpus
under tmp_path or its own YAML file, and the two tests that exercise a code
path which *could* make an HTTP call install a stub client that raises if it
is ever constructed. This module therefore passes with no Docker daemon
running -- it never asks for the `client` or `db_session` fixtures.
"""
from pathlib import Path

import pytest

from scripts import eval_queries, reingest
from scripts.eval_queries import QueriesFileError, load_queries, score
from scripts.reingest import resolve_excludes, select_files

SHIPPED_GOLDEN_QUERIES = Path(__file__).resolve().parent / "fixtures" / "golden_queries.yaml"


class _ExplodingClient:
    """An httpx.Client stand-in that fails the test if it is ever used.

    Constructed with whatever arguments the real client takes, so the failure
    is "you made a request" rather than "you called it wrongly".
    """

    def __init__(self, *args, **kwargs):
        raise AssertionError("an HTTP client was constructed; no request was expected")


def _write(root: Path, rel: str, text: str = "# note\n\nbody\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _corpus(root: Path) -> None:
    """A corpus with both default-excluded files, one at the top level."""
    _write(root, "top.md")
    _write(root, "repo/project/adr/0001-decision.md")
    _write(root, "context/convention.md")
    _write(root, "notes/scratch.draft.md")
    _write(root, "MEMORY.md")
    _write(root, "learning-log.md")  # top level: the PurePath.match hazard
    _write(root, "repo/project/MEMORY.md")
    _write(root, "notes/README.txt")  # not markdown


def _selected_rel(root: Path, excludes) -> list[str]:
    return [path.relative_to(root).as_posix() for path in select_files(root, excludes)]


def test_select_files_finds_markdown_recursively(tmp_path):
    _corpus(tmp_path)

    selected = _selected_rel(tmp_path, [])

    assert "top.md" in selected
    assert "repo/project/adr/0001-decision.md" in selected
    assert "context/convention.md" in selected
    # Only *.md, at any depth.
    assert "notes/README.txt" not in selected
    assert all(name.endswith(".md") for name in selected)


def test_the_default_excludes_drop_memory_indexes_and_the_learning_log(tmp_path):
    _corpus(tmp_path)

    selected = _selected_rel(tmp_path, resolve_excludes([], use_defaults=True))

    assert "MEMORY.md" not in selected
    assert "repo/project/MEMORY.md" not in selected
    # The top-level ledger is the case PurePath.match("**/learning-log.md")
    # does not cover on its own.
    assert "learning-log.md" not in selected
    assert "repo/project/adr/0001-decision.md" in selected


def test_an_explicit_exclude_adds_to_the_defaults_rather_than_replacing_them(tmp_path):
    _corpus(tmp_path)

    excludes = resolve_excludes(["**/*.draft.md"], use_defaults=True)
    selected = _selected_rel(tmp_path, excludes)

    # The user's glob took effect ...
    assert "notes/scratch.draft.md" not in selected
    # ... and did not displace the defaults.
    assert "MEMORY.md" not in selected
    assert "learning-log.md" not in selected
    assert "repo/project/adr/0001-decision.md" in selected


def test_no_default_excludes_admits_the_generated_index_and_the_ledger(tmp_path):
    _corpus(tmp_path)

    selected = _selected_rel(tmp_path, resolve_excludes([], use_defaults=False))

    assert "MEMORY.md" in selected
    assert "repo/project/MEMORY.md" in selected
    assert "learning-log.md" in selected


def test_select_files_returns_posix_relative_paths(tmp_path, capsys):
    _write(tmp_path, "repo/project/adr/0001-decision.md")

    selected = select_files(tmp_path, [])

    # Every selected file is addressable relative to --root, and the relative
    # form the script derives from it is posix even on a backslash host.
    rels = [path.relative_to(tmp_path).as_posix() for path in selected]
    assert rels == ["repo/project/adr/0001-decision.md"]
    assert all("\\" not in rel for rel in rels)

    # And that is what the script actually reports/sends, not just what the
    # test can compute from the return value.
    exit_code = reingest.main(["--root", str(tmp_path), "--dry-run"])
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "repo/project/adr/0001-decision.md" in stdout
    assert "\\" not in stdout


def test_dry_run_makes_no_http_call(tmp_path, monkeypatch, capsys):
    _corpus(tmp_path)
    monkeypatch.setattr(reingest.httpx, "Client", _ExplodingClient)

    exit_code = reingest.main(
        ["--root", str(tmp_path), "--api-key", "nm_test_key", "--dry-run"]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "dry run, no HTTP call made" in stdout


def test_golden_queries_file_parses():
    golden = load_queries(str(SHIPPED_GOLDEN_QUERIES))

    assert golden.gate.top_k >= 1
    assert golden.gate.threshold >= 1
    assert golden.gate.minimum_queries >= 1
    assert len(golden.queries) >= 1


def test_every_golden_query_has_a_query_and_an_expectation():
    golden = load_queries(str(SHIPPED_GOLDEN_QUERIES))

    for entry in golden.queries:
        assert entry.query
        assert entry.expect_path_or_uri
        assert all(isinstance(path, str) and path for path in entry.expect_path_or_uri)


def test_a_golden_query_missing_its_expectation_is_rejected_by_index(tmp_path):
    path = tmp_path / "queries.yaml"
    path.write_text(
        "version: 1\n"
        "gate:\n"
        "  top_k: 5\n"
        "  threshold: 15\n"
        "  minimum_queries: 15\n"
        "queries:\n"
        "  - query: \"first\"\n"
        "    expect_path_or_uri: \"a.md\"\n"
        "  - query: \"second, with no expectation\"\n",
        encoding="utf-8",
    )

    with pytest.raises(QueriesFileError) as exc_info:
        load_queries(str(path))

    # The index names the offending entry: "one of them is wrong" is not an
    # actionable error against a 20-query file.
    assert "queries[1]" in str(exc_info.value)
    assert "expect_path_or_uri" in str(exc_info.value)


def test_score_counts_a_hit_at_rank_five_and_a_miss_at_rank_six():
    results = [{"path_or_uri": f"filler-{index}.md"} for index in range(6)]
    results[4] = {"path_or_uri": "wanted.md"}

    assert score(results, "wanted.md", top_k=5) == 5

    # Same corpus, the wanted note one place further down: outside top_k, so
    # it is a miss rather than a rank-6 hit.
    results[4] = {"path_or_uri": "filler-4.md"}
    results[5] = {"path_or_uri": "wanted.md"}
    assert score(results, "wanted.md", top_k=5) is None


def test_score_accepts_a_list_of_acceptable_paths():
    results = [
        {"path_or_uri": "unrelated.md"},
        {"path_or_uri": "second-acceptable.md"},
    ]

    assert score(results, ["first-acceptable.md", "second-acceptable.md"], top_k=5) == 2
    assert score(results, ["neither.md", "nor-this.md"], top_k=5) is None


def test_an_underpopulated_query_file_reports_the_gate_as_void(monkeypatch, capsys):
    # The shipped fixture carries placeholders only, fewer than
    # gate.minimum_queries. That is the expected slice-1 state: exit 3, no
    # request attempted, and wording that does not read as a failure.
    monkeypatch.setattr(eval_queries.httpx, "Client", _ExplodingClient)

    exit_code = eval_queries.main(
        ["--api-key", "nm_test_key", "--queries", str(SHIPPED_GOLDEN_QUERIES)]
    )

    stdout = capsys.readouterr().out
    assert exit_code == eval_queries.EXIT_NOT_POPULATED == 3
    assert "not yet populated" in stdout
    assert "gate void" in stdout
    assert "GATE FAIL" not in stdout
