"""Tests for the --out / --queries path-confinement added to eval/__main__.py.

WHY THIS FILE IS NOT UNDER tests/
----------------------------------
eval/__init__.py documents a hard boundary: this package imports nothing from
`tests/`, and `tests/` imports nothing from `eval/`, so the two can drift
apart without one faking the other's embedder. Putting a test here that
imports eval.__main__ would cross that boundary from the other direction.

This file tests pure path arithmetic -- no embedder, no database, no network
-- so it carries none of the reasons eval/ is excluded from CI (see
eval/__init__.py: "CI has no model"). It is still kept out of the default
`pytest -q` collection (pyproject.toml's testpaths = ["tests"] does not reach
here) so that boundary stays simple to state. Run it explicitly:

    uv run pytest -q eval/test_cli_paths.py

SonarCloud findings this covers (PR #34, rule "Agentic workflows should not be
vulnerable to path injection attacks"):
  * eval/queries.py:76  (load_query_set reads an operator-supplied path)
  * eval/report.py:472  (write_reports writes to an operator-supplied path)
"""
import pytest

from eval.__main__ import (
    REPO_ROOT,
    confine_out_dir,
    path_is_confined,
    resolve_query_set,
)
from eval.queries import QuerySetError
from eval.report import ReportError

SYNTHETIC_QUERIES = REPO_ROOT / "eval" / "corpora" / "synthetic" / "queries.yaml"


# --------------------------------------------------------------------------
# path_is_confined -- the shared helper
# --------------------------------------------------------------------------


def test_path_is_confined_true_for_a_path_inside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "sub" / "file.yaml"
    assert path_is_confined(inside, root)


def test_path_is_confined_true_for_the_root_itself(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert path_is_confined(root, root)


def test_path_is_confined_false_for_a_traversal_out_of_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    # The textbook payload: climb out of `root` via "..".
    escaped = root / ".." / "escaped.yaml"
    assert not path_is_confined(escaped, root)


def test_path_is_confined_checks_every_root_given(tmp_path):
    repo = tmp_path / "repo"
    corpus = tmp_path / "corpus"
    repo.mkdir()
    corpus.mkdir()
    outside = tmp_path / "elsewhere" / "file.yaml"
    assert path_is_confined(corpus / "file.yaml", repo, corpus)
    assert not path_is_confined(outside, repo, corpus)


# --------------------------------------------------------------------------
# --out, via confine_out_dir
# --------------------------------------------------------------------------


def test_default_out_dir_is_accepted():
    # DEFAULT_OUT (eval-out/) must keep working -- it is what every
    # undecorated `python -m eval` invocation writes to.
    confine_out_dir(REPO_ROOT / "eval-out")  # must not raise


def test_out_dir_traversal_is_rejected():
    traversal = REPO_ROOT / "eval-out" / ".." / ".." / ".." / "escaped-out"
    with pytest.raises(ReportError, match="must resolve inside the repository"):
        confine_out_dir(traversal)


def test_out_dir_outside_the_repo_is_rejected(tmp_path):
    # Not a ".."-shaped payload -- an absolute path elsewhere entirely, which
    # a naive `".." in str(path)` check would wave through.
    outside = tmp_path / "somewhere-else"
    with pytest.raises(ReportError, match="must resolve inside the repository"):
        confine_out_dir(outside)


# --------------------------------------------------------------------------
# --queries, via resolve_query_set
# --------------------------------------------------------------------------


def test_queries_inside_the_repo_is_accepted(tmp_path):
    # Legitimate home #1: a path inside this repository (e.g. the gitignored
    # eval/private/), with --corpus pointed at an unrelated directory outside
    # the repo entirely -- the real-brain shape from eval/README.md.
    corpus = tmp_path / "brain"
    corpus.mkdir()
    resolved = resolve_query_set(corpus, SYNTHETIC_QUERIES)
    assert resolved == SYNTHETIC_QUERIES


def test_queries_inside_the_corpus_dir_is_accepted(tmp_path):
    # Legitimate home #2: <corpus>/queries.yaml, the default lookup -- and
    # also valid if named explicitly.
    corpus = tmp_path / "brain"
    corpus.mkdir()
    query_file = corpus / "queries.yaml"
    query_file.write_text("queries: []\n", encoding="utf-8")

    assert resolve_query_set(corpus, None) == query_file
    assert resolve_query_set(corpus, query_file) == query_file


def test_queries_traversal_out_of_the_corpus_dir_is_rejected(tmp_path):
    corpus = tmp_path / "brain"
    corpus.mkdir()
    # Climbs out of `corpus` via "..", landing on a sibling file that is also
    # outside the repository -- outside both legitimate homes at once.
    escaped = corpus / ".." / "secrets" / "queries.yaml"
    with pytest.raises(QuerySetError, match="must resolve inside the repository"):
        resolve_query_set(corpus, escaped)


def test_queries_outside_both_the_repo_and_the_corpus_is_rejected(tmp_path):
    corpus = tmp_path / "brain"
    corpus.mkdir()
    unrelated = tmp_path / "elsewhere" / "queries.yaml"
    with pytest.raises(QuerySetError, match="must resolve inside the repository"):
        resolve_query_set(corpus, unrelated)


# --------------------------------------------------------------------------
# --corpus stays unconstrained -- pinned so a future "hardening" pass notices
# it would break the harness's actual job (a real brain living outside the
# repository entirely).
# --------------------------------------------------------------------------


def test_corpus_outside_the_repository_is_still_accepted(tmp_path):
    corpus = tmp_path / "a-real-brain-directory"
    corpus.mkdir()
    assert not path_is_confined(corpus, REPO_ROOT)

    query_file = corpus / "queries.yaml"
    query_file.write_text("queries: []\n", encoding="utf-8")
    # No confinement check is ever applied to `corpus` itself -- only to the
    # query-set path resolved from it. Resolving the default query set
    # against this out-of-repo corpus must still succeed.
    assert resolve_query_set(corpus, None) == query_file
