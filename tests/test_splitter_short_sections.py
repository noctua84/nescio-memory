"""Regression tests for the short-section run fix in ``app.core.splitter``.

Before ``SPLITTER_VERSION == "3"``, a document whose *leading* sections were
all under ``MIN_SECTION_CHARS`` merged every one of them forward into a single
unit labelled with the first section's heading. For a daily log -- hundreds of
two-line ``## <date> entry`` sections -- that produced one oversized unit, and
every passage packed out of it carried the first entry's heading path. Since
``heading_path`` is the weight-A full-text-search text *and* prefixes the
embedding input, a wrong label poisons both halves of retrieval.

The corrected contract: a pending run of short sections is flushed as its own
unit once adding the next section would push it past ``TARGET_CHARS``, and
that unit is labelled with its own first heading. A short leading run followed
by a *long* section still merges forward into it (unchanged behaviour, guarded
below), because that run is genuinely preamble to the long section.

No fixtures, no database, no Docker: every body here is built inline and is
entirely synthetic.
"""

import re

import pytest

from app.core.splitter import (
    MAX_CHARS, MIN_SECTION_CHARS, TARGET_CHARS, split_note)

# The shape that motivated the fix: a promotion ledger / daily log.
LOG_ENTRIES = 400


def log_body(entries: int = LOG_ENTRIES) -> str:
    """A run of ``entries`` short ``##`` sections, each well under the floor."""
    return "".join(
        f"## 2026-01-01 entry {index}\n\n"
        f"short text number {index} here with a few more words.\n\n"
        for index in range(entries)
    )


def heading_lines(body: str) -> list[str]:
    """Every ``## `` heading line in ``body``, in document order."""
    return [line for line in body.split("\n") if line.startswith("## ")]


def assert_label_is_inside_its_own_passage(passages) -> None:
    """Every passage's heading path names a heading in that same passage.

    This is the assertion the pre-fix splitter cannot satisfy: it labelled 21
    passages ``2026-01-01 entry 0`` while only the first contained that
    heading.
    """
    for passage in passages:
        assert passage.heading_path is not None, passage.ordinal
        assert f"## {passage.heading_path}" in passage.content, (
            f"passage {passage.ordinal} is labelled {passage.heading_path!r} "
            "but does not contain that heading"
        )


class TestLongRunOfShortSections:
    """The 400-entry log: the case the fix exists for."""

    def test_every_passage_of_a_long_log_carries_its_own_heading_path(self):
        body = log_body()
        assert len(body) > TARGET_CHARS

        passages = split_note(body).passages

        # One distinct label per passage: no two passages share a heading
        # path, which is what "collapsed under one heading" looked like.
        labels = [passage.heading_path for passage in passages]
        assert len(set(labels)) == len(passages)
        assert len(passages) > 1
        assert_label_is_inside_its_own_passage(passages)

    def test_no_passage_of_a_long_log_exceeds_the_target(self):
        passages = split_note(log_body()).passages

        for passage in passages:
            assert len(passage.content) <= TARGET_CHARS, passage.ordinal

    def test_ordinals_stay_contiguous_across_the_flushed_groups(self):
        passages = split_note(log_body()).passages

        assert [passage.ordinal for passage in passages] == list(
            range(len(passages))
        )

    def test_every_section_heading_survives_exactly_once(self):
        body = log_body()
        expected = heading_lines(body)
        assert len(expected) == LOG_ENTRIES

        passages = split_note(body).passages

        # Flushing a pending group must neither drop a section nor emit one
        # twice: the concatenation of the passages still holds each heading
        # exactly once.
        concatenated = "\n".join(passage.content for passage in passages)
        produced = heading_lines(concatenated)
        assert produced == expected

    def test_every_entry_body_survives_exactly_once(self):
        body = log_body()
        concatenated = "\n".join(
            passage.content for passage in split_note(body).passages
        )

        for index in range(LOG_ENTRIES):
            needle = f"short text number {index} here with a few more words."
            assert concatenated.count(needle) == 1, index


class TestShortRunInASmallerDocument:
    """~80 short sections in ~3 KB: the same bug, one order down."""

    def test_eighty_short_sections_in_three_kilobytes_split_and_stay_labelled(
        self,
    ):
        body = "".join(
            f"## Item {index}\n\nline {index} of the log, briefly noted.\n\n"
            for index in range(80)
        )
        assert TARGET_CHARS < len(body) < 2 * MAX_CHARS

        passages = split_note(body).passages

        # The pre-fix splitter produced a single unit here too; its packing
        # then emitted several passages, all labelled "Item 0".
        assert len(passages) > 1
        labels = [passage.heading_path for passage in passages]
        assert len(set(labels)) == len(passages)
        assert_label_is_inside_its_own_passage(passages)
        for passage in passages:
            assert len(passage.content) <= TARGET_CHARS


class TestForwardMergeIsPreserved:
    """The fix must not break the case the forward merge exists for."""

    def test_a_short_leading_run_still_joins_the_following_long_section(self):
        preamble_sections = (
            "## Status\n\naccepted.\n\n"
            "## Owner\n\nthe retrieval team.\n\n"
        )
        long_section = "Prose of the long section, well over the merge floor. " * 30
        body = f"{preamble_sections}## Decision\n\n{long_section}"
        assert len(body) > TARGET_CHARS
        assert len("accepted.") < MIN_SECTION_CHARS

        passages = split_note(body).passages

        # The short run was preamble to the long section, so it is absorbed by
        # it and the merged unit takes the long section's heading path. It is
        # NOT flushed as a group of its own.
        assert len(passages) == 1
        assert passages[0].heading_path == "Decision"
        assert "## Status" in passages[0].content
        assert "## Owner" in passages[0].content

    def test_a_short_run_long_enough_to_flush_does_not_steal_the_long_label(
        self,
    ):
        filler = "Padding sentence that keeps this section short of the floor. " * 4
        assert len(filler) < MIN_SECTION_CHARS
        short_run = "\n\n".join(
            f"## Note {index}\n\n{filler}" for index in range(1, 8)
        )
        long_section = "Prose of the long section, well over the merge floor. " * 30
        body = f"{short_run}\n\n## Decision\n\n{long_section}"

        passages = split_note(body).passages

        # The run outgrew TARGET_CHARS before reaching "## Decision", so at
        # least one group was flushed under its own first heading, and the
        # "Decision" label is not applied to text that predates it.
        labels = [passage.heading_path for passage in passages]
        assert labels[0] == "Note 1"
        assert "Decision" in labels
        decision = next(p for p in passages if p.heading_path == "Decision")
        assert "## Note 1" not in decision.content
        assert_label_is_inside_its_own_passage(passages)


@pytest.mark.parametrize("entries", [2, 5, 20, 120, 400])
def test_no_short_section_run_length_loses_or_duplicates_a_heading(entries):
    """Across run lengths, the section set is conserved exactly."""
    body = log_body(entries)
    expected = heading_lines(body)

    passages = split_note(body).passages

    concatenated = "\n".join(passage.content for passage in passages)
    assert heading_lines(concatenated) == expected
    assert len(passages) >= 1
    for passage in passages:
        assert len(passage.content) <= MAX_CHARS
    # Nothing but whitespace is added or lost at the joins.
    assert re.sub(r"\s+", " ", concatenated).strip() == re.sub(
        r"\s+", " ", body
    ).strip()
