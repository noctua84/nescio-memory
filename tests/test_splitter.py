"""Pure unit tests for ``app.core.splitter``.

No ``db_session``, no ``client``, no container fixture: the splitter imports
neither configuration nor persistence, so these tests run with no Docker
daemon.

Fixtures live in ``tests/fixtures/notes/`` and are **all synthetic**: every one
is written for this repository, on an invented subject, and none reproduces
text, names or topics from any private corpus. This repository is public, so no
real note may ever be committed as a fixture. What each fixture exercises:

``large-adr.md``
    Frontmatter with name/description/``type: adr``, no provenance line, an
    H1 + status block before the first ``##`` (the preamble), 25 ``##``/``###``
    headings including ``###`` levels that must join to their parent ``##``, a
    ``## Consequences`` section under ``MIN_SECTION_CHARS`` that must be merged
    into its neighbour, fenced code blocks, and one section over ``MAX_CHARS``
    so the packing branch runs. Yields more than twenty passages.
``small-note.md``
    A note under 1,500 characters with both frontmatter (``type: feedback``)
    and a ``[Source: empirical - 2026-09-30]`` provenance line.
``structured-3kb.md``
    A ~3 KB note with six ``##`` sections: structure worth splitting at, in the
    same size band as ``headingless-3kb``.
``headingless-3kb.md``
    The control for the above -- same band, no headings, stays whole.
``headingless-large.md``
    Past ``MAX_CHARS`` with no headings at all: paragraph packing only.
``no-frontmatter.md``, ``malformed-frontmatter.md``
    The two metadata degenerate cases.

Exact sizes, recorded so a later threshold change fails loudly rather than
drifts (characters; file read as UTF-8 / ``body`` as returned by
``split_note``):

    large-adr                         19,624/ 19,253   malformed-frontmatter  528/527
    headingless-3kb                    2,896/  2,692   no-frontmatter       1,033/1,032
    headingless-large                 12,350/ 12,349   small-note           1,311/  966
    structured-3kb                     3,486/  3,186
"""

import re
from datetime import date
from pathlib import Path

import pytest

from app.core.splitter import (
    MAX_CHARS, MIN_SECTION_CHARS, SPLITTER_VERSION, TARGET_CHARS, split_note)

FIXTURES = Path(__file__).parent / "fixtures" / "notes"

# Every fixture, so the invariant tests below cannot silently miss a new one.
ALL_FIXTURES = sorted(path.name for path in FIXTURES.glob("*.md"))
assert len(ALL_FIXTURES) == 7, ALL_FIXTURES

HEADING_SEP = " › "


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def collapse(text: str) -> str:
    """Whitespace-insensitive view of a body or a passage concatenation."""
    return re.sub(r"\s+", " ", text).strip()


def normalised_body(content: str) -> str:
    """The body ``split_note`` must return for a document with no frontmatter.

    Leading blank lines removed, trailing whitespace stripped. Indentation on
    the first content line is preserved, because it may open a code block.
    """
    lines = content.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip()


class TestConstants:
    def test_the_published_constants_are_the_agreed_values(self):
        assert SPLITTER_VERSION == "3"
        assert TARGET_CHARS == 1_500
        assert MAX_CHARS == 4_000
        assert MIN_SECTION_CHARS == 300


class TestSplitDecision:
    """Decision 11: structure licenses a split, size alone does not."""

    def test_a_body_at_or_below_the_target_is_one_whole_passage(self):
        result = split_note(read_fixture("small-note.md"))
        assert len(result.body) <= TARGET_CHARS
        assert len(result.passages) == 1
        passage = result.passages[0]
        assert passage.ordinal == 0
        assert passage.heading_path is None
        assert passage.content == result.body

    def test_a_structured_three_kilobyte_note_splits_at_its_headings(self):
        result = split_note(read_fixture("structured-3kb.md"))
        assert TARGET_CHARS < len(result.body) <= MAX_CHARS
        assert len(result.passages) > 1
        paths = {p.heading_path for p in result.passages if p.heading_path is not None}
        assert len(paths) >= 2, paths

    def test_a_headingless_three_kilobyte_note_stays_whole(self):
        result = split_note(read_fixture("headingless-3kb.md"))
        assert TARGET_CHARS < len(result.body) <= MAX_CHARS
        assert len(result.passages) == 1
        assert result.passages[0].heading_path is None

    def test_a_headed_body_below_the_target_is_not_split(self):
        filler = "Prose that carries the section without reaching the target. "
        half = filler * 7
        body = f"## First section\n\n{half}\n\n## Second section\n\n{half}"
        assert 800 <= len(body) <= TARGET_CHARS, len(body)
        result = split_note(body)
        assert len(result.passages) == 1
        assert result.passages[0].heading_path is None
        assert result.passages[0].content == result.body

    def test_a_headingless_oversized_body_is_packed_by_paragraphs(self):
        result = split_note(read_fixture("headingless-large.md"))
        assert len(result.body) > MAX_CHARS
        assert len(result.passages) > 1
        assert all(p.heading_path is None for p in result.passages)
        assert all(len(p.content) <= MAX_CHARS for p in result.passages)

    def test_a_single_paragraph_over_the_cap_is_hard_split(self):
        body = "## One section\n\n" + "x" * 10_000
        result = split_note(body)
        assert len(result.passages) > 1
        assert all(len(p.content) <= MAX_CHARS for p in result.passages)
        assert max(len(p.content) for p in result.passages) == MAX_CHARS


class TestMetadata:
    def test_frontmatter_fields_are_lifted_off_the_body(self):
        result = split_note(read_fixture("small-note.md"))
        assert result.name == (
            "Resetting a fake clock does not drain its queued one-shot callbacks"
        )
        assert result.description is not None and result.description != ""
        assert result.type == "feedback"
        assert not result.body.startswith("---")
        assert "type: feedback" not in result.body

    def test_only_string_frontmatter_values_are_kept(self):
        content = (
            "---\nname: 2026-10-29\ndescription:\n  - a\n  - b\n"
            "type: note\n---\nBody.\n"
        )
        result = split_note(content)
        assert result.name is None
        assert result.description is None
        assert result.type == "note"

    def test_the_provenance_line_is_parsed_into_typed_fields(self):
        result = split_note(read_fixture("small-note.md"))
        assert result.provenance_kind == "empirical"
        assert result.provenance_date == date(2026, 9, 30)
        assert "[Source: empirical — 2026-09-30]" in result.body
        assert "[Source: empirical — 2026-09-30]" in "".join(
            p.content for p in result.passages
        )

    @pytest.mark.parametrize(
        ("marker", "kind", "expected"),
        [
            ("[Source: empirical — 2026-09-30]", "empirical", date(2026, 9, 30)),
            ("[Source: user override - 2025-01-02]", "user override", date(2025, 1, 2)),
            (
                "[Source: agent inference — 2024-12-31]",
                "agent inference",
                date(2024, 12, 31),
            ),
        ],
    )
    def test_an_em_dash_and_a_hyphen_are_both_accepted_in_the_provenance_line(
        self, marker, kind, expected
    ):
        result = split_note(f"A short note.\n{marker}\n")
        assert result.provenance_kind == kind
        assert result.provenance_date == expected

    def test_an_unparseable_provenance_date_yields_no_provenance(self):
        result = split_note("A short note.\n[Source: empirical — 2026-13-45]\n")
        assert result.provenance_kind is None
        assert result.provenance_date is None

    def test_a_body_without_a_provenance_line_has_neither_field(self):
        result = split_note(read_fixture("large-adr.md"))
        assert result.provenance_kind is None
        assert result.provenance_date is None

    def test_a_note_without_frontmatter_splits_anyway(self):
        content = read_fixture("no-frontmatter.md")
        result = split_note(content)
        assert result.name is None
        assert result.description is None
        assert result.type is None
        assert result.body == normalised_body(content)
        assert len(result.passages) >= 1

    def test_malformed_frontmatter_does_not_raise(self):
        content = read_fixture("malformed-frontmatter.md")
        result = split_note(content)
        assert result.name is None
        assert result.description is None
        assert result.type is None
        # The delimiter lines stay in the body: nothing the parser failed to
        # understand is silently dropped.
        assert result.body == normalised_body(content)
        assert result.body.startswith("---")
        assert len(result.passages) >= 1

    def test_frontmatter_delimiters_tolerate_trailing_whitespace(self):
        result = split_note("--- \r\nname: x\r\ntype: note\r\n--- \r\nBody.\r\n")
        assert result.name == "x"
        assert result.type == "note"
        assert "name: x" not in result.body


class TestLargeAdr:
    def test_the_large_adr_splits_at_headings(self):
        result = split_note(read_fixture("large-adr.md"))
        assert len(result.passages) > 20
        # The H1 + status block before the first `##` is the preamble.
        assert result.passages[0].heading_path is None
        assert result.passages[0].content.startswith("# ADR-0007:")

        joined = [
            p.heading_path
            for p in result.passages
            if p.heading_path is not None and HEADING_SEP in p.heading_path
        ]
        assert joined, "expected at least one `>`-joined heading path"

        # The left-hand side of a joined path is the nearest preceding `##`
        # title, so it must be a real H2 heading of this document. (It need not
        # survive as some other passage's whole heading_path: an H2 whose own
        # text is under MIN_SECTION_CHARS is merged into the previous unit by
        # step 6, which is exactly what `## Consequences` does here.)
        h2_titles = {
            line[3:].strip()
            for line in result.body.split("\n")
            if line.startswith("## ")
        }
        for path in joined:
            assert path.split(HEADING_SEP)[0] in h2_titles, path

    def test_the_large_adr_keeps_its_frontmatter_fields(self):
        result = split_note(read_fixture("large-adr.md"))
        assert result.name == "example-project-adr-0007-bounded-cache-eviction"
        assert result.type == "adr"
        assert result.description is not None


class TestInvariants:
    @pytest.mark.parametrize("name", ALL_FIXTURES)
    def test_no_passage_exceeds_the_maximum(self, name):
        result = split_note(read_fixture(name))
        assert result.passages
        for passage in result.passages:
            assert len(passage.content) <= MAX_CHARS, (
                f"{name} ordinal {passage.ordinal} is {len(passage.content)} chars"
            )

    @pytest.mark.parametrize("name", ALL_FIXTURES)
    def test_the_passages_reproduce_the_body(self, name):
        result = split_note(read_fixture(name))
        joined = "\n\n".join(p.content for p in result.passages)
        assert collapse(joined) == collapse(result.body)

    @pytest.mark.parametrize("name", ALL_FIXTURES)
    def test_ordinals_are_contiguous_from_zero(self, name):
        result = split_note(read_fixture(name))
        ordinals = [p.ordinal for p in result.passages]
        assert ordinals == list(range(len(result.passages)))

    @pytest.mark.parametrize("name", ALL_FIXTURES)
    def test_passages_do_not_overlap(self, name):
        """v1 emitted overlapping windows; v2 must not."""
        result = split_note(read_fixture(name))
        for left, right in zip(result.passages, result.passages[1:]):
            if not left.content or not right.content:
                continue
            assert left.content not in right.content
            assert right.content not in left.content

    @pytest.mark.parametrize("name", ALL_FIXTURES)
    def test_the_frozen_dataclasses_reject_mutation(self, name):
        result = split_note(read_fixture(name))
        with pytest.raises(Exception):
            result.passages[0].ordinal = 99  # type: ignore[misc]
        with pytest.raises(Exception):
            result.body = "x"  # type: ignore[misc]


class TestSectioning:
    def test_headings_inside_a_fenced_code_block_are_not_split_points(self):
        prose = "Explanatory prose that pads the body past the target. " * 30
        fence = "```markdown\n## not a heading\n### also not a heading\n```"
        body = f"Preamble.\n\n{fence}\n\n{prose}"
        assert len(body) > TARGET_CHARS
        result = split_note(body)
        # The fenced lines are the body's only `##`-looking lines, so step 4
        # treats the document as headingless: one unit, no heading path.
        assert len(result.passages) == 1
        assert result.passages[0].heading_path is None
        assert "## not a heading" in result.passages[0].content

        # And with a real heading present, the fenced one still creates no
        # section of its own.
        with_real = "## Real heading\n\n" + body
        paths = {p.heading_path for p in split_note(with_real).passages}
        assert "not a heading" not in paths
        assert "also not a heading" not in paths

    def test_an_h1_is_not_a_split_point(self):
        prose = "Status prose that pads the preamble past the target. " * 20
        body = f"# Title\n\n{prose}\n\n# Another title\n\n{prose}"
        result = split_note(body)
        assert all(p.heading_path is None for p in result.passages)

    def test_a_third_level_heading_is_joined_to_its_parent(self):
        section = "Body text for the subsection, long enough to stand alone. " * 16
        sub = "E-2: no eviction pass may run on a background thread"
        body = f"## Consequences\n\n{section}\n\n### {sub}\n\n{section}"
        assert len(body) > TARGET_CHARS
        paths = {p.heading_path for p in split_note(body).passages}
        assert f"Consequences{HEADING_SEP}{sub}" in paths

    def test_a_third_level_heading_with_no_parent_stands_alone(self):
        section = "Body text for the subsection, long enough to stand alone. " * 16
        body = f"### Alpha\n\n{section}\n\n### Beta\n\n{section}"
        assert len(body) > TARGET_CHARS
        paths = {p.heading_path for p in split_note(body).passages}
        assert paths == {"Alpha", "Beta"}

    def test_a_tiny_section_is_merged_into_its_neighbour(self):
        preamble = "Preamble prose that is its own unit, comfortably long. " * 15
        long_section = "Prose of the long section, well over the merge floor. " * 16
        tiny = "A forty-ish character status line here."
        body = f"{preamble}\n\n## Long section\n\n{long_section}\n\n## Tiny\n\n{tiny}"
        assert len(body) > TARGET_CHARS
        assert len(tiny) < MIN_SECTION_CHARS

        result = split_note(body)
        holders = [p for p in result.passages if p.heading_path == "Long section"]
        assert len(holders) == 1
        assert tiny in holders[0].content
        assert "## Tiny" in holders[0].content
        # The short section does not get a passage of its own.
        assert all(p.heading_path != "Tiny" for p in result.passages)

    def test_a_run_of_short_sections_is_grouped_without_mislabelling(self):
        # Every section is under MIN_SECTION_CHARS but the body is over
        # TARGET_CHARS, so the section branch runs and the first section has no
        # "next" unit. Full contract: tests/test_splitter_short_sections.py.
        filler = "Just under the merge floor, so this section cannot stand. " * 4
        assert len(filler) < MIN_SECTION_CHARS
        body = "\n\n".join(f"## Section {n}\n\n{filler}" for n in range(1, 8))
        assert len(body) > TARGET_CHARS
        result = split_note(body)
        assert result.passages[0].heading_path == "Section 1"
        for passage in result.passages:
            assert len(passage.content) <= TARGET_CHARS
            assert f"## {passage.heading_path}" in passage.content  # own label


class TestDegenerateInput:
    def test_an_empty_body_still_yields_one_passage(self):
        result = split_note("")
        assert result.body == ""
        assert len(result.passages) == 1
        assert result.passages[0].ordinal == 0
        assert result.passages[0].heading_path is None
        assert result.passages[0].content == ""

        with_frontmatter = split_note("---\nname: x\n---\n")
        assert with_frontmatter.name == "x"
        assert with_frontmatter.body == ""
        assert len(with_frontmatter.passages) == 1
        assert with_frontmatter.passages[0].content == ""

    def test_a_whitespace_only_body_yields_one_empty_passage(self):
        result = split_note("   \n\n\t\n")
        assert result.body == ""
        assert len(result.passages) == 1
        assert result.passages[0].content == ""

    def test_frontmatter_with_no_closing_delimiter_is_not_frontmatter(self):
        content = "---\nname: x\ntype: note\nBody text with no closing fence.\n"
        result = split_note(content)
        assert result.name is None
        assert result.type is None
        assert result.body == normalised_body(content)
