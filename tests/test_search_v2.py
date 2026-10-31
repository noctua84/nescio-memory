"""POST /api/v2/search: what it finds and in what order.

Search is hybrid -- a full-text arm and a vector arm fused by reciprocal rank
-- so a test that asserts pure full-text semantics (which words match, the
A > B > C weight ordering, phrases, exclusions, an empty tsquery) requests the
`fts_only` fixture. That is VECTOR_WEIGHT=0, the same switch the FTS-only
baseline uses, and it pins the FTS arm: without it the vector arm, which
always fills up to top_k, would answer an exact-result-set assertion for the
wrong reason. Tests with no such dependency run hybrid, as production does.
The fused ranking itself is pinned at the repository level in
tests/test_search_hybrid.py; the stop-word query answered by the vector arm
and the transaction guards are in tests/test_search_contract.py (split out
for length); tenant isolation is in tests/test_isolation_v2*.py.
Every response here is asserted non-degraded: the `client` fixture fakes the
query embedder, so a degraded answer would mean the vector arm silently
dropped out.

The ranking tests depend on ts_rank_cd's default weight vector
{D, C, B, A} = {0.1, 0.2, 0.4, 1.0} and on which text goes at which weight:
A = note name + heading path, B = passage content, C = note description.
Together, test_a_content_match_outranks_a_description_only_match and
test_name_and_heading_matches_outrank_content_matches pin the whole A > B > C
ordering, which is otherwise an untested opinion.
"""
from datetime import date

from tests.factories import make_api_key, make_note, make_passage


def _search(client, key, q, **overrides):
    payload = {"q": q, "top_k": 10}
    payload.update(overrides)
    return client.post("/api/v2/search", json=payload, headers={"X-API-Key": key})


def _results(response):
    assert response.status_code == 200, response.text
    assert response.json()["degraded"] is False
    return response.json()["results"]


def test_search_finds_a_passage_by_a_word_in_its_content(client, db_session, fts_only):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    passage = make_passage(
        db_session, note, content="The pelicanium isotope decays slowly."
    )

    results = _results(_search(client, key, "pelicanium"))

    assert [r["passage_id"] for r in results] == [passage.id]
    assert results[0]["content"] == "The pelicanium isotope decays slowly."
    assert results[0]["score"] > 0


def test_search_finds_a_passage_by_a_word_in_its_notes_name(
    client, db_session, fts_only
):
    # The A-weight section's first half: the term appears only in the note's
    # name, nowhere in the passage text.
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme", name="Pelicanium handling guide")
    passage = make_passage(db_session, note, content="Store it in a cool place.")

    results = _results(_search(client, key, "pelicanium"))

    assert [r["passage_id"] for r in results] == [passage.id]
    assert results[0]["name"] == "Pelicanium handling guide"


def test_search_finds_a_passage_by_a_word_in_its_heading_path(
    client, db_session, fts_only
):
    # The other half of A.
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    passage = make_passage(
        db_session,
        note,
        heading_path="Storage › Pelicanium",
        content="Store it in a cool place.",
    )

    results = _results(_search(client, key, "pelicanium"))

    assert [r["passage_id"] for r in results] == [passage.id]
    assert results[0]["heading_path"] == "Storage › Pelicanium"


def test_search_finds_a_passage_by_a_word_in_its_notes_description(
    client, db_session, fts_only
):
    # The C-weight section: description reaches the index only through it, so
    # without that section this query finds nothing at all.
    key = make_api_key(db_session, "acme")
    note = make_note(
        db_session, "acme", description="Everything about pelicanium decay."
    )
    passage = make_passage(db_session, note, content="Store it in a cool place.")

    results = _results(_search(client, key, "pelicanium"))

    assert [r["passage_id"] for r in results] == [passage.id]
    assert results[0]["description"] == "Everything about pelicanium decay."


def test_a_content_match_outranks_a_description_only_match(
    client, db_session, fts_only
):
    # Decision 9 asserted rather than assumed: description sits at C (0.2),
    # below content's B (0.4), because a note-level description is copied onto
    # every passage of its note and would otherwise outrank the one passage
    # that actually discusses the term.
    key = make_api_key(db_session, "acme")
    content_note = make_note(db_session, "acme", path_or_uri="docs/content.md")
    content_passage = make_passage(
        db_session,
        content_note,
        content="The pelicanium isotope decays slowly in storage.",
    )
    description_note = make_note(
        db_session,
        "acme",
        path_or_uri="docs/description.md",
        description="Everything about pelicanium decay.",
    )
    description_passage = make_passage(
        db_session, description_note, content="Store it in a cool place."
    )

    results = _results(_search(client, key, "pelicanium"))

    assert [r["passage_id"] for r in results] == [
        content_passage.id,
        description_passage.id,
    ]
    assert results[0]["score"] > results[1]["score"]


def test_name_and_heading_matches_outrank_content_matches(client, db_session, fts_only):
    # The A end of the same ordering: name + heading path sit at weight 1.0.
    key = make_api_key(db_session, "acme")
    a_note = make_note(
        db_session,
        "acme",
        path_or_uri="docs/named.md",
        name="Pelicanium handling guide",
    )
    a_passage = make_passage(
        db_session,
        a_note,
        heading_path="Storage › Pelicanium",
        content="Store it in a cool place.",
    )
    b_note = make_note(db_session, "acme", path_or_uri="docs/content.md")
    b_passage = make_passage(
        db_session, b_note, content="The pelicanium isotope decays slowly in storage."
    )

    results = _results(_search(client, key, "pelicanium"))

    assert [r["passage_id"] for r in results] == [a_passage.id, b_passage.id]
    assert results[0]["score"] > results[1]["score"]


def test_a_quoted_phrase_matches_only_an_adjacent_occurrence(
    client, db_session, fts_only
):
    # The websearch_to_tsquery behaviour that motivates it over
    # plainto_tsquery: a quoted phrase becomes a <-> (adjacency) query, so the
    # passage with the same two words far apart does not match.
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    adjacent = make_passage(
        db_session,
        note,
        ordinal=0,
        content="The pelicanium isotope was catalogued last spring.",
    )
    make_passage(
        db_session,
        note,
        ordinal=1,
        content=(
            "The pelicanium store was audited, and separately an isotope "
            "of another element was catalogued."
        ),
    )

    results = _results(_search(client, key, '"pelicanium isotope"'))

    assert [r["passage_id"] for r in results] == [adjacent.id]


def test_a_negated_term_excludes_a_passage(client, db_session, fts_only):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    kept = make_passage(
        db_session, note, ordinal=0, content="Pelicanium nests on temperate coasts."
    )
    make_passage(
        db_session,
        note,
        ordinal=1,
        content="Pelicanium colonies on the antarctic shelf are rare.",
    )

    results = _results(_search(client, key, "pelicanium -antarctic"))

    assert [r["passage_id"] for r in results] == [kept.id]


def test_top_k_limits_the_number_of_results(client, db_session):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    for ordinal in range(4):
        make_passage(
            db_session,
            note,
            ordinal=ordinal,
            content=f"Pelicanium note number {ordinal} with some filler text.",
        )

    # Hybrid on: the vector arm alone would return all four, so the cap is
    # what holds this to two.
    results = _results(_search(client, key, "pelicanium", top_k=2))

    assert len(results) == 2


def test_results_are_ordered_by_descending_score(client, db_session, fts_only):
    # Three deliberately different weights, so the order is forced rather
    # than tied: A > B > C. With the vector arm off, each result reports its
    # FTS rank and a null vector rank -- that arm did not run.
    key = make_api_key(db_session, "acme")
    a_note = make_note(
        db_session, "acme", path_or_uri="docs/a.md", name="Pelicanium guide"
    )
    a_passage = make_passage(db_session, a_note, content="Keep it cool.")
    b_note = make_note(db_session, "acme", path_or_uri="docs/b.md")
    b_passage = make_passage(
        db_session, b_note, content="The pelicanium isotope decays slowly."
    )
    c_note = make_note(
        db_session,
        "acme",
        path_or_uri="docs/c.md",
        description="All about pelicanium.",
    )
    c_passage = make_passage(db_session, c_note, content="Keep it dry.")

    results = _results(_search(client, key, "pelicanium"))

    scores = [r["score"] for r in results]
    assert [r["passage_id"] for r in results] == [
        a_passage.id,
        b_passage.id,
        c_passage.id,
    ]
    assert scores[0] > scores[1] > scores[2]
    assert [r["fts_rank"] for r in results] == [1, 2, 3]
    assert [r["vector_rank"] for r in results] == [None, None, None]


def test_repo_filter_restricts_to_one_scope(client, db_session):
    # Hybrid on, deliberately: both passages match both arms, so the filter
    # must hold on the vector arm too, not only on the full-text one.
    key = make_api_key(db_session, "acme")
    wanted_note = make_note(db_session, "acme", scope="repo_a")
    wanted = make_passage(
        db_session, wanted_note, content="Pelicanium lives in repo_a."
    )
    other_note = make_note(db_session, "acme", scope="repo_b")
    other = make_passage(
        db_session, other_note, content="Pelicanium lives in repo_b too."
    )

    filtered = _results(_search(client, key, "pelicanium", repo_filter="repo_a"))
    assert [r["passage_id"] for r in filtered] == [wanted.id]

    # The empty string is not a filter: both scopes come back.
    unfiltered = _results(_search(client, key, "pelicanium", repo_filter=""))
    assert {r["passage_id"] for r in unfiltered} == {wanted.id, other.id}


def test_an_empty_tsquery_returns_no_results_rather_than_an_error(
    client, db_session, fts_only
):
    # The FTS arm alone: a stop-word-only query reduces to an empty tsquery,
    # which matches nothing and must not raise.
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    make_passage(db_session, note, content="The pelicanium isotope decays slowly.")

    assert _results(_search(client, key, "the")) == []


def test_search_returns_the_notes_description_provenance_and_status_on_each_result(
    client, db_session
):
    key = make_api_key(db_session, "acme")
    note = make_note(
        db_session,
        "acme",
        scope="repo_a",
        path_or_uri="docs/architecture/adr-001.md",
        name="ADR-001: Pelicanium",
        description="Why pelicanium and not albatrossium.",
        type_="adr",
        provenance_kind="user override",
        provenance_date=date(2026, 2, 3),
        status="approved",
    )
    passage = make_passage(
        db_session,
        note,
        ordinal=2,
        heading_path="Decision › Isotopes",
        content="The pelicanium isotope decays slowly.",
    )

    results = _results(_search(client, key, "isotope"))

    assert len(results) == 1
    result = results[0]
    assert result["passage_id"] == passage.id
    assert result["note_id"] == note.id
    assert result["ordinal"] == 2
    assert result["name"] == "ADR-001: Pelicanium"
    assert result["description"] == "Why pelicanium and not albatrossium."
    assert result["type"] == "adr"
    assert result["path_or_uri"] == "docs/architecture/adr-001.md"
    assert result["heading_path"] == "Decision › Isotopes"
    assert result["content"] == "The pelicanium isotope decays slowly."
    assert result["provenance_kind"] == "user override"
    assert result["provenance_date"] == "2026-02-03"
    assert result["status"] == "approved"
    assert result["score"] > 0
    # The only passage, matched by keyword and (as every passage is) by
    # meaning, so it tops both arms.
    assert (result["fts_rank"], result["vector_rank"]) == (1, 1)


def test_search_does_not_filter_or_rerank_by_status(client, db_session):
    # The user decision made explicit: status is metadata the caller may act
    # on, not a search-time filter or a ranking input. A superseded note's
    # passage appears, ranked exactly as its approved twin would be.
    key = make_api_key(db_session, "acme")
    superseded_note = make_note(
        db_session, "acme", path_or_uri="docs/old.md", status="superseded"
    )
    superseded = make_passage(
        db_session, superseded_note, content="The pelicanium isotope decays slowly."
    )
    approved_note = make_note(
        db_session, "acme", path_or_uri="docs/new.md", status="approved"
    )
    approved = make_passage(
        db_session, approved_note, content="The pelicanium isotope decays slowly."
    )

    results = _results(_search(client, key, "pelicanium"))

    by_id = {r["passage_id"]: r for r in results}
    assert set(by_id) == {superseded.id, approved.id}
    assert by_id[superseded.id]["status"] == "superseded"
    assert by_id[approved.id]["status"] == "approved"
    # Identical text, so identical keyword relevance and identical vectors:
    # both arms tie, and a tie in either arm breaks by passage id. The
    # superseded passage is seeded first, so it leads in both arms -- a
    # status-based demotion would be the only thing that could flip it.
    assert [r["passage_id"] for r in results] == [superseded.id, approved.id]
    assert [(r["fts_rank"], r["vector_rank"]) for r in results] == [(1, 1), (2, 2)]
