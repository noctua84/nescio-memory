"""The operator script's verdicts and its exit codes.

`scripts/check_embedding_dimension.py` exists because nothing else in the
project compares EMBEDDING_DIMENSION against what the configured backend really
emits, and its *exit code* is the part a deployment pipeline reads. Nothing
asserted that code anywhere, so a reordered branch or a collapsed constant
would have changed what a pipeline concluded while the whole suite stayed
green -- which is the same shape of hole as issue #36 itself, one level up.

The backend is staged at `app.core.embeddings._fetch` rather than at the
script's own `probe_embedding` reference. Patching the latter would assert the
script's branching against a hand-made EmbeddingProbe and would keep passing if
probe_embedding() stopped detecting anything, so the two halves of the feature
would each be covered and their junction not at all. Patching _fetch runs the
real probe over a staged backend response, so these tests fail if either half
breaks.

No database and no container: the script touches neither.

Non-finite components are built here as ordinary Python floats, because _fetch
is the seam and no JSON is decoded on this path. That the bare `NaN` /
`Infinity` literals really arrive off the wire as these values is pinned
separately, in tests/test_embeddings.py.
"""
import pytest

from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from scripts import check_embedding_dimension as script

# A width that is unmistakably not the configured one. Derived from the setting
# rather than written as a literal so it stays wrong for any configured width.
OFF_WIDTH = settings.embedding_dimension + 7

# The MISMATCH branch's remediation, quoted so the unusable cases can assert it
# is *absent*. Acting on it is actively harmful when the values cannot be
# stored at any width: it migrates the column for nothing and fixes nothing.
MISMATCH_REMEDIATION = "Set EMBEDDING_DIMENSION"


def _clean_vector(width: int | None = None) -> list[float]:
    """A vector of components pgvector's float4 column holds without complaint."""
    return [0.01] * (settings.embedding_dimension if width is None else width)


def _vector_with_a_non_finite_component(
    width: int | None = None, index: int = 0
) -> list[float]:
    """Right type, right shape, and still unstorable.

    Every component is a genuine float and there are exactly `width` of them,
    so only the float4 usability rule can object -- which is the point: a NaN
    is not a malformed response in any way a shape check could see.
    """
    vector = _clean_vector(width)
    vector[index] = float("nan")
    return vector


def _stage(monkeypatch, vector):
    monkeypatch.setattr(embeddings_module, "_fetch", lambda text: vector)


def _stage_failure(monkeypatch, exception):
    def _raiser(text):
        raise exception

    monkeypatch.setattr(embeddings_module, "_fetch", _raiser)


def test_the_four_exit_codes_are_distinct():
    # The codes are the script's entire machine-readable output, so two of them
    # colliding would silently merge two verdicts a pipeline is meant to treat
    # differently -- and every other test here would still pass, because each
    # asserts one code in isolation.
    codes = [
        script.EXIT_OK,
        script.EXIT_MISMATCH,
        script.EXIT_UNREACHABLE,
        script.EXIT_UNUSABLE,
    ]
    assert len(set(codes)) == len(codes)
    # EXIT_OK must be the success code specifically: a shell reads 0 and
    # nothing else as "passed".
    assert script.EXIT_OK == 0


def test_a_usable_vector_of_the_configured_width_exits_ok(monkeypatch, capsys):
    _stage(monkeypatch, _clean_vector())

    assert script.main() == script.EXIT_OK

    out = capsys.readouterr().out
    assert out.startswith("OK")
    # The verdict line names which backend was checked, so an operator reading
    # a pipeline log can tell whether it checked the one they changed.
    assert script._describe_backend() in out
    assert str(settings.embedding_dimension) in out


def test_a_usable_vector_of_the_wrong_width_exits_mismatch(monkeypatch, capsys):
    _stage(monkeypatch, _clean_vector(OFF_WIDTH))

    assert script.main() == script.EXIT_MISMATCH

    out = capsys.readouterr().out
    assert out.startswith("MISMATCH")
    # Both numbers, because "wrong width" is useless without them.
    assert str(OFF_WIDTH) in out
    assert str(settings.embedding_dimension) in out
    # And the remediation, which is correct in this case and only this case.
    assert MISMATCH_REMEDIATION in out


@pytest.mark.parametrize(
    "exception",
    [
        EmbeddingBackendError("connection refused"),
        EmbeddingBackendMisconfigured("OLLAMA_URL is not a valid URL"),
        EmbeddingBackendBadResponse("Ollama response was not a usable embedding"),
    ],
    ids=["unreachable", "misconfigured", "unreadable"],
)
def test_every_backend_failure_is_could_not_check_rather_than_a_verdict(
    monkeypatch, capsys, exception
):
    """The conflation the script makes on purpose, pinned because it is a choice.

    A misconfiguration is a 500 in the API and a transient outage is a 503, and
    this script collapses both -- plus an unreadable response -- into one code.
    That is deliberate: all three mean "could not check", and the distinction a
    pipeline needs is between that and "checked, and it is wrong". A pipeline
    may reasonably tolerate the former and never the latter.

    What must not collapse is the *diagnostic*: the class name is printed so a
    human reading the log can still tell a restarting Ollama from a deployment
    mistake that will never recover.
    """
    _stage_failure(monkeypatch, exception)

    assert script.main() == script.EXIT_UNREACHABLE

    out = capsys.readouterr().out
    assert out.startswith("COULD NOT CHECK")
    assert type(exception).__name__ in out
    assert str(exception) in out
    # It did not check, so it must not pronounce on the width either way.
    assert "MISMATCH" not in out
    assert "UNUSABLE VALUES" not in out
    assert MISMATCH_REMEDIATION not in out


def test_a_non_list_response_is_could_not_check_rather_than_a_crash(
    monkeypatch, capsys
):
    # The one fault probe_embedding() still raises for -- there is no width to
    # measure -- reached through the real probe rather than by staging the
    # exception. Without the script's `except EmbeddingBackendError` covering
    # the subclass, this would exit with a traceback and whatever code Python
    # chose, which no pipeline can interpret.
    _stage(monkeypatch, "notavector")

    assert script.main() == script.EXIT_UNREACHABLE

    out = capsys.readouterr().out
    assert out.startswith("COULD NOT CHECK")
    assert "expected a list" in out


def test_a_correct_width_vector_of_unstorable_values_exits_unusable(
    monkeypatch, capsys
):
    """The verdict that did not exist, and the reason it had to.

    This is the issue-#36 shape the script exists to prevent, one level deeper:
    while the probe measured without checking, a backend emitting
    EMBEDDING_DIMENSION NaNs was reported as `OK` and exited 0 -- a clean bill
    of health for a backend against which every ingest and every search fails
    with a 503.

    The negative assertions carry as much weight as the exit code. The width
    *is* correct here, so a script that only compared widths would say OK; and
    telling the operator to change EMBEDDING_DIMENSION would be wrong advice in
    this state, which is precisely the mistake this branch exists to avoid.
    """
    _stage(monkeypatch, _vector_with_a_non_finite_component())

    assert script.main() == script.EXIT_UNUSABLE

    out = capsys.readouterr().out
    assert out.startswith("UNUSABLE VALUES")
    assert script._describe_backend() in out
    # Which component, so the value can be found again in the backend's own
    # output rather than merely asserted to exist.
    assert "index 0:" in out
    assert "OK" not in out.splitlines()[0]
    assert "MISMATCH" not in out
    assert MISMATCH_REMEDIATION not in out


@pytest.mark.parametrize("index", [0, settings.embedding_dimension - 1])
def test_the_reported_index_is_the_component_that_is_actually_bad(
    monkeypatch, capsys, index
):
    # An off-by-one here sends an operator to inspect the wrong component of a
    # 1024-wide vector, which is worse than printing no index at all. The
    # last-index case also guards a probe loop that stopped early: it would
    # report nothing, and the script would print OK.
    _stage(monkeypatch, _vector_with_a_non_finite_component(index=index))

    assert script.main() == script.EXIT_UNUSABLE

    out = capsys.readouterr().out
    assert f"index {index}:" in out
    assert "unusable components  1 of" in out


def test_unusable_values_are_reported_ahead_of_a_wrong_width(monkeypatch, capsys):
    """The ordering decision, which is the single load-bearing case in this file.

    A backend emitting garbage is usually the wrong model, so it is routinely
    also the wrong width -- both faults at once, with opposite remedies. The
    script reports the unusable values first and exits 3, because MISMATCH's
    remedy is the harmful one to act on: setting EMBEDDING_DIMENSION to this
    width and migrating the learnings.embedding column adopts the shape of a
    backend whose output cannot be stored at any width, so it migrates the
    column for nothing and fixes nothing.

    A future reorder that let the width check win would be invisible without
    this test, and would hand an operator a schema migration as the first thing
    to do about a dead model.
    """
    _stage(monkeypatch, _vector_with_a_non_finite_component(OFF_WIDTH))

    code = script.main()
    assert code == script.EXIT_UNUSABLE
    # Stated as well, because the exit code is the whole decision here and
    # `== EXIT_UNUSABLE` would also hold if the two constants ever collided.
    assert code != script.EXIT_MISMATCH

    out = capsys.readouterr().out
    assert out.startswith("UNUSABLE VALUES")
    # The width is still reported -- withholding it would make the compound
    # case less diagnosable than either fault alone --
    assert str(OFF_WIDTH) in out
    assert str(settings.embedding_dimension) in out
    # -- and said to be wrong, so the operator is not left to discover it on a
    # later run and wonder why it went unmentioned.
    assert "width is wrong too" in out
    # But without the MISMATCH verdict or its remediation, which is the whole
    # point of the ordering.
    assert "MISMATCH" not in out
    assert MISMATCH_REMEDIATION not in out


def test_the_fault_listing_is_capped_so_the_remediation_survives(monkeypatch, capsys):
    """A thousand fault lines would scroll the only useful sentence off the top.

    The realistic shape of this failure is not one bad component but all of
    them -- a backend that is not an embedding model, or one returning an error
    payload shaped like a vector. Printing EMBEDDING_DIMENSION near-identical
    lines would bury the remediation above the operator's scrollback, so the
    listing is capped and the count is summarised instead.

    Asserted by position, not merely by presence: the remediation has to come
    *after* the listing for the cap to have achieved anything.
    """
    width = settings.embedding_dimension
    _stage(monkeypatch, [float("nan")] * width)

    assert script.main() == script.EXIT_UNUSABLE

    out = capsys.readouterr().out
    listed = [line for line in out.splitlines() if line.strip().startswith("index ")]
    assert len(listed) == script.MAX_REPORTED_COMPONENTS
    # The total is still stated, so capping the listing does not hide the scale
    # of the fault -- "3 unusable" and "all 1024 unusable" are different
    # diagnoses and only the count distinguishes them.
    assert f"unusable components  {width} of {width}" in out
    assert f"... and {width - script.MAX_REPORTED_COMPONENTS} more" in out

    remediation = "fix or replace the model"
    assert remediation in out
    assert out.index(remediation) > out.index(listed[-1])


def test_a_usable_vector_is_not_reported_as_unusable(monkeypatch, capsys):
    # The control for the whole file. Every assertion above is about a fault
    # being found, so a usability rule that objected to ordinary components --
    # an inverted comparison, say -- would leave them all green while the
    # script exited 3 for every healthy backend forever, and a deployment
    # pipeline wired to it would never pass again.
    _stage(monkeypatch, _clean_vector())

    assert script.main() == script.EXIT_OK

    out = capsys.readouterr().out
    assert "UNUSABLE VALUES" not in out
    assert "unusable components" not in out
