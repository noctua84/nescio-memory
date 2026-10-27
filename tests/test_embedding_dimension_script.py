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


# ---------------------------------------------------------------------------
# --degenerate: the #56 report, and the usage exit code beside it
# ---------------------------------------------------------------------------
#
# This mode exists to answer one question for a maintainer -- does this backend
# return a zero-norm vector for text a user could really send? -- and it has to
# answer it for a backend that is misbehaving, which is the only kind of
# backend worth asking. So the report's contract is that it reports: every
# input gets a line, including the ones whose value cannot be squared, summed
# or even tested for finiteness without raising. A traceback here is a failure
# of the feature, not a diagnostic.
#
# Staged at `script._fetch`, not at `embeddings_module._fetch` as everything
# above is. The script imports `_fetch` by name, so `_test_degenerate_inputs`
# calls the script module's own reference and patching the source module would
# leave the real backend to be contacted. The base dimension check is staged
# the usual way at the same time, because it runs first and the report is only
# reached once it returns OK -- so both seams are pinned in every test below
# and neither can reach the network.

# The inputs the report probes: blank, newlines, punctuation, emoji, single
# character (see the script's module docstring). Each is mapped to a staged
# response below; an input not named in a mapping is answered with a clean
# vector, so a sixth probe input would be reported as "accepted" rather than
# making these tests fail from the inside.
BLANK = " "
NEWLINES = "\n\n"
PUNCTUATION = "..."
EMOJI = "\N{SLIGHTLY SMILING FACE}"
SINGLE_CHAR = "a"

DEGENERATE_PROBE_INPUTS = [BLANK, NEWLINES, PUNCTUATION, EMOJI, SINGLE_CHAR]

# Subnormal in float4 (below FLOAT32_MIN_NORMAL, 1.17549435e-38) and a
# perfectly ordinary nonzero float64 -- the case the two norms must disagree
# about, which is the only reason the report prints two of them.
SUBNORMAL = 1e-40

# Valid JSON that is not a valid float: the stdlib json module decodes a huge
# integer literal into a Python int of arbitrary size, and math.isfinite()
# RAISES OverflowError on it rather than returning False. Anything that
# computed a norm before judging the components would die here.
UNBOUNDED_INT = 10**400


def _zero_vector() -> list[float]:
    return [0.0] * settings.embedding_dimension


def _all_subnormal_vector() -> list[float]:
    return [SUBNORMAL] * settings.embedding_dimension


def _vector_with(value: object, index: int = 0) -> list:
    vector: list = _clean_vector()
    vector[index] = value
    return vector


def _stage_degenerate(monkeypatch, responses: dict, base: list | None = None):
    """Stage the base probe and one response per degenerate input.

    `responses` maps a probe input to what the backend returns for it; a value
    that is an exception instance is raised instead of returned. Unmapped
    inputs get `_clean_vector()`.
    """
    _stage(monkeypatch, _clean_vector() if base is None else base)

    def _fetch(text):
        value = responses.get(text, _clean_vector())
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(script, "_fetch", _fetch)


def _report(out: str) -> dict[str, str]:
    """The report's lines, keyed by the repr of the input each is about.

    Parsed rather than searched as one string so each input's verdict is
    asserted on its own: `"rejected" in out` would be satisfied by any single
    line rejecting, and a report whose lines were attributed to the wrong
    inputs is exactly as wrong as no report at all.
    """
    lines = {}
    for line in out.splitlines():
        if line.startswith("  ") and " -> " in line:
            left, _, right = line.partition(" -> ")
            lines[left.strip()] = right
    return lines


def test_the_degenerate_report_gives_a_verdict_for_every_shape(monkeypatch, capsys):
    """One line per input, for the four shapes a backend really returns.

    All four on one run, because the report's value is comparative: an
    operator reads it to see which inputs their backend degenerates on, so a
    run that stopped at the first bad shape -- or that reported a norm for the
    zero vector and nothing for the malformed one -- answers a different
    question than the one asked.
    """
    _stage_degenerate(
        monkeypatch,
        {
            BLANK: _zero_vector(),
            NEWLINES: _all_subnormal_vector(),
            PUNCTUATION: _clean_vector(),
            EMOJI: _vector_with(UNBOUNDED_INT),
            SINGLE_CHAR: _vector_with(float("nan")),
        },
    )

    # The report is informational: it cannot change the verdict a pipeline
    # reads, and the base check passed here.
    assert script.main(["--degenerate"]) == script.EXIT_OK

    out = capsys.readouterr().out
    assert out.startswith("OK")
    assert "DEGENERATE INPUTS" in out
    assert script._describe_backend() in out

    report = _report(out)
    # Every probed input is accounted for, so an input that was skipped cannot
    # be mistaken for one that passed.
    for text in DEGENERATE_PROBE_INPUTS:
        assert repr(text) in report, out

    # The zero vector: both norms exactly zero, and _validated()'s refusal
    # stated -- which is the #56 question itself, since that refusal is what
    # becomes a 503 on a query.
    blank = report[repr(BLANK)]
    assert "norm_f64=0.000000e+00" in blank
    assert "norm_f32=0.000000e+00" in blank
    assert "rejected by _validated()" in blank

    # The all-subnormal vector, where the two norms must DISAGREE: its float64
    # norm is an ordinary nonzero number and its float32 norm is exactly zero.
    # A report that computed both in float64 would call this vector healthy,
    # and would satisfy every other assertion in this file.
    newlines = report[repr(NEWLINES)]
    assert "norm_f32=0.000000e+00" in newlines
    assert "norm_f64=0.000000e+00" not in newlines
    assert "rejected by _validated()" in newlines

    # The control, without which "rejected" above would be satisfied by a
    # report that rejected everything it was handed.
    punctuation = report[repr(PUNCTUATION)]
    assert "accepted" in punctuation
    assert "rejected" not in punctuation
    assert "norm_f64=0.000000e+00" not in punctuation
    assert "norm_f32=0.000000e+00" not in punctuation

    # The malformed shapes: a verdict line each, and NOT a norm -- printing a
    # norm would mean the component had been squared, which is the
    # OverflowError the shape check exists to come before.
    emoji = report[repr(EMOJI)]
    assert "too large to evaluate as a float" in emoji
    assert "norm_f64=" not in emoji

    single_char = report[repr(SINGLE_CHAR)]
    assert "expected finite float32 values" in single_char
    assert "norm_f64=" not in single_char


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(UNBOUNDED_INT, "too large to evaluate as a float", id="huge-int"),
        pytest.param(float("nan"), "expected finite float32 values", id="not-a-number"),
        pytest.param(float("inf"), "expected finite float32 values", id="infinity"),
        pytest.param("0.01", "expected numbers", id="string"),
        pytest.param(None, "expected numbers", id="null"),
        pytest.param(True, "expected numbers", id="bool"),
    ],
)
def test_a_malformed_component_is_a_verdict_line_and_not_a_traceback(
    monkeypatch, capsys, value, expected
):
    """Each malformed value named, because each would crash differently.

    The unbounded int raises OverflowError out of math.isfinite; the
    non-finite floats raise nothing there but make math.sqrt and
    _float32_sum_of_squares produce or reject garbage; the str and None cases
    raise TypeError on the first multiplication; and True is an int to
    isinstance but not a number any backend meant to send. All of them are
    things a backend returning an error payload shaped like a vector really
    emits, and all of them have to come back as a line about that input --
    this report is run against a broken backend by definition, so a traceback
    is the one outcome that makes it useless.
    """
    _stage_degenerate(monkeypatch, {BLANK: _vector_with(value)})

    assert script.main(["--degenerate"]) == script.EXIT_OK

    out = capsys.readouterr().out
    report = _report(out)
    assert expected in report[repr(BLANK)]
    # And the run continued: the remaining inputs are still reported, so one
    # bad shape does not truncate the report for the others.
    assert len(report) == len(DEGENERATE_PROBE_INPUTS), out


def test_a_non_list_response_for_one_input_is_reported_not_raised(monkeypatch, capsys):
    # _fetch does not validate, so "the backend returned a string" arrives
    # here as a string. Iterating it would yield characters and report a
    # misleading per-character fault instead of the real one.
    _stage_degenerate(monkeypatch, {BLANK: "notavector"})

    assert script.main(["--degenerate"]) == script.EXIT_OK

    report = _report(capsys.readouterr().out)
    assert "not a list of numbers" in report[repr(BLANK)]
    assert "accepted" in report[repr(PUNCTUATION)]


def test_a_backend_failure_on_one_input_does_not_end_the_report(monkeypatch, capsys):
    # A backend that refuses blank text specifically is itself an answer to
    # #56, and it is only an answer if the other inputs are still measured.
    _stage_degenerate(monkeypatch, {BLANK: EmbeddingBackendError("connection refused")})

    assert script.main(["--degenerate"]) == script.EXIT_OK

    report = _report(capsys.readouterr().out)
    assert "backend error: connection refused" in report[repr(BLANK)]
    assert len(report) == len(DEGENERATE_PROBE_INPUTS)


def test_the_report_never_prints_the_vector_itself(monkeypatch, capsys):
    # Stated in _test_degenerate_inputs' docstring, and worth pinning: a
    # 1024-component dump would bury the five lines the mode exists to print,
    # and an embedding is derived from the input text. The staged component is
    # chosen to be unmistakable in the output if it ever leaked.
    _stage_degenerate(
        monkeypatch, {BLANK: [0.123456789] * settings.embedding_dimension}
    )

    assert script.main(["--degenerate"]) == script.EXIT_OK

    out = capsys.readouterr().out
    assert "0.123456789" not in out
    # One line per input, and nothing per component.
    assert len(_report(out)) == len(DEGENERATE_PROBE_INPUTS)


def test_the_degenerate_report_does_not_run_without_the_flag(monkeypatch, capsys):
    """The default path is unchanged -- asserted by the call not happening.

    Not by the absence of the header: a report that printed nothing would also
    produce no header, and would still have sent five requests to the backend.
    What must not happen on an ordinary run of this script is the embedding of
    five extra inputs.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        script, "_test_degenerate_inputs", lambda: calls.append("called")
    )
    _stage(monkeypatch, _clean_vector())

    assert script.main() == script.EXIT_OK
    assert calls == []

    out = capsys.readouterr().out
    assert out.startswith("OK")
    assert "DEGENERATE INPUTS" not in out

    # The same staging, with the flag: so "not called" above is a fact about
    # the flag and not about this test's monkeypatching.
    assert script.main(["--degenerate"]) == script.EXIT_OK
    assert calls == ["called"]


def test_the_degenerate_flag_does_not_mask_a_failing_base_check(monkeypatch, capsys):
    # Documented in the script's module docstring: with --degenerate the exit
    # code is still the base check's. The report is informational, and a mode
    # that swallowed a MISMATCH into a 0 would turn a diagnostic flag into a
    # way of making a pipeline pass.
    _stage_degenerate(monkeypatch, {}, base=_clean_vector(OFF_WIDTH))

    assert script.main(["--degenerate"]) == script.EXIT_MISMATCH

    out = capsys.readouterr().out
    assert out.startswith("MISMATCH")
    # And the report did not run: measuring degenerate norms against a backend
    # of the wrong width would be measuring the wrong model.
    assert "DEGENERATE INPUTS" not in out


def test_an_unknown_flag_exits_usage_and_not_unreachable(capsys):
    """64, not argparse's default 2 -- which is EXIT_UNREACHABLE here.

    The collision is the whole point: a pipeline reading the exit code alone
    could not tell "the deploy script passed a flag this version does not
    have" from "the embedding backend is down", and those have opposite
    responses. Asserted as a raised SystemExit because argparse exits rather
    than returning, so a caller of main() sees this as an exception and not as
    a return value.
    """
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--no-such-flag"])

    assert excinfo.value.code == script.EXIT_USAGE
    assert excinfo.value.code == 64
    assert excinfo.value.code != script.EXIT_UNREACHABLE
    assert excinfo.value.code != script.EXIT_OK

    # argparse's own diagnostics are kept: the usage line and the message
    # still go to stderr, and only the code changed.
    err = capsys.readouterr().err
    assert "usage:" in err
    assert "--no-such-flag" in err


def test_the_usage_code_is_distinct_from_every_verdict_code():
    # The same argument as test_the_four_exit_codes_are_distinct, extended to
    # the fifth code. Each test above asserts one code in isolation, so a
    # constant collapsed onto another would leave them all green.
    codes = [
        script.EXIT_OK,
        script.EXIT_MISMATCH,
        script.EXIT_UNREACHABLE,
        script.EXIT_UNUSABLE,
        script.EXIT_USAGE,
    ]
    assert len(set(codes)) == len(codes)
