"""The `--degenerate` report of `scripts/check_embedding_dimension.py`.

Covers the #56 report: that every probed input gets a line (including the
ones whose value cannot be squared, summed or even tested for finiteness
without raising), that a malformed or failing response is a verdict line and
not a traceback, that the vector itself is never printed, and that the flag
neither runs the report by default nor masks a failing base check. The base
verdicts and exit codes are in `test_embedding_dimension_script_verdicts.py`;
the staging both share is in `tests/_embedding_dimension_helpers.py`.
"""

import pytest

from app.config import settings
from app.core.errors import EmbeddingBackendError
from scripts import check_embedding_dimension as script
from tests._embedding_dimension_helpers import OFF_WIDTH, _clean_vector, _stage

# ---------------------------------------------------------------------------
# --degenerate: the #56 report
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
