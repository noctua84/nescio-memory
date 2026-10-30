"""The CLI-argument validators in scripts/_cli_safety.py and their wiring.

No server, no database, no network. Every URL here is synthetic and every
file is written under tmp_path. The wiring tests install a client stand-in
that raises if it is ever constructed, so "the script rejected the URL" is
asserted as *no request was made*, not merely as an exit code.
"""
import httpx
import pytest

from scripts import eval_queries, reingest
from scripts._cli_safety import (
    ALLOWED_HOSTS_ENV_VAR,
    DEFAULT_ALLOWED_HOSTS,
    validate_base_url,
    validate_queries_path,
)


class _ExplodingClient:
    """An httpx.Client stand-in that fails the test if it is ever used."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("an HTTP client was constructed; no request was expected")


def _install_transport(monkeypatch, module, handler) -> list[httpx.Request]:
    """Route every request ``module`` makes through ``handler``."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording)
    real_client = httpx.Client  # bound before patching the shared module attr

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", factory)
    return seen


@pytest.fixture(autouse=True)
def _no_inherited_allowlist(monkeypatch):
    """The ambient environment must not widen the allowlist under test."""
    monkeypatch.delenv(ALLOWED_HOSTS_ENV_VAR, raising=False)


# --------------------------------------------------------------------------
# validate_base_url
# --------------------------------------------------------------------------


def test_the_default_allowlist_is_exactly_the_three_loopback_names():
    assert set(DEFAULT_ALLOWED_HOSTS) == {"localhost", "127.0.0.1", "::1"}


@pytest.mark.parametrize(
    "url, expected",
    [
        # The return value is rebuilt from the parsed parts, so a trailing
        # slash is normalised away rather than echoed back.
        ("http://localhost:8080", "http://localhost:8080"),
        ("http://localhost:8080/", "http://localhost:8080"),
        ("https://127.0.0.1", "https://127.0.0.1"),
        # An IPv6 literal has to come back bracketed, or the rebuilt string
        # is not a usable URL (the port would read as another hextet).
        ("http://[::1]:8080/", "http://[::1]:8080"),
        ("http://[::1]", "http://[::1]"),
    ],
)
def test_an_allowed_url_is_accepted_and_returned_rebuilt(url, expected):
    assert validate_base_url(url) == expected


def test_the_hostname_is_matched_case_insensitively():
    # Hostnames are case-insensitive by DNS, so rejecting this would be a
    # false positive rather than a closed hole.
    assert validate_base_url("http://LOCALHOST:8080") == "http://localhost:8080"


@pytest.mark.parametrize(
    "url",
    [
        # Plain off-allowlist hosts.
        "http://evil.example",
        "https://evil.example:8080/api",
        # Userinfo: the host is evil.example, "localhost" is only the user.
        "http://localhost@evil.example",
        "http://u:p@localhost",
        # The allowed name hidden in a fragment or a query string -- parsed
        # away by urlsplit, so a substring check on the raw URL would pass.
        "http://evil.example#@localhost",
        "http://evil.example?x=localhost",
        # Suffix confusion: an allowed label is not an allowed host.
        "http://localhost.evil.example",
        "http://notlocalhost",
        # Non-http schemes.
        "ftp://localhost",
        "file:///etc/passwd",
        "javascript:fetch('http://evil.example')",
        # No host at all.
        "http://",
        "/api/v2/search",
        "",
    ],
)
def test_a_url_that_is_not_a_plain_allowed_http_url_is_rejected(url):
    with pytest.raises(ValueError):
        validate_base_url(url)


def test_a_url_with_a_path_on_an_allowed_host_is_rejected():
    # A base URL carries no path; this used to be normalised (trailing
    # slash stripped, path kept) but is now rejected outright so the
    # returned string never carries any argv-derived path material.
    with pytest.raises(ValueError):
        validate_base_url("http://127.0.0.1:8080/prefix/")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080?x=1",
        "http://localhost:8080#frag",
    ],
)
def test_a_query_string_or_fragment_on_an_allowed_host_is_still_rejected(url):
    # Even pointed at an allowed host, these are not part of a base URL and
    # would be silently dropped or appended by the client.
    with pytest.raises(ValueError):
        validate_base_url(url)


def test_the_environment_variable_extends_the_allowlist(monkeypatch):
    monkeypatch.setenv(ALLOWED_HOSTS_ENV_VAR, " Memory.Internal , ,other.host")

    # Case-folded, whitespace-trimmed, empty entries ignored.
    assert validate_base_url("https://memory.internal") == "https://memory.internal"
    assert validate_base_url("https://Memory.Internal") == "https://memory.internal"
    assert validate_base_url("http://other.host:9000") == "http://other.host:9000"
    # Extending does not replace: the defaults still work ...
    assert validate_base_url("http://localhost:8080") == "http://localhost:8080"
    # ... and nothing else was opted in.
    with pytest.raises(ValueError):
        validate_base_url("https://evil.example")


def test_the_allowlist_is_read_per_call_not_at_import(monkeypatch):
    with pytest.raises(ValueError):
        validate_base_url("https://late.host")

    monkeypatch.setenv(ALLOWED_HOSTS_ENV_VAR, "late.host")

    assert validate_base_url("https://late.host") == "https://late.host"


# --------------------------------------------------------------------------
# validate_queries_path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["queries.yaml", "queries.yml", "queries.YAML"])
def test_an_existing_yaml_file_is_accepted_and_resolved(tmp_path, name):
    path = tmp_path / name
    path.write_text("version: 1\n", encoding="utf-8")

    resolved = validate_queries_path(str(tmp_path / "." / name))

    assert resolved == path.resolve()
    assert resolved.is_absolute()


def test_a_wrong_suffix_is_rejected_even_when_the_file_exists(tmp_path):
    path = tmp_path / "secrets.txt"
    path.write_text("not yaml\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_queries_path(str(path))


def test_a_missing_yaml_path_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        validate_queries_path(str(tmp_path / "absent.yaml"))


def test_a_directory_named_like_a_yaml_file_is_rejected(tmp_path):
    directory = tmp_path / "x.yaml"
    directory.mkdir()

    with pytest.raises(ValueError):
        validate_queries_path(str(directory))


# --------------------------------------------------------------------------
# Wiring into the two CLIs
# --------------------------------------------------------------------------


def _write_populated_queries(tmp_path, expect="synthetic-note.md"):
    """A YAML file whose gate is satisfied by its single query."""
    path = tmp_path / "queries.yaml"
    path.write_text(
        "version: 1\n"
        "gate:\n"
        "  top_k: 5\n"
        "  threshold: 1\n"
        "  minimum_queries: 1\n"
        "queries:\n"
        '  - query: "synthetic query"\n'
        f'    expect_path_or_uri: "{expect}"\n',
        encoding="utf-8",
    )
    return path


def test_eval_queries_rejects_a_disallowed_url_without_making_a_request(
    tmp_path, monkeypatch, capsys
):
    queries = _write_populated_queries(tmp_path)
    monkeypatch.setattr(eval_queries.httpx, "Client", _ExplodingClient)

    exit_code = eval_queries.main(
        [
            "--api-key",
            "nm_test_key",
            "--queries",
            str(queries),
            "--url",
            "http://evil.example",
        ]
    )

    assert exit_code != 0
    assert exit_code == eval_queries.EXIT_ERROR
    assert "evil.example" in capsys.readouterr().err


def test_reingest_rejects_a_disallowed_url_without_making_a_request(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "note.md").write_text("# note\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr(reingest.httpx, "Client", _ExplodingClient)

    exit_code = reingest.main(
        [
            "--root",
            str(tmp_path),
            "--api-key",
            "nm_test_key",
            "--url",
            "http://evil.example",
        ]
    )

    assert exit_code != 0
    assert exit_code == reingest.EXIT_USAGE_ERROR
    assert "evil.example" in capsys.readouterr().err


def test_load_queries_wraps_a_rejected_path_as_a_queries_file_error(tmp_path):
    path = tmp_path / "secrets.txt"
    path.write_text("not yaml\n", encoding="utf-8")

    # The validator raises ValueError; the script's own error type is what
    # main() catches, so the wrap has to happen inside load_queries.
    with pytest.raises(eval_queries.QueriesFileError):
        eval_queries.load_queries(str(path))


def test_eval_queries_runs_against_a_host_opted_in_by_the_environment(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv(ALLOWED_HOSTS_ENV_VAR, "memory.internal")
    queries = _write_populated_queries(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"path_or_uri": "synthetic-note.md"}]})

    seen = _install_transport(monkeypatch, eval_queries, handler)

    exit_code = eval_queries.main(
        [
            "--api-key",
            "nm_test_key",
            "--queries",
            str(queries),
            "--url",
            "https://memory.internal:9443",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == eval_queries.EXIT_PASS
    assert len(seen) == 1
    # The opted-in host is where the request actually went, and the key went
    # with it -- which is precisely why the allowlist has to gate it.
    assert seen[0].url.host == "memory.internal"
    assert seen[0].url.port == 9443
    assert seen[0].headers["X-API-Key"] == "nm_test_key"
    assert "GATE FAIL" not in stdout


def test_reingest_runs_against_a_host_opted_in_by_the_environment(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv(ALLOWED_HOSTS_ENV_VAR, "memory.internal")
    (tmp_path / "note.md").write_text("# note\n\nbody\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"skipped": False, "passages": 3})

    seen = _install_transport(monkeypatch, reingest, handler)

    exit_code = reingest.main(
        [
            "--root",
            str(tmp_path),
            "--api-key",
            "nm_test_key",
            "--url",
            "https://memory.internal:9443",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == reingest.EXIT_OK
    assert len(seen) == 1
    assert seen[0].url.host == "memory.internal"
    assert seen[0].url.port == 9443
    assert "note.md: ok" in stdout
