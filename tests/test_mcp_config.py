"""McpSettings and the parts of nescio_mcp that do not need the `mcp` extra.

No server, database or network. Nothing here imports the SDK, so these run with
or without the extra.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from nescio_mcp.config import McpSettings, configuration_errors

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
KEY = "nm_test_s3cret_mcp_key"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """The ambient environment must not satisfy or widen anything under test."""
    for name in list(os.environ):
        if name.upper().startswith("NESCIO_"):
            monkeypatch.delenv(name)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("NESCIO_URL", "http://localhost:8080")
    monkeypatch.setenv("NESCIO_API_KEY", KEY)


def _refusal(monkeypatch, **env) -> str:
    """Build the settings from env, which must fail; return the error text."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError) as excinfo:
        McpSettings()
    return str(excinfo.value)


# --- defaults and the happy path ---------------------------------------------


def test_the_defaults_are_the_documented_ones(configured):
    settings = McpSettings()

    assert settings.timeout_s == 120
    assert settings.max_retry_after_s == 30


def test_the_url_is_rebuilt_as_a_bare_base_url(configured, monkeypatch):
    monkeypatch.setenv("NESCIO_URL", "http://LOCALHOST:9000/")

    assert McpSettings().url == "http://localhost:9000"


def test_the_http_client_carries_the_settings(configured, monkeypatch):
    monkeypatch.setenv("NESCIO_TIMEOUT_S", "45")

    with McpSettings().build_http_client() as http:
        assert str(http.base_url) == "http://localhost:8080"
        assert http.headers["X-API-Key"] == KEY
        assert http.timeout == httpx.Timeout(45.0)
        # A redirect must not be able to carry the key off the allowlist.
        assert http.follow_redirects is False


# --- NESCIO_URL ----------------------------------------------------------------


def test_an_off_allowlist_host_is_refused(monkeypatch):
    message = _refusal(
        monkeypatch, NESCIO_URL="http://evil.example:8080", NESCIO_API_KEY=KEY
    )

    assert "NESCIO_URL" in message
    assert "evil.example" in message
    assert "--url" not in message


def test_nescio_allowed_hosts_opens_the_allowlist(monkeypatch, configured):
    monkeypatch.setenv("NESCIO_URL", "https://memory.example.org")
    monkeypatch.setenv("NESCIO_ALLOWED_HOSTS", "memory.example.org")

    assert McpSettings().url == "https://memory.example.org"


@pytest.mark.parametrize(
    "url",
    [
        "not a url",
        "ftp://localhost",
        "file:///etc/passwd",
        "http://",
        "http://[::1",
        "http://localhost:notaport",
        "http://localhost/api/v1",
        "http://localhost?x=1",
        "http://localhost#frag",
        "http://user:pass@localhost",
    ],
)
def test_a_malformed_url_is_refused(monkeypatch, url):
    message = _refusal(monkeypatch, NESCIO_URL=url, NESCIO_API_KEY=KEY)

    assert "NESCIO_URL" in message


def test_a_refused_url_is_not_echoed(monkeypatch):
    # It may hold credentials of its own, and this text goes to a log.
    message = _refusal(
        monkeypatch, NESCIO_URL="http://user:hunter2@localhost", NESCIO_API_KEY=KEY
    )

    assert "hunter2" not in message


def test_a_missing_url_is_refused(monkeypatch):
    message = _refusal(monkeypatch, NESCIO_API_KEY=KEY)

    assert "url" in message


# --- NESCIO_API_KEY ------------------------------------------------------------


def test_a_missing_key_is_refused(monkeypatch):
    message = _refusal(monkeypatch, NESCIO_URL="http://localhost:8080")

    assert "api_key" in message


@pytest.mark.parametrize("key", ["", "   "])
def test_an_empty_key_is_refused(monkeypatch, key):
    message = _refusal(monkeypatch, NESCIO_URL="http://localhost:8080", NESCIO_API_KEY=key)

    assert "NESCIO_API_KEY must not be empty" in message


@pytest.mark.parametrize(
    "key", ["nm_with space", "nm_tab\tkey", "nm_newline\nkey", "nm_ümlaut"]
)
def test_a_key_that_cannot_be_a_header_is_refused_without_being_quoted(monkeypatch, key):
    message = _refusal(monkeypatch, NESCIO_URL="http://localhost:8080", NESCIO_API_KEY=key)

    assert "NESCIO_API_KEY" in message
    assert "nm_" not in message


def test_the_key_is_not_in_the_settings_repr(configured):
    assert KEY not in repr(McpSettings())


@pytest.mark.parametrize(
    "env",
    [
        {"NESCIO_URL": "http://evil.example"},
        {"NESCIO_TIMEOUT_S": "0"},
        {"NESCIO_TIMEOUT_S": "not a number"},
        {"NESCIO_MAX_RETRY_AFTER_S": "61"},
    ],
)
def test_the_key_is_not_in_a_validation_error(monkeypatch, env):
    # Without hide_input_in_errors a model validator's error prints every input,
    # and the key is one of them. Each case fails on a field other than the key.
    settings = {"NESCIO_URL": "http://localhost:8080", "NESCIO_API_KEY": KEY, **env}
    for name, value in settings.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as excinfo:
        McpSettings()

    assert KEY not in str(excinfo.value)
    assert KEY not in repr(excinfo.value)
    assert KEY not in " ".join(configuration_errors(excinfo.value))


# --- the bounds ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NESCIO_TIMEOUT_S", "0"),
        ("NESCIO_TIMEOUT_S", "0.5"),
        ("NESCIO_TIMEOUT_S", "601"),
        ("NESCIO_TIMEOUT_S", "nan"),
        ("NESCIO_TIMEOUT_S", "inf"),
        ("NESCIO_MAX_RETRY_AFTER_S", "-1"),
        ("NESCIO_MAX_RETRY_AFTER_S", "61"),
        ("NESCIO_MAX_RETRY_AFTER_S", "nan"),
        ("NESCIO_MAX_RETRY_AFTER_S", "inf"),
    ],
)
def test_an_out_of_bounds_value_is_refused_by_name(configured, monkeypatch, name, value):
    assert name in _refusal(monkeypatch, **{name: value})


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("NESCIO_TIMEOUT_S", "timeout_s", 1.0),
        ("NESCIO_TIMEOUT_S", "timeout_s", 600.0),
        ("NESCIO_MAX_RETRY_AFTER_S", "max_retry_after_s", 0.0),
        ("NESCIO_MAX_RETRY_AFTER_S", "max_retry_after_s", 60.0),
    ],
)
def test_the_edges_of_a_bound_are_accepted(configured, monkeypatch, name, field, value):
    monkeypatch.setenv(name, str(value))

    actual = getattr(McpSettings(), field)

    assert math.isclose(actual, value)


def test_a_dotenv_file_is_not_read(monkeypatch, tmp_path):
    # NESCIO_ALLOWED_HOSTS comes from os.environ alone, so a .env that could set
    # the URL but not the allowlist would be worse than one that does nothing.
    (tmp_path / ".env").write_text(
        f"NESCIO_URL=http://localhost:8080\nNESCIO_API_KEY={KEY}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError):
        McpSettings()


# --- the README ------------------------------------------------------------------


def _readme_mcp_rows() -> dict[str, str]:
    """First cell to second cell of each table row in the MCP server section."""
    text = README.read_text(encoding="utf-8")
    section = text.split("\n## MCP server", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 3:
            rows[cells[1].strip("`")] = cells[2].strip("`")
    return rows


def test_every_setting_has_a_readme_row():
    rows = _readme_mcp_rows()

    missing = [
        f"NESCIO_{name.upper()}"
        for name in McpSettings.model_fields
        if f"NESCIO_{name.upper()}" not in rows
    ]

    assert not missing, f"settings without a README MCP server row: {missing}"
    assert "NESCIO_ALLOWED_HOSTS" in rows


@pytest.mark.parametrize("field", ["timeout_s", "max_retry_after_s"])
def test_the_readme_defaults_match_the_settings(field):
    documented = _readme_mcp_rows()[f"NESCIO_{field.upper()}"]

    assert float(documented) == McpSettings.model_fields[field].default


# --- what the package must not pull in --------------------------------------------


def _clean_subprocess_env() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name != "DATABASE_URL" and not name.upper().startswith("NESCIO_")
    }


def test_the_settings_and_client_import_neither_the_app_nor_the_sdk():
    # In a clean interpreter, with no DATABASE_URL: importing app.config would
    # raise, and importing the SDK would break running without the extra.
    code = (
        "import sys, nescio_mcp, nescio_mcp.config, nescio_mcp.api_client\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('app', 'mcp')]\n"
        "sys.exit(repr(bad) if bad else 0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_running_without_the_extra_says_how_to_install_it():
    # mcp set to None in sys.modules makes `import mcp` raise ModuleNotFoundError,
    # which is what an environment without the extra does.
    code = (
        "import runpy, sys\n"
        "sys.modules['mcp'] = None\n"
        "runpy.run_module('nescio_mcp', run_name='__main__')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "--extra mcp" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_readme_registers_the_server_with_one_command():
    text = README.read_text(encoding="utf-8")
    section = text.split("\n## MCP server", 1)[1].split("\n## ", 1)[0]

    # Options before the name, the command after `--`, and an option between the
    # last --env and the name, because --env takes several values.
    assert re.search(
        r"claude mcp add\b.*--env NESCIO_URL=.*--env NESCIO_API_KEY=.*"
        r"--transport stdio nescio-memory -- uv run --directory \S+ "
        r"--extra mcp python -m nescio_mcp",
        section,
        flags=re.DOTALL,
    )
