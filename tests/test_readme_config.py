"""The README configuration table must agree with the Settings defaults.

A default that is changed in app/config.py but not in the table sends operators
tuning against a number the service no longer uses; this pins the pair.
"""
import re
from pathlib import Path

from app.config import Settings

README = Path(__file__).resolve().parent.parent / "README.md"
ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def _readme_default(env_var: str) -> str:
    """The backticked default in the README config row for `env_var`."""
    for line in README.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 3 and cells[1] == f"`{env_var}`":
            match = re.fullmatch(r"`([^`]*)`", cells[2])
            assert match, f"{env_var} default cell is not a single code span: {cells[2]!r}"
            return match.group(1)
    raise AssertionError(f"no README config row for {env_var}")


def test_query_embed_timeout_default_covers_a_cold_model_load():
    # 5 s was too short for Ollama reloading an unloaded model.
    assert Settings.model_fields["query_embed_timeout_s"].default == 15.0


def test_readme_query_embed_timeout_matches_the_settings_default():
    documented = float(_readme_default("QUERY_EMBED_TIMEOUT_S"))
    assert documented == Settings.model_fields["query_embed_timeout_s"].default


def _readme_config_variables() -> set[str]:
    """Every env var named in the first cell of the README Configuration table.

    Combined cells such as `HOST` / `PORT` name several variables.
    """
    text = README.read_text(encoding="utf-8")
    section = text.split("\n## Configuration", 1)[1].split("\n## ", 1)[0]
    names: set[str] = set()
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 3:
            names.update(re.findall(r"`([A-Z][A-Z0-9_]*)`", cells[1]))
    return names


def _env_example_variables() -> set[str]:
    """Every NAME= or # NAME= line in .env.example."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE))


def test_every_setting_has_a_readme_row():
    documented = _readme_config_variables()
    missing = [
        name.upper() for name in Settings.model_fields if name.upper() not in documented
    ]
    assert not missing, f"settings without a README Configuration row: {missing}"


def test_every_setting_has_an_env_example_line():
    present = _env_example_variables()
    missing = [
        name.upper() for name in Settings.model_fields if name.upper() not in present
    ]
    assert not missing, f"settings without a NAME= line in .env.example: {missing}"
