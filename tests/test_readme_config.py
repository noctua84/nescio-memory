"""The README configuration table must agree with the Settings defaults.

A default that is changed in app/config.py but not in the table sends operators
tuning against a number the service no longer uses; this pins the pair.
"""
import re
from pathlib import Path

from app.config import Settings

README = Path(__file__).resolve().parent.parent / "README.md"


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
