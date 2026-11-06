"""Settings for the MCP server, all from the process environment.

Not in app/config.py: that module requires DATABASE_URL and carries the API's
settings, and this process has no business with either. Same house style though:
every bound is a model validator whose error names the env var, and a bad value
stops the process at startup rather than failing per tool call.

There is deliberately no env_file. NESCIO_ALLOWED_HOSTS is read from os.environ
by scripts/_cli_safety.py, so a .env could set the URL but not widen the
allowlist, and this process would pick up a file sitting in whatever checkout it
was launched from.
"""
from __future__ import annotations

import string

import httpx
from pydantic import SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nescio_mcp.api_client import API_KEY_HEADER, DEFAULT_MAX_RETRY_AFTER_S
from scripts._cli_safety import validate_base_url

_KEY_CHARACTERS = frozenset(string.ascii_letters + string.digits + string.punctuation)


class McpSettings(BaseSettings):
    # NESCIO_URL: the service's base URL. Host on the allowlist, see below.
    url: str
    # NESCIO_API_KEY: this client's own key, the only authority the server has.
    api_key: SecretStr
    # NESCIO_TIMEOUT_S: per request. Generous because an ingest of a large note
    # embeds many sub-batches before it answers.
    timeout_s: float = 120.0
    # NESCIO_MAX_RETRY_AFTER_S: caps the one wait before a retry, so a bad
    # Retry-After degrades to a slow retry instead of a hung tool call.
    max_retry_after_s: float = DEFAULT_MAX_RETRY_AFTER_S

    model_config = SettingsConfigDict(
        env_prefix="NESCIO_",
        extra="ignore",
        # A model validator's error otherwise prints the whole input, which here
        # means the plaintext key: SecretStr only wraps it after validation.
        hide_input_in_errors=True,
    )

    # The key is sent to this host on every request, so it has to be one the
    # operator named. The rules are the CLI scripts'; only the wording changes.
    # The URL is not echoed, as it may carry credentials or a query string.
    @model_validator(mode="after")
    def _url_must_be_an_allowed_base_url(self) -> McpSettings:
        try:
            self.url = validate_base_url(self.url)
        except ValueError as exc:
            reason = str(exc).replace(f"invalid --url {self.url!r}: ", "")
            raise ValueError(f"NESCIO_URL is not usable: {reason}") from None
        return self

    # httpx sends the key as a header, and h11 rejects a malformed header value
    # with an exception that quotes it. Refusing that key here means no request,
    # and so no such exception, can ever exist.
    @model_validator(mode="after")
    def _api_key_must_be_a_sendable_header(self) -> McpSettings:
        key = self.api_key.get_secret_value()
        if not key.strip():
            raise ValueError("NESCIO_API_KEY must not be empty")
        if not _KEY_CHARACTERS.issuperset(key):
            raise ValueError(
                "NESCIO_API_KEY must be printable ASCII with no whitespace"
            )
        return self

    # Written as `not low <= value <= high` so that NaN, which compares false
    # against everything, is refused as well.
    @model_validator(mode="after")
    def _timeout_must_be_bounded(self) -> McpSettings:
        if not 1 <= self.timeout_s <= 600:
            raise ValueError(f"NESCIO_TIMEOUT_S must be 1..600, got {self.timeout_s}")
        return self

    @model_validator(mode="after")
    def _max_retry_after_must_be_bounded(self) -> McpSettings:
        if not 0 <= self.max_retry_after_s <= 60:
            raise ValueError(
                f"NESCIO_MAX_RETRY_AFTER_S must be 0..60, got {self.max_retry_after_s}"
            )
        return self

    def build_http_client(self) -> httpx.Client:
        # follow_redirects stays off: a redirect could carry the key to a host
        # the allowlist never saw. It surfaces as an error instead.
        return httpx.Client(
            base_url=self.url,
            timeout=self.timeout_s,
            headers={API_KEY_HEADER: self.api_key.get_secret_value()},
            follow_redirects=False,
        )


def configuration_errors(exc: ValidationError) -> list[str]:
    """One line per problem, naming the env var and holding no input value."""
    lines = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        if error["loc"]:
            lines.append(f"NESCIO_{str(error['loc'][0]).upper()}: {error['msg']}")
        else:
            lines.append(error["msg"].removeprefix("Value error, "))
    return lines
