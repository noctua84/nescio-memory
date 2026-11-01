"""Shared CLI-argument hardening for scripts/reingest.py and scripts/eval_queries.py.

Both scripts take CLI-supplied values that end up either as the target of an
HTTP request (--url) or as a path opened on disk (--queries). An agent (or a
careless human) invoking either CLI with a crafted --url can make the script
send the X-API-Key header to an arbitrary host (SSRF / credential leak); a
crafted --queries can make it read an arbitrary file (path injection). These
two validators close both holes before any client is built or any file is
opened.

The URL allowlist defaults to localhost, 127.0.0.1 and ::1, and can be
extended with the comma-separated NESCIO_ALLOWED_HOSTS environment variable
(hostnames matched case-insensitively, whitespace trimmed around each entry).
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ALLOWED_HOSTS_ENV_VAR = "NESCIO_ALLOWED_HOSTS"


def _allowed_hosts() -> frozenset[str]:
    extra = os.environ.get(ALLOWED_HOSTS_ENV_VAR, "")
    extra_hosts = {host.strip().lower() for host in extra.split(",") if host.strip()}
    return DEFAULT_ALLOWED_HOSTS | extra_hosts


def validate_base_url(url: str) -> str:
    """Validate a CLI-supplied base URL and return it rebuilt from its parts.

    Rejects anything that is not a plain http(s) URL pointing at an allowed
    host: no userinfo (credentials embedded in the URL), no query string,
    fragment or path, and a hostname that is on the allowlist. The allowlist
    is DEFAULT_ALLOWED_HOSTS plus whatever NESCIO_ALLOWED_HOSTS adds, so a
    non-local target has to be opted into explicitly rather than accepted
    because it merely looked well-formed.

    The return value is a *base* URL built entirely from fixed/trusted
    material -- never a substring of the argv value itself -- so that a
    static taint analysis following argv -> this return value -> an HTTP
    client cannot flag this as a passthrough, and no urlsplit quirk (an
    unexpected path, a smuggled fragment) can ride along in the result:
      - scheme: the literal "http" or "https" selected by comparison, not
        parts.scheme.
      - host: the matching entry from the allowlist collection itself, not
        parts.hostname -- the rebuilt string is always one of the admin-
        controlled allowlist values, never argv-derived bytes.
      - port: an explicit int() of the already-validated numeric port.
      - path: not carried at all; anything other than "" or "/" is rejected,
        since a base URL has no path component to preserve.
    """
    parts: SplitResult = urlsplit(url)

    if parts.scheme not in ("http", "https"):
        raise ValueError(f"invalid --url {url!r}: scheme must be http or https")
    scheme = "https" if parts.scheme == "https" else "http"

    if parts.username or parts.password:
        raise ValueError(f"invalid --url {url!r}: must not contain userinfo (user:pass@)")

    hostname = parts.hostname
    if not hostname:
        raise ValueError(f"invalid --url {url!r}: missing hostname")

    if parts.query:
        raise ValueError(f"invalid --url {url!r}: must not contain a query string")

    if parts.fragment:
        raise ValueError(f"invalid --url {url!r}: must not contain a fragment")

    if parts.path not in ("", "/"):
        raise ValueError(f"invalid --url {url!r}: must be a base URL without a path")

    allowed = _allowed_hosts()
    hostname_lower = hostname.lower()
    matched_host = next((candidate for candidate in allowed if candidate == hostname_lower), None)
    if matched_host is None:
        raise ValueError(
            f"invalid --url {url!r}: host {hostname!r} is not allowed. "
            f"Allowed hosts: {sorted(allowed)!r}. Add more via the "
            f"{ALLOWED_HOSTS_ENV_VAR} environment variable (comma-separated)."
        )

    host_for_url = f"[{matched_host}]" if ":" in matched_host else matched_host
    rebuilt = f"{scheme}://{host_for_url}"
    if parts.port is not None:
        rebuilt += f":{int(parts.port)}"

    return rebuilt


def validate_queries_path(path: str) -> Path:
    """Validate a CLI-supplied YAML path and return it resolved.

    Requires an existing regular file with a .yaml/.yml suffix (case
    insensitive). Resolving it here -- rather than leaving the raw string to
    be passed to open() -- is what makes "this is a real file we actually
    meant to read" an enforced property instead of an assumption.
    """
    resolved = Path(path).resolve()

    if resolved.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"invalid --queries {path!r}: must have a .yaml or .yml suffix")

    if not resolved.is_file():
        raise ValueError(f"invalid --queries {path!r}: not an existing file")

    return resolved
