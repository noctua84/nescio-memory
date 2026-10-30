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
    host: no userinfo (credentials embedded in the URL), no query string or
    fragment, and a hostname that is on the allowlist. The allowlist is
    DEFAULT_ALLOWED_HOSTS plus whatever NESCIO_ALLOWED_HOSTS adds, so a
    non-local target has to be opted into explicitly rather than accepted
    because it merely looked well-formed.

    The return value is rebuilt from the parsed components (never the raw
    input string) so that quirks urlsplit tolerates but does not surface
    back out -- e.g. a trailing slash -- cannot smuggle anything past the
    validation that just happened.
    """
    parts: SplitResult = urlsplit(url)

    if parts.scheme not in ("http", "https"):
        raise ValueError(f"invalid --url {url!r}: scheme must be http or https")

    if parts.username or parts.password:
        raise ValueError(f"invalid --url {url!r}: must not contain userinfo (user:pass@)")

    hostname = parts.hostname
    if not hostname:
        raise ValueError(f"invalid --url {url!r}: missing hostname")

    if parts.query:
        raise ValueError(f"invalid --url {url!r}: must not contain a query string")

    if parts.fragment:
        raise ValueError(f"invalid --url {url!r}: must not contain a fragment")

    allowed = _allowed_hosts()
    if hostname.lower() not in allowed:
        raise ValueError(
            f"invalid --url {url!r}: host {hostname!r} is not allowed. "
            f"Allowed hosts: {sorted(allowed)!r}. Add more via the "
            f"{ALLOWED_HOSTS_ENV_VAR} environment variable (comma-separated)."
        )

    host_for_url = f"[{hostname}]" if ":" in hostname else hostname
    rebuilt = f"{parts.scheme}://{host_for_url}"
    if parts.port is not None:
        rebuilt += f":{parts.port}"
    path = parts.path.rstrip("/")
    if path:
        rebuilt += path

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
