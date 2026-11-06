"""Request validation shared by more than one /api/v1 route module.

`path_or_uri` is accepted by ingest (written) and by delete-by-path (looked
up). Both apply the same rule so a value the service would refuse to store is
refused when asked for too, instead of one route quietly accepting what the
other rejects.
"""

import re
from pathlib import Path

from fastapi import HTTPException

_URI_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")


def validate_path_or_uri(path_or_uri: str) -> None:
    """Accept any URI as-is; reject an absolute path or one with '..'.

    A value matching ``scheme://`` is treated as a URI and is not subject to
    the path rule at all. Anything else is a path, stored and reflected back
    the same way the removed learnings API's file_path was, so it gets the
    same rule: no leading '/'
    and no '..' path component.
    """
    if _URI_RE.match(path_or_uri):
        return
    if path_or_uri.startswith("/") or ".." in Path(path_or_uri).parts:
        raise HTTPException(
            status_code=400,
            detail=(
                "path_or_uri must be a relative path without '..', or an "
                "absolute URI"
            ),
        )
