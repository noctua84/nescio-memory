"""Ingest pipeline: hashing, versioning, and text composition for embeddings and FTS.

The three version constants below are a re-ingest contract: bumping any of them
makes every stored note's `pipeline_version` stale, so the next ingest of each
note will re-split, re-embed, and rewrite `tsv` rather than skip under the
content_hash + pipeline_version match. This is the mechanism that turns changes
to the splitter, FTS weights, or embedding input composition into automatic
re-processing across the corpus.

Each time either side of the FTS contract changes -- the Python composition in
this module (fts_lead_a, fts_lead_c) OR the Postgres setweight/to_tsvector SQL
in app/repositories/note.py -- FTS_VERSION bumps. The tsv is materialised at
ingest time, so a change on either side is invisible to an already-stored row.
"""

import hashlib

from app.core.splitter import SPLITTER_VERSION
from app.config import settings

# Bumped whenever fts_lead_a/fts_lead_c change, OR whenever the setweight/
# to_tsvector expression in app/repositories/note.py:replace_passages changes
# (the weights, the 'english' configuration, or which part goes at which weight).
FTS_VERSION = "1"

# Bumped whenever embedding_input's composition changes -- which fields it
# joins, in what order, with what separator. A stored embedding was produced
# from one exact string; a different string is a different vector.
EMBED_INPUT_VERSION = "1"


def content_hash(content: str) -> str:
    """SHA256 hexdigest of the raw submitted content, before splitting."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def pipeline_version() -> str:
    """Every version the stored passages depend on, in one comparable string.

    Example: "s2+fts1+ei1+ollama+1024". Read from settings at CALL time, not
    import time, so tests can monkeypatch. Ingest skips a note only when the
    content hash AND this string both match, so adding a component here makes
    a change in that component force a re-ingest -- which is the point.

    What this string deliberately CANNOT see: a change in the embedding model
    behind an unchanged backend name and dimension (e.g. a re-pulled Ollama
    tag). That is what the `force` flag on POST /api/v2/ingest and
    scripts/reingest.py --force exist for.
    """
    return (
        f"s{SPLITTER_VERSION}"
        f"+fts{FTS_VERSION}"
        f"+ei{EMBED_INPUT_VERSION}"
        f"+{settings.embedding_backend}"
        f"+{settings.embedding_dimension}"
    )


def embedding_input(name: str | None, heading_path: str | None, content: str) -> str:
    """What gets embedded: the non-empty parts joined by newline.

    Composed here at embed time and never stored. `description` is deliberately
    absent (decision 6 in the slice plan). Changing this bumps EMBED_INPUT_VERSION.
    """
    parts = []
    if name:
        parts.append(name)
    if heading_path:
        parts.append(heading_path)
    if content:
        parts.append(content)
    return "\n".join(parts)


def fts_lead_a(name: str | None, heading_path: str | None) -> str:
    """The A-weighted FTS document: name and heading_path joined by space.

    Composed in Python precisely so the SQL never concatenates a NULL -- in
    Postgres, 'x' || NULL is NULL, which would silently blank the whole
    A-weight section for any note without a name or heading path.
    Changing this bumps FTS_VERSION.
    """
    parts = []
    if name:
        parts.append(name)
    if heading_path:
        parts.append(heading_path)
    return " ".join(parts)


def fts_lead_c(description: str | None) -> str:
    """The C-weighted FTS document: the note's description, or empty string.

    C rather than B (decision 9): description is note-level and duplicated
    onto every passage of the note, so at content's weight it would score all
    of a note's passages identically and could outrank a passage that actually
    discusses the term. Changing this bumps FTS_VERSION.
    """
    return description or ""
