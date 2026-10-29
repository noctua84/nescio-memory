"""Tests for app.core.pipeline module."""

import re
import pytest
from app.core import pipeline
from app.config import settings


class TestContentHash:
    """Tests for content_hash function."""

    def test_different_inputs_produce_different_hashes(self):
        """content_hash('a') != content_hash('b')."""
        hash_a = pipeline.content_hash("a")
        hash_b = pipeline.content_hash("b")
        assert hash_a != hash_b

    def test_same_input_produces_same_hash(self):
        """Same input produces same output across calls."""
        hash1 = pipeline.content_hash("test content")
        hash2 = pipeline.content_hash("test content")
        assert hash1 == hash2

    def test_hash_is_sha256_hexdigest(self):
        """Hash output is a valid SHA256 hexdigest."""
        hash_result = pipeline.content_hash("test")
        assert len(hash_result) == 64  # SHA256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in hash_result)


class TestPipelineVersion:
    """Tests for pipeline_version function."""

    def test_format_matches_regex(self):
        r"""pipeline_version() matches ^s\d+\+fts\d+\+ei\d+\+\w+\+\d+$."""
        pv = pipeline.pipeline_version()
        pattern = r"^s\d+\+fts\d+\+ei\d+\+\w+\+\d+$"
        assert re.match(pattern, pv), f"Format mismatch: {pv}"

    def test_includes_all_components(self):
        """pipeline_version includes SPLITTER_VERSION, FTS_VERSION, EMBED_INPUT_VERSION, backend, dimension."""
        pv = pipeline.pipeline_version()
        assert f"s{pipeline.SPLITTER_VERSION}" in pv
        assert f"fts{pipeline.FTS_VERSION}" in pv
        assert f"ei{pipeline.EMBED_INPUT_VERSION}" in pv
        assert settings.embedding_backend in pv
        assert str(settings.embedding_dimension) in pv

    def test_reads_from_settings_at_call_time(self, monkeypatch):
        """pipeline_version() reads from settings at CALL time, not import time."""
        original_pv = pipeline.pipeline_version()
        monkeypatch.setattr(settings, "embedding_dimension", 999)
        new_pv = pipeline.pipeline_version()
        assert original_pv != new_pv
        assert "999" in new_pv


class TestEmbeddingInput:
    """Tests for embedding_input function."""

    def test_none_none_content_returns_content(self):
        """embedding_input(None, None, 'c') == 'c'."""
        result = pipeline.embedding_input(None, None, "c")
        assert result == "c"

    def test_name_and_content(self):
        """embedding_input joins name and content with newline."""
        result = pipeline.embedding_input("n", None, "c")
        assert result == "n\nc"

    def test_all_parts(self):
        """embedding_input joins name, heading_path, and content with newlines."""
        result = pipeline.embedding_input("n", "h", "c")
        assert result == "n\nh\nc"

    def test_empty_parts_omitted(self):
        """Empty strings are omitted from the result."""
        result = pipeline.embedding_input("name", "", "content")
        assert result == "name\ncontent"

    def test_description_not_in_signature(self):
        """'description' does not appear in embedding_input signature."""
        import inspect
        sig = inspect.signature(pipeline.embedding_input)
        # Check that the parameter names are only name, heading_path, content
        param_names = list(sig.parameters.keys())
        assert param_names == ["name", "heading_path", "content"]
        assert "description" not in param_names


class TestFtsLeadA:
    """Tests for fts_lead_a function."""

    def test_none_none_returns_empty(self):
        """fts_lead_a(None, None) == ''."""
        assert pipeline.fts_lead_a(None, None) == ""

    def test_name_only(self):
        """fts_lead_a('n', None) == 'n'."""
        assert pipeline.fts_lead_a("n", None) == "n"

    def test_heading_only(self):
        """fts_lead_a(None, 'h') == 'h'."""
        assert pipeline.fts_lead_a(None, "h") == "h"

    def test_both_parts(self):
        """fts_lead_a joins name and heading_path with space."""
        result = pipeline.fts_lead_a("name", "heading")
        assert result == "name heading"

    def test_empty_strings_omitted(self):
        """Empty strings are omitted from the result."""
        assert pipeline.fts_lead_a("", "heading") == "heading"
        assert pipeline.fts_lead_a("name", "") == "name"


class TestFtsLeadC:
    """Tests for fts_lead_c function."""

    def test_none_returns_empty(self):
        """fts_lead_c(None) == ''."""
        assert pipeline.fts_lead_c(None) == ""

    def test_empty_string_returns_empty(self):
        """fts_lead_c('') == ''."""
        assert pipeline.fts_lead_c("") == ""

    def test_description_returned(self):
        """fts_lead_c returns the description when provided."""
        assert pipeline.fts_lead_c("description") == "description"


class TestVersionConstants:
    """Tests for version constants."""

    def test_fts_version_is_string(self):
        """FTS_VERSION is a string."""
        assert isinstance(pipeline.FTS_VERSION, str)

    def test_embed_input_version_is_string(self):
        """EMBED_INPUT_VERSION is a string."""
        assert isinstance(pipeline.EMBED_INPUT_VERSION, str)

    def test_splitter_version_imported(self):
        """SPLITTER_VERSION is imported from app.core.splitter."""
        assert hasattr(pipeline, "SPLITTER_VERSION")
        assert isinstance(pipeline.SPLITTER_VERSION, str)
