"""Tests for configuration parsing and validation."""
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, load_settings, parse_flag


@pytest.mark.parametrize("value, expected", [
    ("1", True), ("true", True), ("YES", True), (" on ", True),
    ("0", False), ("false", False), ("off", False), ("", False),
])
def test_parse_flag_values(value, expected):
    assert parse_flag(value) is expected


def test_parse_flag_default_when_missing():
    assert parse_flag(None) is False
    assert parse_flag(None, default=True) is True


def test_defaults():
    s = load_settings({})
    assert s.docs_dir == PROJECT_ROOT / "sample_docs"
    assert s.docs_dir.is_absolute()
    assert s.enable_dense is False
    assert (s.chunk_words, s.chunk_overlap) == (150, 20)


def test_environment_overrides(tmp_path):
    s = load_settings({"DOCS_DIR": str(tmp_path), "ENABLE_DENSE": "1", "CHUNK_WORDS": "80",
                       "CHUNK_OVERLAP": "10", "EMBED_MODEL": "my/model"})
    assert s.docs_dir == Path(tmp_path)
    assert s.enable_dense is True
    assert (s.chunk_words, s.chunk_overlap, s.embed_model) == (80, 10, "my/model")


def test_relative_docs_dir_is_resolved_against_project_root():
    assert load_settings({"DOCS_DIR": "my_docs"}).docs_dir == PROJECT_ROOT / "my_docs"


@pytest.mark.parametrize("env", [
    {"CHUNK_WORDS": "abc"},
    {"CHUNK_WORDS": "-5"},
    {"CHUNK_WORDS": "0"},
    {"CHUNK_WORDS": "50", "CHUNK_OVERLAP": "50"},
    {"CHUNK_OVERLAP": "150"},
])
def test_invalid_chunk_settings_are_rejected(env):
    with pytest.raises(ValueError):
        load_settings(env)
