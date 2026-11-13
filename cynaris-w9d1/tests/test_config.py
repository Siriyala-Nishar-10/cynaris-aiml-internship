"""Tests for settings parsing."""
import pytest

from research_crew.config import PROJECT_ROOT, load_settings


def test_defaults():
    s = load_settings({})
    assert s.llm_model == "llama3.2" and s.ollama_host == "http://localhost:11434"
    assert s.outputs_dir == PROJECT_ROOT / "outputs"
    assert (s.max_results, s.max_iter) == (5, 6)


def test_overrides(tmp_path):
    s = load_settings({"LLM_MODEL": "qwen2.5:3b", "OUTPUTS_DIR": str(tmp_path), "MAX_RESULTS": "8"})
    assert s.llm_model == "qwen2.5:3b" and s.outputs_dir == tmp_path and s.max_results == 8


def test_invalid_numbers_are_rejected():
    for env in [{"MAX_RESULTS": "abc"}, {"MAX_RESULTS": "0"}, {"MAX_ITER": "-1"}]:
        with pytest.raises(ValueError):
            load_settings(env)
