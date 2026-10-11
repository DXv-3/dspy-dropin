import dspy
import pytest

import dspy_dropin
from dspy_dropin.config import configure, detect_model


def test_detect_priority():
    assert detect_model({}) is None
    assert detect_model({"OPENAI_API_KEY": "x"}) == "openai/gpt-4o-mini"
    assert detect_model({"ANTHROPIC_API_KEY": "x"}).startswith("anthropic/")
    assert detect_model({"OLLAMA_BASE_URL": "http://h"}).startswith("ollama")
    assert detect_model({"DSPY_MODEL": "groq/x", "OPENAI_API_KEY": "k"}) == "groq/x"


def test_no_model_raises():
    with pytest.raises(RuntimeError, match="DSPY_MODEL"):
        configure()


def test_env_builds_lm_idempotent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    lm1 = configure()
    lm2 = configure()
    assert lm1 is lm2 and isinstance(lm1, dspy.LM)
    assert lm1.model == "openai/gpt-4o-mini"
    assert dspy.settings.lm is lm1
    assert configure(force=True) is not lm1


def test_ollama_api_base(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    lm = configure()
    assert lm.kwargs.get("api_base") == "http://localhost:11434"


def test_dummy_model_env(monkeypatch):
    monkeypatch.setenv("DSPY_MODEL", "dummy")
    monkeypatch.setenv("DSPY_DUMMY_ANSWER", "hi")
    assert dspy_dropin.ask("anything") == "hi"


def test_echo_lm_type_aware(monkeypatch):
    monkeypatch.setenv("DSPY_MODEL", "dummy")
    assert dspy_dropin.classify("x", ["a", "b"]) == "a"
    assert dspy_dropin.task("q -> items: list[str], n: int")(q="x").n == 42
