import dspy
import pytest

import dspy_dropin
from dspy_dropin.testing import make_dummy_lm


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in [
        "DSPY_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)
    dspy_dropin.reset()
    yield
    dspy_dropin.reset()


@pytest.fixture
def dummy():
    def _use(answers=None, **kw):
        if answers is None:
            lm = make_dummy_lm(**kw)
        else:
            lm = dspy.utils.DummyLM(answers)
        dspy_dropin.configure(lm=lm, force=True)
        return lm

    return _use
