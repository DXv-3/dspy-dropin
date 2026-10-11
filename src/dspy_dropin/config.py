"""Environment-driven, idempotent DSPy configuration."""

from __future__ import annotations

import os
import threading
from typing import Any

import dspy

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"configured": False, "model": None, "lm": None}

# (env var that must be set, default model) in priority order.
PROVIDERS: list[tuple[str, str]] = [
    ("OPENAI_API_KEY", "openai/gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-3-5-haiku-latest"),
    ("GROQ_API_KEY", "groq/llama-3.1-8b-instant"),
    ("GEMINI_API_KEY", "gemini/gemini-1.5-flash"),
    ("MISTRAL_API_KEY", "mistral/mistral-small-latest"),
    ("TOGETHER_API_KEY", "together_ai/meta-llama/Llama-3-8b-chat-hf"),
    ("OLLAMA_BASE_URL", "ollama_chat/llama3.1"),
]


def detect_model(env: dict[str, str] | None = None) -> str | None:
    """Return the LiteLLM model string implied by the environment, or None."""
    env = dict(os.environ if env is None else env)
    if env.get("DSPY_MODEL"):
        return env["DSPY_MODEL"]
    for var, model in PROVIDERS:
        if env.get(var):
            return model
    return None


def _env_num(name: str, default: float, cast: type = float, lo: float | None = None, hi: float | None = None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return cast(default)
    try:
        v = cast(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a number (got {raw[:20]!r})") from None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise RuntimeError(f"{name} must be between {lo} and {hi}")
    return v


def _lm_kwargs(model: str) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "cache": os.environ.get("DSPY_CACHE", "1") not in ("0", "false", "False"),
        "temperature": _env_num("DSPY_TEMPERATURE", 0.0, float, 0.0, 2.0),
        "max_tokens": _env_num("DSPY_MAX_TOKENS", 4000, int, 1, 1_000_000),
        # Hard per-request timeout (seconds) so a hung provider can't hang the CLI/server forever.
        "timeout": _env_num("DSPY_TIMEOUT", 60.0, float, 1.0, 3600.0),
        "num_retries": _env_num("DSPY_NUM_RETRIES", 3, int, 0, 10),
    }
    if os.environ.get("DSPY_API_BASE"):
        kw["api_base"] = os.environ["DSPY_API_BASE"]
    elif model.startswith("ollama") and os.environ.get("OLLAMA_BASE_URL"):
        kw["api_base"] = os.environ["OLLAMA_BASE_URL"]
        kw["api_key"] = ""
    if os.environ.get("DSPY_API_KEY"):
        kw["api_key"] = os.environ["DSPY_API_KEY"]
    return kw


def configure(model: str | None = None, lm: Any = None, force: bool = False, **lm_kwargs: Any) -> Any:
    """Configure DSPy once. Safe to call repeatedly; returns the active LM.

    Pass ``lm`` to use any ``dspy.LM``-compatible object (e.g. ``DummyLM`` in tests),
    or ``model`` to override env detection. ``force=True`` reconfigures.
    """
    with _LOCK:
        if _STATE["configured"] and not force and model is None and lm is None:
            return _STATE["lm"]
        if lm is None:
            model = model or detect_model()
            if model is None:
                raise RuntimeError(
                    "No LM configured. Set DSPY_MODEL (e.g. openai/gpt-4o-mini) or a provider key such as "
                    "OPENAI_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY / OLLAMA_BASE_URL. "
                    "Run `dspy-dropin doctor` for details."
                )
            if model == "dummy" or model.startswith("dummy/"):
                from .testing import make_dummy_lm

                lm = make_dummy_lm()
                dspy.configure(lm=lm)
                _STATE.update(configured=True, model="dummy", lm=lm)
                return lm
            kw = _lm_kwargs(model)
            kw.update(lm_kwargs)
            lm = dspy.LM(model, **kw)
        dspy.configure(lm=lm)
        _STATE.update(configured=True, model=getattr(lm, "model", model), lm=lm)
        return lm


def ensure_configured() -> None:
    if not _STATE["configured"] and dspy.settings.lm is None:
        configure()


def is_configured() -> bool:
    return bool(_STATE["configured"] or dspy.settings.lm is not None)


def reset() -> None:
    with _LOCK:
        _STATE.update(configured=False, model=None, lm=None)
        dspy.configure(lm=None)
