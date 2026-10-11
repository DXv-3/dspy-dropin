"""dspy_dropin: zero-setup DSPy. ``from dspy_dropin import ask``."""

from .config import configure, detect_model, is_configured, reset
from .decorators import program
from .oneliners import ask, classify, cot, extract, rag, react, summarize, task
from .optimize import load, optimize, save

__version__ = "0.1.0"
__all__ = [
    "ask",
    "task",
    "cot",
    "react",
    "classify",
    "extract",
    "summarize",
    "rag",
    "program",
    "optimize",
    "save",
    "load",
    "configure",
    "detect_model",
    "is_configured",
    "reset",
    "__version__",
]
