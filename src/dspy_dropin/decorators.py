"""``@program``: typed Python function -> DSPy module."""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any

import dspy

from .config import ensure_configured


def signature_from_function(fn: Callable) -> type[dspy.Signature]:
    hints = typing.get_type_hints(fn, include_extras=True)
    params = inspect.signature(fn).parameters
    fields: dict[str, Any] = {}
    for name in params:
        fields[name] = (hints.get(name, str), dspy.InputField())
    ret = hints.get("return", str)
    out_name = getattr(fn, "__dropin_output__", None) or "output"
    origin = typing.get_origin(ret)
    if isinstance(ret, type) and issubclass(ret, tuple) or origin is tuple:
        raise TypeError("tuple returns unsupported; return a pydantic model or use outputs=")
    fields[out_name] = (ret, dspy.OutputField())
    doc = inspect.getdoc(fn) or f"Perform the task `{fn.__name__}`."
    return dspy.Signature(fields, doc)


class Program(dspy.Module):
    def __init__(self, fn: Callable, reasoning: bool, output: str):
        super().__init__()
        fn.__dropin_output__ = output  # type: ignore[attr-defined]
        self.fn = fn
        self.output = output
        self.signature = signature_from_function(fn)
        self.predictor = dspy.ChainOfThought(self.signature) if reasoning else dspy.Predict(self.signature)
        functools.update_wrapper(self, fn)

    def forward(self, *args: Any, **kwargs: Any):
        bound = inspect.signature(self.fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return self.predictor(**bound.arguments)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ensure_configured()
        return getattr(super().__call__(*args, **kwargs), self.output)

    def predict(self, *args: Any, **kwargs: Any) -> dspy.Prediction:
        """Return the full Prediction (e.g. with ``.reasoning``)."""
        ensure_configured()
        return super().__call__(*args, **kwargs)


def program(fn: Callable | None = None, *, reasoning: bool = False, output: str = "output"):
    """Turn a typed, docstring'd function into an LM-backed DSPy module.

    >>> @program
    ... def sentiment(review: str) -> Literal["pos", "neg"]:
    ...     \"\"\"Classify the sentiment of a product review.\"\"\"
    """

    def wrap(f: Callable) -> Program:
        return Program(f, reasoning=reasoning, output=output)

    return wrap(fn) if fn is not None else wrap
