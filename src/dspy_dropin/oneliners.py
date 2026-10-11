"""One-line helpers over DSPy modules."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import dspy

from .config import ensure_configured


class _Lazy(dspy.Module):
    """Wraps a DSPy module so the LM is auto-configured on first call."""

    def __init__(self, inner: dspy.Module):
        super().__init__()
        self.inner = inner

    def __call__(self, *args: Any, **kwargs: Any):
        # Configure *before* dspy.Module.__call__ snapshots the settings context.
        ensure_configured()
        return super().__call__(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.inner(*args, **kwargs)


def _sig_inputs(sig: str | type[dspy.Signature]) -> list[str]:
    s = dspy.ensure_signature(sig)
    return list(s.input_fields.keys())


def _call(module: dspy.Module, sig, args, kwargs):
    if args:
        names = _sig_inputs(sig)
        kwargs = {**dict(zip(names, args, strict=False)), **kwargs}
    return module(**kwargs)


class _Task(_Lazy):
    def __init__(self, sig, inner):
        super().__init__(inner)
        self._sig = sig

    def forward(self, *args, **kwargs):
        return _call(self.inner, self._sig, args, kwargs)


def task(
    signature: str | type[dspy.Signature], instructions: str | None = None, reasoning: bool = False
) -> dspy.Module:
    """Build a callable module from a signature: ``task("question -> answer")("2+2?")``."""
    sig = dspy.ensure_signature(signature, instructions)
    inner = dspy.ChainOfThought(sig) if reasoning else dspy.Predict(sig)
    return _Task(sig, inner)


def cot(signature: str | type[dspy.Signature] = "question -> answer", instructions: str | None = None):
    """Chain-of-thought module (``.reasoning`` is available on the result)."""
    return task(signature, instructions, reasoning=True)


def react(signature: str | type[dspy.Signature], tools: Sequence[Callable], max_iters: int = 10):
    """ReAct agent module with Python functions as tools."""
    sig = dspy.ensure_signature(signature)
    return _Task(sig, dspy.ReAct(sig, tools=list(tools), max_iters=max_iters))


def ask(prompt: str, reasoning: bool = False, **context: Any) -> str:
    """Ask a question, get a string back."""
    if context:
        sig = "context, question -> answer"
        pred = task(sig, reasoning=reasoning)(context=json.dumps(context, default=str), question=prompt)
    else:
        pred = task("question -> answer", reasoning=reasoning)(question=prompt)
    return pred.answer


def classify(text: str, labels: Sequence[str], instructions: str | None = None) -> str:
    """Classify ``text`` into exactly one of ``labels``."""
    labels = list(labels)
    if not labels:
        raise ValueError("labels must be non-empty")
    Lbl = Literal[tuple(labels)]  # type: ignore[valid-type]
    sig = dspy.Signature(
        {"text": (str, dspy.InputField()), "label": (Lbl, dspy.OutputField())},
        instructions or "Classify the text into exactly one of the allowed labels.",
    )
    return task(sig)(text=text).label


def extract(text: str, schema: Any, instructions: str | None = None) -> Any:
    """Extract structured data. ``schema`` = pydantic model, dict {field: type}, or list of field names.

    Returns a pydantic instance for pydantic schemas, otherwise a dict.
    """
    instr = instructions or "Extract the requested information from the text."
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        BaseModel = None  # type: ignore
    if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
        sig = dspy.Signature({"text": (str, dspy.InputField()), "result": (schema, dspy.OutputField())}, instr)
        return task(sig)(text=text).result
    fields: dict[str, Any] = dict(schema) if isinstance(schema, dict) else {name: str for name in schema}
    spec = {"text": (str, dspy.InputField())}
    spec.update({k: (v, dspy.OutputField()) for k, v in fields.items()})
    pred = task(dspy.Signature(spec, instr))(text=text)
    return {k: getattr(pred, k) for k in fields}


def summarize(text: str, style: str = "concise", max_words: int | None = None) -> str:
    instr = f"Write a {style} summary of the text."
    if max_words:
        instr += f" Use at most {max_words} words."
    return task("text -> summary", instr)(text=text).summary


def _keyword_retrieve(question: str, docs: Iterable[str], k: int) -> list[str]:
    q = {w.lower().strip(".,?!") for w in question.split() if len(w) > 2}
    scored = [(sum(w in d.lower() for w in q), i, d) for i, d in enumerate(docs)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [d for _, _, d in scored[:k]]


def rag(
    question: str,
    docs: Sequence[str] | None = None,
    retriever: Callable[[str], Sequence[str]] | None = None,
    k: int = 5,
    reasoning: bool = True,
) -> dspy.Prediction:
    """Answer from documents. Pass ``docs`` (keyword-ranked) or a ``retriever(question)->list[str]``.

    Returns a Prediction with ``.answer`` and ``.context``.
    """
    if retriever is not None:
        ctx = list(retriever(question))[:k]
    elif docs is not None:
        ctx = _keyword_retrieve(question, docs, k)
    else:
        raise ValueError("pass docs=... or retriever=...")
    pred = task("context: list[str], question -> answer", reasoning=reasoning)(context=ctx, question=question)
    return dspy.Prediction(answer=pred.answer, context=ctx, reasoning=getattr(pred, "reasoning", None))
