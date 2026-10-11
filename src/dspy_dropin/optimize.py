"""Optimization + persistence helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import dspy

from .config import ensure_configured


def to_examples(data: Sequence[Any], inputs: Sequence[str] | None = None) -> list[dspy.Example]:
    """Accept dspy.Example or dicts. Dicts: ``inputs`` names input keys (default: all but the last key)."""
    out = []
    for row in data:
        if isinstance(row, dspy.Example):
            out.append(row)
            continue
        keys = list(row.keys())
        ins = list(inputs) if inputs else keys[:-1]
        out.append(dspy.Example(**row).with_inputs(*ins))
    return out


def exact_match(field: str = "answer") -> Callable:
    def metric(example, pred, trace=None):
        return str(getattr(example, field)).strip().lower() == str(getattr(pred, field, "")).strip().lower()

    return metric


def optimize(
    module: dspy.Module,
    trainset: Sequence[Any],
    metric: Callable | None = None,
    strategy: str = "bootstrap",
    valset: Sequence[Any] | None = None,
    inputs: Sequence[str] | None = None,
    save_to: str | Path | None = None,
    **kwargs: Any,
) -> dspy.Module:
    """Compile ``module`` with a DSPy optimizer.

    strategy: "bootstrap" (BootstrapFewShot), "random" (BootstrapFewShotWithRandomSearch),
    "mipro" (MIPROv2, auto="light" by default), "labeled" (LabeledFewShot).
    """
    ensure_configured()
    train = to_examples(trainset, inputs)
    val = to_examples(valset, inputs) if valset else None
    metric = metric or exact_match()
    s = strategy.lower()
    if s == "bootstrap":
        opt = dspy.BootstrapFewShot(metric=metric, **{"max_bootstrapped_demos": 4, "max_labeled_demos": 8, **kwargs})
        compiled = opt.compile(module, trainset=train)
    elif s == "random":
        opt = dspy.BootstrapFewShotWithRandomSearch(metric=metric, **kwargs)
        compiled = opt.compile(module, trainset=train, valset=val)
    elif s == "mipro":
        opt = dspy.MIPROv2(metric=metric, **{"auto": "light", **kwargs})
        compiled = opt.compile(module, trainset=train, valset=val)
    elif s == "labeled":
        opt = dspy.LabeledFewShot(**{"k": 8, **kwargs})
        compiled = opt.compile(module, trainset=train)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    if save_to:
        save(compiled, save_to)
    return compiled


def save(module: dspy.Module, path: str | Path) -> Path:
    """Save module state (demos, instructions) to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(p))
    return p


def load(module: dspy.Module, path: str | Path) -> dspy.Module:
    """Load JSON state into a freshly-constructed module of the same shape; returns it."""
    module.load(str(path))
    return module


def evaluate(
    module: dspy.Module,
    devset: Sequence[Any],
    metric: Callable | None = None,
    inputs: Sequence[str] | None = None,
    num_threads: int = 1,
) -> float:
    ensure_configured()
    ev = dspy.Evaluate(
        devset=to_examples(devset, inputs),
        metric=metric or exact_match(),
        num_threads=num_threads,
        display_progress=False,
    )
    res = ev(module)
    return float(getattr(res, "score", res))
