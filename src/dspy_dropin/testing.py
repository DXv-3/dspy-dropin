"""Offline fakes for tests, demos and CI smoke checks (no API keys needed).

``DSPY_MODEL=dummy`` makes ``configure()`` use :func:`make_dummy_lm`: an LM that fills *every*
requested output field with ``DSPY_DUMMY_ANSWER`` (default ``"42"``), whatever the signature.
"""

from __future__ import annotations

import itertools
import os
import re
import warnings
from typing import Any

try:  # DummyLM is public API, but guard the import so a DSPy layout change gives a clear error.
    from dspy.utils import DummyLM
except ImportError as _e:  # pragma: no cover - depends on the installed DSPy
    raise ImportError(
        "dspy_dropin.testing needs dspy.utils.DummyLM (dspy>=3.4,<3.5 is supported); "
        "install a supported DSPy version or use a real model instead of DSPY_MODEL=dummy"
    ) from _e

COMMON_FIELDS = ("answer", "output", "summary", "label", "result", "reasoning")
_OUT_BLOCK = re.compile(r"Your output fields are:\s*(.*?)(?:\n\s*\n|All interactions)", re.S)
_FIELD = re.compile(r"^\s*\d+\.\s*`([^`]+)`\s*\(([^\n]*)\)", re.M)
_LIT = re.compile(r"Literal\[\s*['\"]([^'\"]*)['\"]")


def _value_for(type_text: str, value: str) -> Any:
    """Make the canned value parse for common annotations (Literal -> first option, list -> [value])."""
    if m := _LIT.search(type_text):
        return m.group(1)
    if type_text.startswith("list"):
        return [value]
    if type_text == "bool":
        return True
    return value


# EchoLM relies on two DSPy *internals* of DummyLM: the engine calling ``_use_example(messages)`` when
# ``follow_examples=True``, and ``_format_answer_fields(dict)``. Both are guarded: if DSPy stops calling the hook we
# still answer COMMON_FIELDS from the scripted rows, and if the formatter disappears we emit ChatAdapter-style
# ``[[ ## field ## ]]`` blocks ourselves. ``hook_supported()`` reports which path is active (checked in the tests).
_HOOKS = ("_use_example", "_format_answer_fields")


def hook_supported() -> bool:
    return all(callable(getattr(DummyLM, h, None)) for h in _HOOKS)


def _chat_format(values: dict[str, Any]) -> str:
    import json as _json

    parts = [f"[[ ## {k} ## ]]\n{v if isinstance(v, str) else _json.dumps(v)}" for k, v in values.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


class EchoLM(DummyLM):
    """DummyLM that answers whatever output fields the prompt asks for with a fixed value."""

    def __init__(self, value: str = "42", extra: dict[str, Any] | None = None):
        row = {f: value for f in COMMON_FIELDS}
        super().__init__(itertools.repeat(row), follow_examples=True)  # type: ignore[arg-type]
        self.value = value
        self.extra = dict(extra or {})
        if not hook_supported():
            warnings.warn(
                "This DSPy version changed DummyLM internals; EchoLM will only fill common output fields "
                f"({', '.join(COMMON_FIELDS)}). Supported: dspy>=3.4,<3.5.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _fields_from(self, messages) -> list[tuple[str, str]]:
        try:
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            block = _OUT_BLOCK.search(system if isinstance(system, str) else "")
            fields = _FIELD.findall(block.group(1)) if block else []
        except Exception:  # noqa: BLE001 - prompt format is a DSPy internal; never crash the fake
            fields = []
        return fields or [(n, "str") for n in COMMON_FIELDS]

    def _format(self, values: dict[str, Any]) -> str:
        fmt = getattr(super(), "_format_answer_fields", None)
        if callable(fmt):
            try:
                return fmt(values)
            except Exception:  # noqa: BLE001
                pass
        return _chat_format(values)

    def _use_example(self, messages):  # called by DSPy's dummy engine with the full prompt
        fields = self._fields_from(messages or [])
        return self._format({n: self.extra.get(n, _value_for(t.strip(), self.value)) for n, t in fields})


def make_dummy_lm(answer: str | None = None, extra: dict[str, Any] | None = None) -> EchoLM:
    value = answer if answer is not None else os.environ.get("DSPY_DUMMY_ANSWER", "42")
    return EchoLM(value, extra)
