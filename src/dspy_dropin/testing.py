"""Offline fakes for tests, demos and CI smoke checks (no API keys needed).

``DSPY_MODEL=dummy`` makes ``configure()`` use :func:`make_dummy_lm`: an LM that fills *every*
requested output field with ``DSPY_DUMMY_ANSWER`` (default ``"42"``), whatever the signature.
"""

from __future__ import annotations

import itertools
import os
import re
from typing import Any

from dspy.utils import DummyLM

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


class EchoLM(DummyLM):
    """DummyLM that answers whatever output fields the prompt asks for with a fixed value."""

    def __init__(self, value: str = "42", extra: dict[str, Any] | None = None):
        row = {f: value for f in COMMON_FIELDS}
        super().__init__(itertools.repeat(row), follow_examples=True)  # type: ignore[arg-type]
        self.value = value
        self.extra = dict(extra or {})

    def _use_example(self, messages):  # called by DSPy's dummy engine with the full prompt
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        block = _OUT_BLOCK.search(system)
        fields = _FIELD.findall(block.group(1)) if block else []
        if not fields:
            fields = [(n, "str") for n in COMMON_FIELDS]
        return self._format_answer_fields({n: self.extra.get(n, _value_for(t.strip(), self.value)) for n, t in fields})


def make_dummy_lm(answer: str | None = None, extra: dict[str, Any] | None = None) -> EchoLM:
    value = answer if answer is not None else os.environ.get("DSPY_DUMMY_ANSWER", "42")
    return EchoLM(value, extra)
