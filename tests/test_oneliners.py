from typing import Literal

import pydantic

import dspy_dropin as d


def test_ask(dummy):
    dummy([{"answer": "Paris"}])
    assert d.ask("Capital of France?") == "Paris"


def test_ask_with_context(dummy):
    dummy([{"answer": "3"}])
    assert d.ask("How many?", items=[1, 2, 3]) == "3"


def test_task_positional_and_kw(dummy):
    dummy([{"answer": "4"}, {"answer": "5"}])
    t = d.task("question -> answer")
    assert t("2+2?").answer == "4"
    assert t(question="2+3?").answer == "5"


def test_cot_has_reasoning(dummy):
    dummy([{"reasoning": "because", "answer": "yes"}])
    p = d.cot()("Is water wet?")
    assert p.answer == "yes" and p.reasoning == "because"


def test_classify(dummy):
    dummy([{"label": "spam"}])
    assert d.classify("WIN $$$", ["spam", "ham"]) == "spam"


def test_extract_fields(dummy):
    dummy([{"name": "Ada", "age": "36"}])
    out = d.extract("Ada is 36", {"name": str, "age": int})
    assert out == {"name": "Ada", "age": 36}


def test_extract_pydantic(dummy):
    class Person(pydantic.BaseModel):
        name: str
        age: int

    dummy([{"result": '{"name": "Ada", "age": 36}'}])
    p = d.extract("Ada is 36", Person)
    assert isinstance(p, Person) and p.age == 36


def test_summarize(dummy):
    dummy([{"summary": "short"}])
    assert d.summarize("long text " * 50, max_words=5) == "short"


def test_rag_docs(dummy):
    dummy([{"reasoning": "r", "answer": "blue"}])
    docs = ["The sky is blue.", "Grass is green.", "Bananas are yellow."]
    p = d.rag("What color is the sky?", docs=docs, k=1)
    assert p.answer == "blue" and p.context == ["The sky is blue."]


def test_rag_retriever(dummy):
    dummy([{"reasoning": "r", "answer": "x"}])
    p = d.rag("q", retriever=lambda q: ["a", "b"], k=1)
    assert p.context == ["a"]


def test_react(dummy):
    calls = []

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    dummy(
        [
            {"next_thought": "add", "next_tool_name": "add", "next_tool_args": {"a": 2, "b": 3}},
            {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "2+3=5", "answer": "5"},
        ]
    )
    p = d.react("question -> answer", tools=[add])("what is 2+3?")
    assert p.answer == "5" and calls == [(2, 3)]


def test_program_decorator(dummy):
    @d.program
    def sentiment(review: str) -> Literal["pos", "neg"]:
        """Classify the sentiment of a product review."""

    assert "sentiment" in sentiment.signature.instructions.lower()
    dummy([{"output": "neg"}])
    assert sentiment("terrible") == "neg"


def test_program_reasoning_named_output(dummy):
    @d.program(reasoning=True, output="total")
    def add(a: int, b: int = 1) -> int:
        """Add numbers."""

    dummy([{"reasoning": "r", "total": "3"}])
    assert add.predict(2).total == 3
