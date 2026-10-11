"""Plain Python script. Run: DSPY_MODEL=dummy python examples/quickstart.py (or set OPENAI_API_KEY)."""

from typing import Literal

import pydantic

from dspy_dropin import ask, classify, cot, extract, program, rag, react, summarize, task

print(ask("What is the capital of France?"))
print(task("question -> answer: float")("What is 0.1 + 0.2?").answer)
print(cot()("Is 97 prime?").reasoning)
print(classify("I want a refund now!", ["billing", "support", "sales"]))
print(summarize("DSPy is a framework for programming, not prompting, language models. " * 5, max_words=15))


class Invoice(pydantic.BaseModel):
    vendor: str
    total: float


print(extract("ACME Corp invoice, total due $1,204.50", {"vendor": str, "total": float}))
_ = Invoice  # with a real LM: extract(text, Invoice) -> Invoice(...)

docs = ["DSPy was created at Stanford NLP.", "Bananas are yellow.", "DSPy optimizers tune prompts."]
print(rag("Where was DSPy created?", docs=docs).answer)


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


agent = react("question -> answer", tools=[get_weather])
# print(agent("Weather in Paris?").answer)  # needs a real LM (tool-calling loop)


@program
def triage(ticket: str) -> Literal["bug", "feature", "question"]:
    """Triage a GitHub issue."""


print(triage("App crashes when I click save"))
