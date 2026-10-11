"""Serve your own (optionally optimized) modules: POST /m/triage {"ticket": "..."}."""

from dspy_dropin import task
from dspy_dropin.server import serve

modules = {
    "triage": task("ticket -> category, priority: int"),
    "qa": task("question -> answer", reasoning=True),
}
# modules["qa"].load("qa_optimized.json")  # load an optimized program

if __name__ == "__main__":
    serve(port=8000, modules=modules)
