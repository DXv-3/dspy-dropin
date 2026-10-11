# dspy-dropin

**[DSPy](https://github.com/stanfordnlp/dspy) with zero setup.** Install it, set one env var, and you get one-liners,
a `@program` decorator, optimizers with save/load, a CLI, an HTTP server for non-Python workflows, and a GitHub Action.

DSPy is a great way to build reliable LM pipelines. `dspy-dropin` handles the setup so you can use it in a
script, a notebook, a shell pipeline, an n8n flow, or CI without extra wiring. Everything underneath is still plain DSPy,
so you can switch to the full API whenever you need it.

## 60-second quickstart

```bash
pip install "git+https://github.com/DXv-3/dspy-dropin"
export OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY / GROQ_API_KEY / OLLAMA_BASE_URL / DSPY_MODEL=...
dspy-dropin doctor --ping             # check the setup
dspy-dropin ask "Explain DSPy in one sentence"
```

```python
from typing import Literal
from dspy_dropin import ask, task, classify, extract, program

ask("What is the capital of France?")                          # -> "Paris"
qa = task("question -> answer: float")                         # a real dspy.Module
qa("What is 0.1 + 0.2?").answer
classify("I want a refund!", ["billing", "support", "sales"])  # -> "billing"
extract("ACME invoice, total $1,204.50", {"vendor": str, "total": float})

@program
def triage(ticket: str) -> Literal["bug", "feature", "question"]:
    """Triage a GitHub issue."""

triage("App crashes on save")                                  # -> "bug"
```

Want to try it with no API key? Run `export DSPY_MODEL=dummy`. This uses an offline fake LM that fills in every
output field, so you can wire up and test whole pipelines for free.

## Environment

`configure()` runs automatically on first use. It is idempotent and has caching on by default. The model is chosen in this order:

| Variable | Effect |
|---|---|
| `DSPY_MODEL` | Any LiteLLM model string, e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-latest`, `ollama_chat/llama3.1`. `dummy` = offline fake. Takes precedence over everything below. |
| `OPENAI_API_KEY` | → `openai/gpt-4o-mini` |
| `ANTHROPIC_API_KEY` | → `anthropic/claude-3-5-haiku-latest` |
| `GROQ_API_KEY` | → `groq/llama-3.1-8b-instant` |
| `GEMINI_API_KEY` | → `gemini/gemini-1.5-flash` |
| `MISTRAL_API_KEY` | → `mistral/mistral-small-latest` |
| `TOGETHER_API_KEY` | → `together_ai/meta-llama/Llama-3-8b-chat-hf` |
| `OLLAMA_BASE_URL` | → `ollama_chat/llama3.1` at that URL |
| `DSPY_API_BASE` / `DSPY_API_KEY` | Custom endpoint (any OpenAI-compatible server) |
| `DSPY_TEMPERATURE` (0.0), `DSPY_MAX_TOKENS` (4000), `DSPY_CACHE` (1) | LM defaults |
| `DSPY_DUMMY_ANSWER` | Value returned by the `dummy` model (default `42`) |

To set things explicitly, use `dspy_dropin.configure(model="openai/gpt-4o", temperature=0.7)`, or `configure(lm=any_dspy_lm, force=True)`.

## API reference

| Function | Returns |
|---|---|
| `ask(prompt, reasoning=False, **context)` | `str`. Keyword args are passed as JSON context. |
| `task(signature, instructions=None, reasoning=False)` | A callable `dspy.Module`. Accepts positional or keyword inputs and returns a `dspy.Prediction`. |
| `cot(signature="question -> answer")` | A Chain-of-Thought module. The prediction has `.reasoning`. |
| `react(signature, tools=[fn, ...], max_iters=10)` | A ReAct agent module. Tools are typed Python functions with docstrings. |
| `classify(text, labels)` | One of `labels`, enforced with a `Literal` type. |
| `extract(text, schema)` | `schema` can be a pydantic model (returns an instance), a `{field: type}` dict, or a list of names (both return a dict). |
| `summarize(text, style="concise", max_words=None)` | `str` |
| `rag(question, docs=[...] or retriever=fn, k=5)` | A Prediction with `.answer`, `.context`, and `.reasoning`. Plain `docs` are keyword-ranked; plug in any `retriever(q) -> list[str]` for vectors. |
| `@program` / `@program(reasoning=True, output="name")` | Builds a Signature from the function's types and docstring. Calling it returns the output value; `.predict(...)` returns the full Prediction. |
| `optimize(module, trainset, metric=None, strategy=..., save_to=None)` | The compiled module. Strategies: `bootstrap` (BootstrapFewShot), `random` (BootstrapFewShotWithRandomSearch), `mipro` (MIPROv2, `auto="light"`), `labeled`. `trainset` can be `dspy.Example`s or dicts (the last key is the label by default, or pass `inputs=[...]`). The default metric is exact match on `answer`. |
| `save(module, path)` / `load(module, path)` | JSON state (demos and instructions) |
| `dspy_dropin.optimize.evaluate(module, devset, metric)` | `float` score |
| `dspy_dropin.testing.make_dummy_lm(answer)` | An offline LM for your own tests |

### Optimize and reuse

```python
import dspy_dropin as dd
qa = dd.task("question -> answer")
train = [{"question": "2+2?", "answer": "4"}, ...]
best = dd.optimize(qa, train, strategy="mipro", save_to="qa.json")
later = dd.load(dd.task("question -> answer"), "qa.json")
```

## CLI

```bash
dspy-dropin doctor [--ping]                                 # check env and model; --ping makes one live call
dspy-dropin ask "question"            # or: echo "question" | dspy-dropin ask
dspy-dropin run "email -> intent, urgency: int" -i email="Server down!" [--reasoning] [--load qa.json]
dspy-dropin run "a, b -> c" --json '{"a": 1, "b": 2}'
dspy-dropin optimize "question -> answer" train.jsonl -o qa.json --strategy bootstrap [--metric mymod:fn]
dspy-dropin serve --port 8000 [--backend auto|fastapi|stdlib]
dspy-dropin --model ollama_chat/llama3.1 ask "hi"           # per-call model override
```

## HTTP server (for n8n, Zapier, Make, Node, anything)

```bash
dspy-dropin serve --host 0.0.0.0 --port 8000      # stdlib, no extras needed
pip install "dspy-dropin[server] @ git+https://github.com/DXv-3/dspy-dropin"   # optional FastAPI + uvicorn
```

`GET /health`, `POST /ask {"prompt"}`, `POST /run {"signature", "inputs"}`, `POST /m/<name>` for your own modules
(`serve(modules={"triage": my_module})`). See [examples/http_n8n_zapier.md](examples/http_n8n_zapier.md).
The server has no authentication, so bind to localhost or put it behind your own proxy.

## GitHub Action

```yaml
- uses: DXv-3/dspy-dropin@main
  id: ai
  env: { OPENAI_API_KEY: "${{ secrets.OPENAI_API_KEY }}" }
  with:
    signature: "diff -> summary, risks: list[str]"
    inputs: '{"diff": "..."}'
- run: echo '${{ steps.ai.outputs.result }}'
```

Inputs: `signature`, `inputs` (JSON), `prompt` (used for `ask` mode when `signature` is empty), `model`, `load`, `python-version`, `version`.
The output `result` is also written to the job summary. A full PR-summary workflow is in [examples/github_workflow.yml](examples/github_workflow.yml).

## Examples

- [examples/quickstart.py](examples/quickstart.py): every one-liner
- [examples/notebook_walkthrough.py](examples/notebook_walkthrough.py): notebook-style task → optimize → save/load → evaluate
- [examples/server_custom.py](examples/server_custom.py): serve your own modules
- [examples/http_n8n_zapier.md](examples/http_n8n_zapier.md): no-code and HTTP integrations

All of them run offline with `DSPY_MODEL=dummy`.

## Development

```bash
pip install -e ".[server,dev]" httpx
pytest -q && ruff check . && ruff format --check .
```

Tests use an offline fake LM based on `dspy.utils.DummyLM`, so they need no API keys. CI runs on Python 3.10–3.12.

## Status

Built on DSPy 3.x (`dspy.LM`, `dspy.configure`, `Signature`, `Predict`, `ChainOfThought`, `ReAct`,
`BootstrapFewShot`, `MIPROv2`, `Module.save/load`). The wrapper's logic is fully tested offline. Live provider calls go
through DSPy/LiteLLM and haven't been exercised in this repo's CI yet, so please open an issue if you hit anything.

## Credit & license

MIT © DXv-3. All of the real work is done by **[DSPy](https://github.com/stanfordnlp/dspy)** (MIT, Stanford NLP):
please star and cite it. `dspy-dropin` is an independent convenience layer and is not affiliated with the DSPy team.
