# %% [markdown]
# # dspy-dropin walkthrough (notebook-style; open in VS Code / Jupytext, or run as a script)
# Offline by default: uses the dummy LM unless you set DSPY_MODEL / OPENAI_API_KEY.

# %%
import os

os.environ.setdefault("DSPY_MODEL", "dummy")
import dspy_dropin as dd

dd.configure()

# %% Build a module from a signature
qa = dd.task("question -> answer")
print(qa("What is 6 * 7?").answer)

# %% Optimize it with a few labeled examples, save, reload
train = [{"question": f"What is 6 * 7? (v{i})", "answer": "42"} for i in range(6)]
compiled = dd.optimize(qa, train, strategy="bootstrap", save_to="qa_optimized.json")
fresh = dd.load(dd.task("question -> answer"), "qa_optimized.json")
print("demos:", len(fresh.inner.demos))

# %% Evaluate
from dspy_dropin.optimize import evaluate  # noqa: E402

print("score:", evaluate(fresh, train))

# %% Switch strategies with a real LM:
# compiled = dd.optimize(qa, train, strategy="mipro")   # MIPROv2(auto="light")
