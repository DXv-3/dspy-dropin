import dspy

import dspy_dropin as d
from dspy_dropin.optimize import evaluate, exact_match, to_examples

TRAIN = [{"question": f"q{i}", "answer": "42"} for i in range(4)]


def test_to_examples():
    ex = to_examples(TRAIN)
    assert set(ex[0].inputs().keys()) == {"question"}


def test_optimize_bootstrap_save_load(dummy, tmp_path):
    dummy(answer="42")
    mod = d.task("question -> answer")
    out = tmp_path / "opt.json"
    compiled = d.optimize(mod, TRAIN, exact_match(), strategy="bootstrap", save_to=out)
    assert out.exists()
    demos = compiled.inner.demos
    assert len(demos) >= 1
    fresh = d.load(d.task("question -> answer"), out)
    assert len(fresh.inner.demos) == len(demos)
    assert evaluate(fresh, TRAIN) == 100.0


def test_optimize_labeled(dummy):
    dummy(answer="42")
    compiled = d.optimize(d.task("question -> answer"), TRAIN, strategy="labeled", k=2)
    assert len(compiled.inner.demos) == 2


def test_bad_strategy(dummy):
    dummy()
    import pytest

    with pytest.raises(ValueError):
        d.optimize(d.task("q -> a"), TRAIN, strategy="nope")


def test_mipro_constructs():
    # Not run (needs many LM calls); verify the optimizer exists in this DSPy version.
    assert hasattr(dspy, "MIPROv2")
