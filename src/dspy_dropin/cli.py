"""``dspy-dropin`` command line."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

from . import __version__, config
from .security import TOKEN_ENV, redact


def _kv(pairs: list[str]) -> dict[str, Any]:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def cmd_doctor(a) -> int:
    import dspy

    print(f"dspy-dropin {__version__} | dspy {dspy.__version__} | python {sys.version.split()[0]}")
    # Only ever print whether a variable is set, never its value.
    for var in [
        "DSPY_MODEL",
        *[v for v, _ in config.PROVIDERS],
        "DSPY_API_KEY",
        "DSPY_API_BASE",
        "DSPY_CACHE",
        TOKEN_ENV,
    ]:
        print(f"  {var:<20} {'set' if os.environ.get(var) else '-'}")
    model = config.detect_model()
    print(f"  detected model: {redact(model) if model else 'NONE'}")
    if not model:
        print("FAIL: set DSPY_MODEL or a provider API key.")
        return 1
    try:
        config.configure()
        print("  configure(): ok")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL configure(): {redact(e)}")
        return 1
    if a.ping:
        from .oneliners import ask

        try:
            print(f"  ping: {redact(ask('Reply with the single word: pong'))!r}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL ping: {redact(e)}")
            return 1
    print("OK")
    return 0


def cmd_ask(a) -> int:
    from .oneliners import ask

    prompt = " ".join(a.prompt) if a.prompt else sys.stdin.read()
    print(ask(prompt, reasoning=a.reasoning))
    return 0


def cmd_run(a) -> int:
    from .oneliners import task
    from .server import _pred_to_dict

    inputs = _kv(a.input)
    if a.json:
        try:
            extra = json.loads(a.json)
        except ValueError as e:
            raise RuntimeError(f"--json is not valid JSON: {e}") from None
        if not isinstance(extra, dict):
            raise RuntimeError("--json must be a JSON object")
        inputs.update(extra)
    mod = task(a.signature, a.instructions, reasoning=a.reasoning)
    if a.load:
        mod.load(a.load)
    print(json.dumps(_pred_to_dict(mod(**inputs)), indent=2))
    return 0


def cmd_optimize(a) -> int:
    from .oneliners import task
    from .optimize import optimize

    with open(a.trainset) as f:
        rows = [json.loads(line) for line in f if line.strip()] if a.trainset.endswith(".jsonl") else json.load(f)
    metric = None
    if a.metric:
        mod_name, _, fn = a.metric.partition(":")
        metric = getattr(importlib.import_module(mod_name), fn)
    import dspy

    sig = dspy.ensure_signature(a.signature)
    inputs = list(sig.input_fields)
    if metric is None:
        from .optimize import exact_match

        metric = exact_match(next(iter(sig.output_fields)))
    module = task(a.signature, reasoning=a.reasoning)
    optimize(module, rows, metric, strategy=a.strategy, inputs=inputs, save_to=a.out)
    print(f"saved {a.out}")
    return 0


def cmd_serve(a) -> int:
    from .server import serve

    serve(a.host, a.port, backend=a.backend)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("dspy-dropin", description="Zero-setup DSPy from the shell.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--model", help="override DSPY_MODEL (e.g. openai/gpt-4o-mini, dummy)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check environment and LM")
    d.add_argument("--ping", action="store_true", help="make one live LM call")
    d.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("ask", help="ask a question (or read stdin)")
    s.add_argument("prompt", nargs="*")
    s.add_argument("--reasoning", action="store_true")
    s.set_defaults(fn=cmd_ask)

    r = sub.add_parser("run", help='run a signature, e.g. run "text -> summary" -i text=...')
    r.add_argument("signature")
    r.add_argument("-i", "--input", action="append", default=[], help="key=value (repeatable)")
    r.add_argument("--json", help="inputs as JSON object")
    r.add_argument("--instructions")
    r.add_argument("--reasoning", action="store_true")
    r.add_argument("--load", help="load optimized state JSON")
    r.set_defaults(fn=cmd_run)

    o = sub.add_parser("optimize", help="optimize a signature on a JSON/JSONL trainset")
    o.add_argument("signature")
    o.add_argument("trainset")
    o.add_argument("-o", "--out", default="optimized.json")
    o.add_argument("--strategy", default="bootstrap", choices=["bootstrap", "random", "mipro", "labeled"])
    o.add_argument("--metric", help="module:function (default exact match on first output)")
    o.add_argument("--reasoning", action="store_true")
    o.set_defaults(fn=cmd_optimize)

    v = sub.add_parser("serve", help="HTTP JSON server")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--backend", default="auto", choices=["auto", "fastapi", "stdlib"])
    v.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    if a.model:
        os.environ["DSPY_MODEL"] = a.model
    try:
        return a.fn(a)
    except RuntimeError as e:
        print(f"error: {redact(e)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
