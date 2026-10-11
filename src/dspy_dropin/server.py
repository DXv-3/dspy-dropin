"""Expose modules as JSON HTTP endpoints. Uses FastAPI if installed (``[server]`` extra), else stdlib.

Endpoints:
  GET  /health                          -> {"ok": true, "model": ...}
  POST /ask      {"prompt": "..."}       -> {"answer": "..."}
  POST /run      {"signature": "q -> a", "inputs": {...}, "reasoning": false} -> {outputs}
  POST /m/<name> {inputs}               -> outputs of a registered module
"""

import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import dspy

from . import config
from .oneliners import ask, task


def _pred_to_dict(pred: Any) -> dict[str, Any]:
    if isinstance(pred, dspy.Prediction):
        out = dict(pred.toDict())
    elif isinstance(pred, Mapping):
        out = dict(pred)
    else:
        out = {"output": pred}
    return json.loads(json.dumps(out, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)))


def handle(path: str, body: dict[str, Any], modules: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Framework-agnostic request handler (used by both backends and tests)."""
    try:
        if path == "/health":
            return 200, {"ok": True, "model": config._STATE.get("model"), "modules": sorted(modules)}
        config.ensure_configured()
        if path == "/ask":
            return 200, {"answer": ask(body["prompt"], reasoning=bool(body.get("reasoning")))}
        if path == "/run":
            mod = task(body["signature"], body.get("instructions"), reasoning=bool(body.get("reasoning")))
            return 200, _pred_to_dict(mod(**body.get("inputs", {})))
        if path.startswith("/m/"):
            name = path[3:]
            if name not in modules:
                return 404, {"error": f"unknown module {name!r}"}
            return 200, _pred_to_dict(modules[name](**body))
        return 404, {"error": "not found"}
    except KeyError as e:
        return 400, {"error": f"missing field {e}"}
    except Exception as e:  # noqa: BLE001
        return 500, {"error": f"{type(e).__name__}: {e}"}


def make_fastapi_app(modules: Mapping[str, Any] | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    modules = dict(modules or {})
    app = FastAPI(title="dspy-dropin")

    @app.get("/health")
    def health():
        return handle("/health", {}, modules)[1]

    @app.post("/{path:path}")
    async def post(path: str, request: Request):
        body = await request.json() if await request.body() else {}
        code, data = await run_in_threadpool(handle, "/" + path, body, modules)
        return JSONResponse(data, status_code=code)

    return app


def make_stdlib_server(host: str, port: int, modules: Mapping[str, Any] | None = None) -> ThreadingHTTPServer:
    modules = dict(modules or {})

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, data: dict[str, Any]) -> None:
            raw = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            self._send(*handle(self.path, {}, modules))

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            self._send(*handle(self.path, body, modules))

        def log_message(self, *a):  # quiet
            pass

    return ThreadingHTTPServer((host, port), H)


def serve(
    host: str = "127.0.0.1", port: int = 8000, modules: Mapping[str, Any] | None = None, backend: str = "auto"
) -> None:
    """Run the server (blocking)."""
    if backend in ("auto", "fastapi"):
        try:
            import uvicorn

            uvicorn.run(make_fastapi_app(modules), host=host, port=port)
            return
        except ImportError:
            if backend == "fastapi":
                raise
    srv = make_stdlib_server(host, port, modules)
    print(f"dspy-dropin serving on http://{host}:{port} (stdlib)")
    srv.serve_forever()
