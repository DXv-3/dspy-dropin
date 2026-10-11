"""Expose modules as JSON HTTP endpoints. Uses FastAPI if installed (``[server]`` extra), else stdlib.

Endpoints:
  GET  /health                          -> {"ok": true, "model": ..., "modules": [...]}
  POST /ask      {"prompt": "..."}       -> {"answer": "..."}
  POST /run      {"signature": "q -> a", "inputs": {...}, "reasoning": false} -> {outputs}
  POST /m/<name> {inputs}               -> outputs of a registered module

Security defaults:
  * binds 127.0.0.1 unless told otherwise; binding elsewhere without a token prints a loud warning
  * ``DSPY_DROPIN_TOKEN`` set  -> every endpoint except a minimal ``/health`` requires ``Authorization: Bearer <token>``
  * request bodies are capped (``DSPY_DROPIN_MAX_BODY`` bytes, default 1 MiB) and must be JSON objects
  * inputs are validated; internal errors return a generic message and are logged with secrets redacted
  * socket timeouts on the stdlib server; LM calls time out via ``DSPY_TIMEOUT`` (see config)
"""

import json
import logging
import os
import re
import signal
import sys
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import dspy

from . import config
from .oneliners import ask, task
from .security import check_bearer, configured_token, exposure_warning, redact

log = logging.getLogger("dspy_dropin.server")

DEFAULT_MAX_BODY = 1024 * 1024
MAX_PROMPT_CHARS = 100_000
MAX_SIGNATURE_CHARS = 1_000
MAX_INSTRUCTIONS_CHARS = 20_000
MAX_FIELDS = 64
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
POST_PATHS = ("/ask", "/run")
SEC_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store", "X-Frame-Options": "DENY"}


class BadRequest(ValueError):
    """Client error -> HTTP 400 with this message."""


def max_body() -> int:
    try:
        n = int(os.environ.get("DSPY_DROPIN_MAX_BODY", DEFAULT_MAX_BODY))
    except ValueError:
        return DEFAULT_MAX_BODY
    return n if n > 0 else DEFAULT_MAX_BODY


def _pred_to_dict(pred: Any) -> dict[str, Any]:
    if isinstance(pred, dspy.Prediction):
        out = dict(pred.toDict())
    elif isinstance(pred, Mapping):
        out = dict(pred)
    else:
        out = {"output": pred}
    return json.loads(json.dumps(out, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)))


def _str(body: Mapping[str, Any], key: str, limit: int, required: bool = True) -> str | None:
    if key not in body or body[key] is None:
        if required:
            raise KeyError(key)
        return None
    v = body[key]
    if not isinstance(v, str):
        raise BadRequest(f"{key!r} must be a string")
    if required and not v.strip():
        raise BadRequest(f"{key!r} must not be empty")
    if len(v) > limit:
        raise BadRequest(f"{key!r} exceeds {limit} characters")
    return v


def _bool(body: Mapping[str, Any], key: str) -> bool:
    v = body.get(key, False)
    if not isinstance(v, bool) and v is not None:
        raise BadRequest(f"{key!r} must be a boolean")
    return bool(v)


def _fields(v: Any, what: str) -> dict[str, Any]:
    if v is None:
        return {}
    if not isinstance(v, Mapping):
        raise BadRequest(f"{what} must be a JSON object")
    if len(v) > MAX_FIELDS:
        raise BadRequest(f"{what} has more than {MAX_FIELDS} fields")
    for k in v:
        if not isinstance(k, str) or not _IDENT.match(k):
            raise BadRequest(f"{what}: invalid field name {str(k)[:64]!r}")
    return dict(v)


def handle(
    path: str, body: Any, modules: Mapping[str, Any], method: str | None = None, authorized: bool = True
) -> tuple[int, dict[str, Any]]:
    """Framework-agnostic request handler (used by both backends and tests).

    ``method=None`` skips method checks (direct/programmatic use); the HTTP backends always pass it.
    """
    try:
        path = urlsplit(path).path
        if path == "/health":
            if method not in (None, "GET", "HEAD"):
                return 405, {"error": "method not allowed"}
            if not authorized:  # liveness only; don't reveal model/modules to unauthenticated callers
                return 200, {"ok": True}
            return 200, {"ok": True, "model": config._STATE.get("model"), "modules": sorted(modules)}
        known = path in POST_PATHS or path.startswith("/m/")
        if not known:
            return 404, {"error": "not found"}
        if method not in (None, "POST"):
            return 405, {"error": "method not allowed"}
        if not authorized:
            return 401, {"error": "unauthorized"}
        if not isinstance(body, Mapping):
            raise BadRequest("body must be a JSON object")
        if path == "/ask":
            prompt = _str(body, "prompt", MAX_PROMPT_CHARS)
            reasoning = _bool(body, "reasoning")
            config.ensure_configured()
            return 200, {"answer": ask(prompt, reasoning=reasoning)}
        if path == "/run":
            sig = _str(body, "signature", MAX_SIGNATURE_CHARS)
            if "->" not in sig:
                raise BadRequest("'signature' must look like 'inputs -> outputs'")
            instructions = _str(body, "instructions", MAX_INSTRUCTIONS_CHARS, required=False)
            inputs = _fields(body.get("inputs"), "'inputs'")
            reasoning = _bool(body, "reasoning")
            try:
                mod = task(sig, instructions, reasoning=reasoning)
            except Exception as e:  # noqa: BLE001 - signature parse errors are the client's fault
                raise BadRequest(f"invalid signature: {type(e).__name__}") from e
            config.ensure_configured()
            return 200, _pred_to_dict(mod(**inputs))
        name = path[3:]
        if not _NAME.match(name) or name not in modules:
            return 404, {"error": f"unknown module {name[:64]!r}"}
        inputs = _fields(body, "body")
        config.ensure_configured()
        return 200, _pred_to_dict(modules[name](**inputs))
    except KeyError as e:
        return 400, {"error": f"missing field {e}"}
    except BadRequest as e:
        return 400, {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        log.error("request to %s failed: %s", path[:100], redact(f"{type(e).__name__}: {e}"))
        return 500, {"error": "internal error"}


def make_fastapi_app(modules: Mapping[str, Any] | None = None, token: str | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    modules = dict(modules or {})
    token = token if token is not None else configured_token()
    cap = max_body()
    app = FastAPI(title="dspy-dropin", docs_url=None, redoc_url=None, openapi_url=None)

    def reply(code: int, data: dict[str, Any], extra: dict[str, str] | None = None):
        return JSONResponse(data, status_code=code, headers={**SEC_HEADERS, **(extra or {})})

    async def read_capped(request: Request) -> bytes | None:
        cl = request.headers.get("content-length")
        if cl is not None and (not cl.isdigit() or int(cl) > cap):
            return None
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > cap:
                return None
        return bytes(buf)

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def route(path: str, request: Request):
        authorized = check_bearer(request.headers.get("authorization"), token)
        body: Any = {}
        if request.method == "POST":
            if not authorized and "/" + path != "/health":
                return reply(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
            raw = await read_capped(request)
            if raw is None:
                return reply(413, {"error": f"body exceeds {cap} bytes"}, {"Connection": "close"})
            if raw.strip():
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    return reply(400, {"error": "invalid JSON"})
        code, data = await run_in_threadpool(handle, "/" + path, body, modules, request.method, authorized)
        return reply(code, data, {"WWW-Authenticate": "Bearer"} if code == 401 else None)

    return app


def make_stdlib_server(
    host: str, port: int, modules: Mapping[str, Any] | None = None, token: str | None = None
) -> ThreadingHTTPServer:
    modules = dict(modules or {})
    token = token if token is not None else configured_token()
    cap = max_body()

    class H(BaseHTTPRequestHandler):
        timeout = 30  # socket timeout: slow/idle clients can't pin a thread forever
        server_version = "dspy-dropin"
        sys_version = ""

        def _send(self, code: int, data: dict[str, Any], extra: dict[str, str] | None = None) -> None:
            raw = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in {**SEC_HEADERS, **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(raw)

        def _authorized(self) -> bool:
            return check_bearer(self.headers.get("Authorization"), token)

        def _dispatch(self, body: Any = None) -> None:
            code, data = handle(self.path, {} if body is None else body, modules, self.command, self._authorized())
            self._send(code, data, {"WWW-Authenticate": "Bearer"} if code == 401 else None)

        def do_GET(self):  # noqa: N802
            self._dispatch()

        do_HEAD = do_GET  # noqa: N815

        def do_POST(self):  # noqa: N802
            if urlsplit(self.path).path != "/health" and not self._authorized():
                self.close_connection = True
                return self._send(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                self.close_connection = True
                return self._send(411, {"error": "Content-Length required"})
            cl = self.headers.get("Content-Length") or "0"
            if not cl.isdigit():
                self.close_connection = True
                return self._send(400, {"error": "invalid Content-Length"})
            n = int(cl)
            if n > cap:
                self.close_connection = True
                return self._send(413, {"error": f"body exceeds {cap} bytes"}, {"Connection": "close"})
            raw = self.rfile.read(n) if n else b""
            if raw.strip():
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    return self._send(400, {"error": "invalid JSON"})
            else:
                body = {}
            self._dispatch(body)

        def _not_allowed(self):
            self._dispatch()

        do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _not_allowed  # noqa: N815

        def log_message(self, fmt, *a):  # quiet by default; never log headers (Authorization)
            log.debug("%s - %s", self.address_string(), fmt % a)

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    return srv


def serve(
    host: str = "127.0.0.1", port: int = 8000, modules: Mapping[str, Any] | None = None, backend: str = "auto"
) -> None:
    """Run the server (blocking). Binds 127.0.0.1 by default."""
    token = configured_token()
    warning = exposure_warning(host, token)
    if warning:
        print(warning, file=sys.stderr, flush=True)
        log.warning("binding %s without %s", host, "DSPY_DROPIN_TOKEN")
    if token:
        print("dspy-dropin: bearer-token auth ENABLED (DSPY_DROPIN_TOKEN)", file=sys.stderr)
    if backend in ("auto", "fastapi"):
        try:
            import uvicorn

            uvicorn.run(
                make_fastapi_app(modules, token=token),
                host=host,
                port=port,
                timeout_keep_alive=5,
                timeout_graceful_shutdown=10,
                server_header=False,
            )
            return
        except ImportError:
            if backend == "fastapi":
                raise
    srv = make_stdlib_server(host, port, modules, token=token)

    def _stop(signum, _frame):
        threading.Thread(target=srv.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:  # not in main thread
        pass
    print(f"dspy-dropin serving on http://{host}:{port} (stdlib)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
