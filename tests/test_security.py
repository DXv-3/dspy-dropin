"""Security regression tests: auth, bind defaults, body caps, validation, no secret leakage, fake-LM guards."""

import http.client
import json
import threading
import warnings

import pytest

from dspy_dropin import cli, config, security, testing
from dspy_dropin.server import handle, make_stdlib_server, serve

TOKEN = "s3cret-token-for-tests-0123456789"


@pytest.fixture
def stdlib_srv(dummy, monkeypatch):
    servers = []

    def _start(token=None, max_body=None, modules=None):
        if max_body:
            monkeypatch.setenv("DSPY_DROPIN_MAX_BODY", str(max_body))
        dummy(answer="ok")
        srv = make_stdlib_server("127.0.0.1", 0, modules or {}, token=token)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv.server_address[1]

    yield _start
    for s in servers:
        s.shutdown()
        s.server_close()


def req(port, method, path, body=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers=headers or {})
    r = c.getresponse()
    out = r.status, dict(r.getheaders()), r.read()
    c.close()
    return out


# ---------- auth ----------
def test_token_enforced_when_set(stdlib_srv):
    port = stdlib_srv(token=TOKEN)
    assert req(port, "POST", "/ask", {"prompt": "hi"})[0] == 401
    assert req(port, "POST", "/ask", {"prompt": "hi"}, {"Authorization": "Bearer wrong"})[0] == 401
    assert req(port, "POST", "/ask", {"prompt": "hi"}, {"Authorization": TOKEN})[0] == 401
    assert req(port, "POST", "/ask", {"prompt": "hi"}, {"Authorization": f"Basic {TOKEN}"})[0] == 401
    st, hd, body = req(port, "POST", "/ask", {"prompt": "hi"}, {"Authorization": f"Bearer {TOKEN}"})
    assert st == 200 and json.loads(body) == {"answer": "ok"}
    assert req(port, "POST", "/m/x", {}, {})[0] == 401
    # health is liveness-only without the token
    st, _, body = req(port, "GET", "/health")
    assert st == 200 and json.loads(body) == {"ok": True}
    st, _, body = req(port, "GET", "/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert "modules" in json.loads(body)


def test_token_from_env(monkeypatch, dummy):
    monkeypatch.setenv("DSPY_DROPIN_TOKEN", TOKEN)
    dummy(answer="ok")
    srv = make_stdlib_server("127.0.0.1", 0, {})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert req(srv.server_address[1], "POST", "/ask", {"prompt": "x"})[0] == 401
    finally:
        srv.shutdown()
        srv.server_close()


def test_check_bearer_is_strict():
    assert security.check_bearer(None, None)
    assert not security.check_bearer(None, TOKEN)
    assert not security.check_bearer("Bearer ", TOKEN)
    assert not security.check_bearer(f"Bearer {TOKEN}x", TOKEN)
    assert security.check_bearer(f"Bearer {TOKEN}", TOKEN)
    assert security.configured_token({"DSPY_DROPIN_TOKEN": "  "}) is None


def test_fastapi_auth_and_caps(dummy, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dspy_dropin.server import make_fastapi_app

    monkeypatch.setenv("DSPY_DROPIN_MAX_BODY", "200")
    dummy(answer="fa")
    c = TestClient(make_fastapi_app({}, token=TOKEN))
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert c.post("/ask", json={"prompt": "x"}).status_code == 401
    assert c.post("/ask", json={"prompt": "x"}, headers=auth).json() == {"answer": "fa"}
    assert c.post("/ask", content=b"{bad", headers={**auth, "content-type": "application/json"}).status_code == 400
    assert c.post("/ask", json={"prompt": "x" * 500}, headers=auth).status_code == 413
    assert c.post("/ask", json=["x"], headers=auth).status_code == 400
    assert c.get("/ask", headers=auth).status_code == 405
    assert c.delete("/run", headers=auth).status_code == 405
    assert c.get("/docs").status_code == 404
    assert c.get("/health").json() == {"ok": True}
    r = c.post("/ask", json={"prompt": "x"}, headers=auth)
    assert r.headers["x-content-type-options"] == "nosniff"


# ---------- bind defaults / warnings ----------
def test_default_bind_is_localhost():
    import inspect

    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"
    import argparse

    seen = {}
    orig = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        ns = orig(self, *a, **k)
        seen.update(vars(ns))
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        with pytest.raises(SystemExit):
            cli.main(["serve"])
    finally:
        argparse.ArgumentParser.parse_args = orig
    assert seen["host"] == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com"])
def test_exposure_warning_without_token(host):
    w = security.exposure_warning(host, None)
    assert w and "WITHOUT authentication" in w and "DSPY_DROPIN_TOKEN" in w
    assert security.exposure_warning(host, TOKEN) is None


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_no_warning_on_loopback(host):
    assert security.exposure_warning(host, None) is None


def test_serve_prints_loud_warning(monkeypatch, capsys):
    import dspy_dropin.server as srvmod

    class Fake:
        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(srvmod, "make_stdlib_server", lambda *a, **k: Fake())
    serve("0.0.0.0", 0, backend="stdlib")
    assert "WITHOUT authentication" in capsys.readouterr().err
    monkeypatch.setenv("DSPY_DROPIN_TOKEN", TOKEN)
    serve("0.0.0.0", 0, backend="stdlib")
    err = capsys.readouterr().err
    assert "WITHOUT" not in err and "ENABLED" in err and TOKEN not in err


# ---------- body caps / malformed requests ----------
def test_body_size_cap_and_bad_requests(stdlib_srv):
    port = stdlib_srv(max_body=1000)
    assert req(port, "POST", "/ask", {"prompt": "x" * 2000})[0] == 413
    st, _, _ = req(port, "POST", "/ask", raw=b"{}", headers={"Content-Length": "99999999"})
    assert st == 413
    assert req(port, "POST", "/ask", raw=b"{}", headers={"Content-Length": "-1"})[0] == 400
    assert req(port, "POST", "/ask", raw=b"{nope")[0] == 400
    assert req(port, "POST", "/ask", raw=b"\xff\xfe")[0] == 400
    assert req(port, "POST", "/ask", raw=b"[1,2]")[0] == 400
    assert req(port, "POST", "/ask", raw=b'"str"')[0] == 400
    assert req(port, "GET", "/ask")[0] == 405
    assert req(port, "PUT", "/run", {})[0] == 405
    assert req(port, "DELETE", "/m/x")[0] == 405
    assert req(port, "GET", "/../../etc/passwd")[0] == 404
    assert req(port, "GET", "/health?x=1")[0] == 200
    st, hd, _ = req(port, "GET", "/health")
    assert hd.get("X-Content-Type-Options") == "nosniff" and "Python" not in hd.get("Server", "")


@pytest.mark.parametrize(
    "path,body,code",
    [
        ("/ask", {"prompt": 123}, 400),
        ("/ask", {"prompt": ""}, 400),
        ("/ask", {"prompt": "x" * 100_001}, 400),
        ("/ask", {"prompt": "hi", "reasoning": "yes"}, 400),
        ("/ask", {}, 400),
        ("/run", {"signature": "no arrow here"}, 400),
        ("/run", {"signature": ["a -> b"]}, 400),
        ("/run", {"signature": "a -> b" + " " * 2000}, 400),
        ("/run", {"signature": "a -> b", "inputs": "x"}, 400),
        ("/run", {"signature": "a -> b", "inputs": {"__class__": 1}}, 200),
        ("/run", {"signature": "a -> b", "inputs": {"bad key": 1}}, 400),
        ("/run", {"signature": "a -> b", "inputs": {f"k{i}": 1 for i in range(100)}}, 400),
        ("/run", {"signature": "a: NotAType[[ -> b"}, 400),
        ("/run", {"signature": "a -> b", "instructions": 5}, 400),
        ("/m/../../x", {}, 404),
        ("/nope", {}, 404),
    ],
)
def test_input_validation(dummy, path, body, code):
    dummy(answer="ok")
    got, data = handle(path, body, {}, "POST")
    assert got == code, data
    if code == 400:
        assert "error" in data and "Traceback" not in json.dumps(data)


def test_internal_errors_are_generic_and_redacted(dummy, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-SUPERSECRET-123456")
    dummy(answer="ok")

    def boom(**_):
        raise RuntimeError("provider said: invalid key sk-live-SUPERSECRET-123456 at /srv/app.py:12")

    with caplog.at_level("ERROR"):
        code, data = handle("/m/boom", {}, {"boom": boom}, "POST")
    assert code == 500 and data == {"error": "internal error"}
    assert "SUPERSECRET" not in caplog.text and "***" in caplog.text


# ---------- no key leakage ----------
def test_doctor_never_prints_secrets(capsys, monkeypatch):
    secret = "sk-test-DONOTPRINT-abcdef123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("DSPY_API_KEY", "another-secret-value-xyz")
    monkeypatch.setenv("DSPY_DROPIN_TOKEN", TOKEN)
    monkeypatch.setenv("DSPY_MODEL", "dummy")
    assert cli.main(["doctor", "--ping"]) == 0
    out = capsys.readouterr()
    for s in (secret, "another-secret-value-xyz", TOKEN):
        assert s not in out.out + out.err
    assert "DSPY_DROPIN_TOKEN" in out.out


def test_doctor_redacts_configure_errors(capsys, monkeypatch):
    secret = "sk-test-LEAKY-abcdef123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def bad(*a, **k):
        raise ValueError(f"auth failed for {secret}")

    monkeypatch.setattr(config, "configure", bad)
    assert cli.main(["doctor"]) == 1
    assert secret not in capsys.readouterr().out


def test_redact_shapes():
    s = security.redact("k=sk-ant-abcdefghijkl gsk_abcdefghijkl ghp_" + "a" * 36 + " Authorization: Bearer xyz", env={})
    assert "abcdefghijkl" not in s and "ghp_" not in s and "xyz" not in s


def test_bad_numeric_env_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("DSPY_TIMEOUT", "soon")
    with pytest.raises(RuntimeError, match="DSPY_TIMEOUT"):
        config._lm_kwargs("openai/gpt-4o-mini")
    monkeypatch.delenv("DSPY_TIMEOUT")
    assert config._lm_kwargs("openai/gpt-4o-mini")["timeout"] == 60.0


# ---------- fake LM robustness ----------
def test_dummy_hook_supported_on_pinned_dspy():
    assert testing.hook_supported(), "DSPy DummyLM internals changed: update testing.EchoLM"


def test_dummy_fills_arbitrary_fields(dummy):
    from dspy_dropin import task

    dummy(answer="z")
    out = task("question -> zebra, giraffe")(question="q")
    assert out.zebra == "z" and out.giraffe == "z"


def test_dummy_survives_prompt_format_change(monkeypatch):
    lm = testing.make_dummy_lm("v")
    monkeypatch.setattr(testing, "_OUT_BLOCK", None)  # simulate DSPy changing the prompt layout
    txt = lm._use_example([{"role": "system", "content": "whatever"}])
    assert "answer" in txt and "v" in txt


def test_dummy_falls_back_without_formatter(monkeypatch):
    monkeypatch.delattr(testing.DummyLM, "_format_answer_fields")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lm = testing.EchoLM("v")
    assert any("DummyLM internals" in str(x.message) for x in w)
    out = lm._use_example([{"role": "system", "content": ""}])
    assert "[[ ## answer ## ]]\nv" in out and "[[ ## completed ## ]]" in out


def test_cli_run_json_must_be_object(capsys):
    assert cli.main(["--model", "dummy", "run", "a -> b", "--json", "[1]"]) == 2
    assert cli.main(["--model", "dummy", "run", "a -> b", "--json", "{bad"]) == 2
