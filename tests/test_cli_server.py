import json
import threading
import urllib.request

from dspy_dropin import cli
from dspy_dropin.server import handle, make_stdlib_server


def test_doctor_no_model(capsys):
    assert cli.main(["doctor"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_doctor_dummy(capsys):
    assert cli.main(["--model", "dummy", "doctor", "--ping"]) == 0
    assert "OK" in capsys.readouterr().out


def test_ask_cli(capsys, monkeypatch):
    monkeypatch.setenv("DSPY_DUMMY_ANSWER", "pong")
    assert cli.main(["--model", "dummy", "ask", "ping?"]) == 0
    assert capsys.readouterr().out.strip() == "pong"


def test_run_cli(capsys):
    assert cli.main(["--model", "dummy", "run", "text -> summary", "-i", "text=hello"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"] == "42"


def test_optimize_cli(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps({"question": f"q{i}", "answer": "42"}) for i in range(3)))
    out = tmp_path / "o.json"
    assert cli.main(["--model", "dummy", "optimize", "question -> answer", str(data), "-o", str(out)]) == 0
    assert out.exists()
    assert cli.main(["--model", "dummy", "run", "question -> answer", "-i", "question=x", "--load", str(out)]) == 0


def test_handle(dummy):
    dummy(answer="ok")
    assert handle("/health", {}, {})[1]["ok"]
    assert handle("/ask", {"prompt": "hi"}, {}) == (200, {"answer": "ok"})
    code, data = handle("/run", {"signature": "a -> b", "inputs": {"a": "x"}}, {})
    assert code == 200 and "b" in data
    assert handle("/ask", {}, {})[0] == 400
    assert handle("/m/nope", {}, {})[0] == 404


def test_stdlib_server(dummy):
    dummy(answer="srv")
    srv = make_stdlib_server("127.0.0.1", 0, {})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/ask"
        req = urllib.request.Request(
            url, data=json.dumps({"prompt": "hi"}).encode(), headers={"Content-Type": "application/json"}
        )
        assert json.loads(urllib.request.urlopen(req).read()) == {"answer": "srv"}
    finally:
        srv.shutdown()


def test_fastapi_app(dummy):
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dspy_dropin import task
    from dspy_dropin.server import make_fastapi_app

    dummy(answer="fa")
    c = TestClient(make_fastapi_app({"qa": task("question -> answer")}))
    assert c.get("/health").json()["modules"] == ["qa"]
    r = c.post("/m/qa", json={"question": "x"})
    assert r.json()["answer"] == "fa", r.text
