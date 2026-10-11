# Using dspy-dropin from n8n, Zapier, Make, or anything that speaks HTTP

Start the server (stdlib backend needs no extras; `pip install "dspy-dropin[server] @ git+https://github.com/DXv-3/dspy-dropin"` for FastAPI):

```bash
export OPENAI_API_KEY=...        # or DSPY_MODEL=dummy to try it offline
dspy-dropin serve --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{"ok": true, "model": "...", "modules": [...]}` |
| POST | `/ask` | `{"prompt": "...", "reasoning": false}` | `{"answer": "..."}` |
| POST | `/run` | `{"signature": "email -> intent, urgency: int", "inputs": {"email": "..."}}` | `{"intent": "...", "urgency": 3}` |
| POST | `/m/<name>` | input fields | output fields of a registered module (see `server_custom.py`) |

## curl

```bash
curl -s localhost:8000/run -H 'content-type: application/json' \
  -d '{"signature":"email -> intent, urgency: int","inputs":{"email":"Server is down!!"}}'
```

## n8n
Add an **HTTP Request** node: Method `POST`, URL `http://<host>:8000/run`, Body Content Type `JSON`,
JSON body as above (use expressions like `{{$json.body}}` for inputs). The response fields land on `$json`.

## Zapier / Make
Use **Webhooks by Zapier → Custom Request** (or Make's **HTTP → Make a request**) with the same
URL, method, header `Content-Type: application/json`, and body.
