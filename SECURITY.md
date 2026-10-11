# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report privately through GitHub's
**[private vulnerability reporting](https://github.com/DXv-3/dspy-dropin/security/advisories/new)** (Security tab →
"Report a vulnerability") with the affected version/commit, reproduction steps and impact. Expect an acknowledgement
within 7 days; confirmed issues are fixed before new features.

## Supported versions

Only the latest `main` (installed via `git+https://github.com/DXv-3/dspy-dropin`) is supported.

## Scope

In scope:

- `dspy_dropin.server` (stdlib and FastAPI backends): auth bypass of `DSPY_DROPIN_TOKEN`, body-size or validation
  bypass, information leaks (API keys, tokens, stack traces), request smuggling/DoS within the documented limits.
- CLI (`dspy-dropin doctor` etc.) and config: any path that prints or logs secrets.
- The composite GitHub Action (`action.yml`) and workflows: script/expression injection, excessive permissions.

Out of scope:

- DSPy, LiteLLM, FastAPI/uvicorn and model providers themselves (report upstream), and what a model chooses to output.
- Running the server on a non-loopback address **without** `DSPY_DROPIN_TOKEN` (it warns loudly; that is a deployment
  choice), or behind a proxy that strips/forwards credentials incorrectly.
- Prompt injection against your own signatures/modules.

## Secure defaults

- Server binds `127.0.0.1`; set `DSPY_DROPIN_TOKEN` (long random value) before binding anything else.
- `DSPY_DROPIN_MAX_BODY` caps request bodies (default 1 MiB); `DSPY_TIMEOUT` bounds each LM call (default 60 s).
- `doctor` prints only whether secrets are set, never their values; errors are redacted before printing/logging.
