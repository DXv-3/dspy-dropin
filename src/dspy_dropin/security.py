"""Small security helpers shared by the server, CLI and config: secret redaction, bearer-token auth, host checks."""

from __future__ import annotations

import hmac
import ipaddress
import os
import re

TOKEN_ENV = "DSPY_DROPIN_TOKEN"
_SECRET_ENV = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD)$|^DSPY_API_KEY$")
_SECRET_SHAPES = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{8,}|sk-ant-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})"
)


def secret_values(env: dict[str, str] | None = None) -> list[str]:
    env = dict(os.environ if env is None else env)
    vals = [v for k, v in env.items() if _SECRET_ENV.search(k) and v and len(v) >= 4]
    return sorted(set(vals), key=len, reverse=True)


def redact(text: object, env: dict[str, str] | None = None) -> str:
    """Remove every configured secret (and anything key-shaped) from ``text``."""
    s = str(text)
    for v in secret_values(env):
        s = s.replace(v, "***")
    s = _SECRET_SHAPES.sub("***", s)
    return re.sub(r"(?i)(authorization:?\s*(bearer|token)\s+)\S+", r"\1***", s)


def configured_token(env: dict[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    tok = (env.get(TOKEN_ENV) or "").strip()
    return tok or None


def check_bearer(header: str | None, token: str | None) -> bool:
    """True when no token is configured, or ``header`` is exactly ``Bearer <token>`` (constant-time compare)."""
    if token is None:
        return True
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:].strip().encode(), token.encode())


def is_loopback(host: str) -> bool:
    if host in ("localhost", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def exposure_warning(host: str, token: str | None) -> str | None:
    """Loud warning text when the server would be reachable from the network without auth."""
    if is_loopback(host) or token:
        return None
    bar = "!" * 78
    return (
        f"{bar}\n"
        f"WARNING: dspy-dropin is binding {host} WITHOUT authentication.\n"
        f"Anyone who can reach this port can spend your LLM credits and run arbitrary signatures.\n"
        f"Set {TOKEN_ENV}=<long random secret> to require 'Authorization: Bearer <token>', or bind 127.0.0.1.\n"
        f"{bar}"
    )
