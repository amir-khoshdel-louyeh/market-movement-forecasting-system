"""Simple token auth for API."""
import os
from functools import wraps
from flask import request, jsonify
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def get_api_token() -> str | None:
    tok = os.getenv("API_TOKEN", "").strip()
    return tok if tok else None

def get_read_token() -> str | None:
    tok = os.getenv("API_TOKEN_READ", "").strip() or os.getenv("API_TOKEN_READONLY", "").strip()
    return tok if tok else None

def is_write_request() -> bool:
    return request.method in ("POST", "PUT", "PATCH", "DELETE")

def _extract_provided() -> str:
    provided = request.headers.get("X-API-Token", "") or request.headers.get("X-API-TOKEN", "")
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        provided = auth[7:].strip() or provided
    if not provided:
        provided = request.args.get("token", "") or request.args.get("api_token", "")
    return provided

def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = get_api_token()
        read_tok = get_read_token()
        if not expected and not read_tok:
            return fn(*args, **kwargs)
        provided = _extract_provided()
        # write token allows everything
        if expected and provided == expected:
            return fn(*args, **kwargs)
        # read token only allows non-mutating
        if read_tok and provided == read_tok:
            if is_write_request():
                return jsonify({"ok": False, "error": "read-only token"}), 403
            return fn(*args, **kwargs)
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return wrapper

def require_write(fn):
    """Stricter: only write/admin token, read-only rejected."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = get_api_token()
        if not expected:
            return fn(*args, **kwargs)
        provided = _extract_provided()
        if provided != expected:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper
