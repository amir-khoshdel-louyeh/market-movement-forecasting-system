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

def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = get_api_token()
        if not expected:
            return fn(*args, **kwargs)
        provided = request.headers.get("X-API-Token", "") or request.headers.get("X-API-TOKEN", "")
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:].strip() or provided
        # also allow token via query for SSE convenience (read-only)
        if not provided:
            provided = request.args.get("token", "") or request.args.get("api_token", "")
        if provided != expected:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper
