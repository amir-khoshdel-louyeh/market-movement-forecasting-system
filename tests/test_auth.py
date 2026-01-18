import os
from unittest.mock import patch

def test_auth_allows_when_no_token(tmp_db):
    from src.mmfs_web import app
    os.environ.pop("API_TOKEN", None)
    client=app.test_client()
    # /start is protected but should pass when no token
    r=client.post("/start", json={"symbol":"btcusdt","interval":"1m"})
    # may fail due to network but not 401
    assert r.status_code != 401

def test_auth_blocks_when_token_set(tmp_db):
    from src.mmfs_web import app
    os.environ["API_TOKEN"]="secret123"
    client=app.test_client()
    r=client.post("/api/predict", json={})
    assert r.status_code == 401
    r2=client.post("/api/predict", json={}, headers={"X-API-Token":"secret123"})
    # will be 400 due to empty df, but not 401
    assert r2.status_code != 401
    os.environ.pop("API_TOKEN", None)
