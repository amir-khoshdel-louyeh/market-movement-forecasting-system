def test_index_and_ml_pages(tmp_db):
    from src.mmfs_web import app
    c = app.test_client()
    assert c.get("/").status_code == 200
    html = c.get("/").data.decode()
    assert "chart" in html.lower()
    assert "plotly" in html.lower()
    html_ml = c.get("/ml").data.decode()
    assert "ML Dashboard" in html_ml
    assert "prediction" in html_ml.lower()

def test_static_js_exists():
    from pathlib import Path
    p = Path("src/static/js/app.js")
    assert p.exists()
    txt = p.read_text()
    assert "Plotly.newPlot" in txt
    assert "EventSource" in txt

def test_openapi_and_rate_headers(tmp_db):
    from src.mmfs_web import app
    c = app.test_client()
    assert c.get("/api/openapi.json").status_code == 200
    assert c.get("/api/openapi.json").json["openapi"] == "3.0.0"
