import pandas as pd, numpy as np
from unittest.mock import patch

def test_health_and_candles(tmp_db):
    from src.mmfs_web import app, state
    # seed df
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"Open":[1]*5,"High":[2]*5,"Low":[0.5]*5,"Close":[1.5]*5,"Volume":[1e6]*5}, index=dates)
    df.index.name="Date"
    with state.lock:
        state.df = df
        state.symbol="btcusdt"
        state.interval="1m"
    client = app.test_client()
    assert client.get("/api/candles").status_code == 200
    assert client.get("/api/models").status_code == 200

def test_predict_requires_data(tmp_db):
    from src.mmfs_web import app, state
    from src.model_registry import ModelRegistry
    from src.models.baseline import MovingAverageCrossoverModel
    # empty df should 400
    with state.lock:
        import pandas as pd
        state.df = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    client = app.test_client()
    reg = ModelRegistry(db_path=tmp_db)
    if not reg.list_models():
        m=MovingAverageCrossoverModel()
        reg.register_model(name=m.name, model_type=m.model_type, version=m.version, model_obj=m, hyperparameters=m.get_hyperparameters())
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 400

def test_train_and_backtest_mocked(tmp_db):
    from src.mmfs_web import app
    from src.model_registry import ModelRegistry
    from src.models.baseline import MovingAverageCrossoverModel
    import pandas as pd, numpy as np
    reg = ModelRegistry(db_path=tmp_db)
    if not reg.list_models():
        m=MovingAverageCrossoverModel()
        reg.register_model(name=m.name, model_type=m.model_type, version=m.version, model_obj=m, hyperparameters=m.get_hyperparameters())
    dates = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    df = pd.DataFrame({"Open":50000+np.random.randn(100),"High":50001,"Low":49999,"Close":50000+np.cumsum(np.random.randn(100)),"Volume":1e6}, index=dates)
    df.index.name="Date"
    with patch("src.mmfs_web._get_historical_data", return_value=df):
        client=app.test_client()
        r=client.post("/api/train/models", json={"symbol":"btcusdt","interval":"1m"})
        assert r.status_code==200
        assert r.get_json()["ok"] is True
    with patch("src.mmfs_web._get_historical_data", return_value=df):
        client=app.test_client()
        r=client.post("/api/backtest", json={"symbol":"btcusdt","interval":"1m","days":5})
        assert r.status_code==200
        assert r.get_json()["ok"] is True
