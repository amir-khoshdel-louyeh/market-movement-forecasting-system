import numpy as np
from src.backtest import label_next, backtest_model

def test_label_next():
    assert label_next(100, 100.3, 0.2) == "up"
    assert label_next(100, 99.7, 0.2) == "down"
    assert label_next(100, 100.05, 0.2) == "neutral"
    assert label_next(100, 100, 0.2) == "neutral"

def test_backtest_model(tmp_db):
    from src.models.baseline import MomentumModel
    rng = np.random.default_rng(0)
    closes = 50000 + np.cumsum(rng.standard_normal(60))
    arr = np.zeros((60,5))
    arr[:,3]=closes
    arr[:,4]=1e6
    m = MomentumModel()
    res = backtest_model(m, arr, threshold=0.2, window_size=10, step=2)
    assert "total" in res and "accuracy" in res
    assert 0 <= res["accuracy"] <= 1
    assert res["total"] > 0
