import numpy as np
from src.models.baseline import MovingAverageCrossoverModel, MomentumModel, RandomModel, VolumeWeightedModel
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel

def _candles(n=40, seed=0):
    rng = np.random.default_rng(seed)
    closes = 50000 + np.cumsum(rng.standard_normal(n)*10)
    arr = np.zeros((n,5))
    arr[:,0]=closes
    arr[:,1]=closes+2
    arr[:,2]=closes-2
    arr[:,3]=closes
    arr[:,4]=1e6
    return arr

def test_baseline_models_predict(tmp_db):
    c = _candles()
    for cls in [MovingAverageCrossoverModel, MomentumModel, VolumeWeightedModel, RandomModel]:
        m = cls()
        feats = m.prepare_features(c)
        pred, conf = m.predict(feats)
        assert pred in ("up","down","neutral")
        assert 0 <= conf <= 1

def test_lstm_prepare_and_predict(tmp_db):
    c = _candles(35)
    m = LSTMModel(seq_len=10, hidden_size=8)
    feats = m.prepare_features(c)
    assert "sequence" in feats
    assert feats["sequence"].shape == (10,5)
    pred, conf = m.predict(feats)
    assert pred in ("up","down","neutral")

def test_transformer_prepare_and_predict(tmp_db):
    c = _candles(35)
    m = TransformerModel(seq_len=10, d_model=8, nhead=2)
    feats = m.prepare_features(c)
    assert "sequence" in feats
    pred, conf = m.predict(feats)
    assert pred in ("up","down","neutral")

def test_lstm_fit_no_torch(tmp_db):
    c = _candles(50)
    m = LSTMModel(seq_len=10)
    res = m.fit(c, epochs=1)
    # either ok or torch missing, both acceptable but should return dict
    assert isinstance(res, dict)
    assert "ok" in res
