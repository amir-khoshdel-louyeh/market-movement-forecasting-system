from src.model_registry import ModelRegistry
from src.prediction_logger import PredictionLogger
from src.models.baseline import MovingAverageCrossoverModel

def test_logger_roundtrip(tmp_db):
    reg = ModelRegistry(db_path=tmp_db)
    log = PredictionLogger(db_path=tmp_db)
    m = MovingAverageCrossoverModel()
    mid = reg.register_model(name=m.name, model_type=m.model_type, version=m.version, model_obj=m, hyperparameters=m.get_hyperparameters())
    pid = log.log_prediction(model_id=mid, symbol="btcusdt", interval="1m", prediction="up", confidence=0.6, features={"a":1})
    pending = log.get_pending_predictions(symbol="btcusdt")
    assert any(p["id"]==pid for p in pending)
    log.update_actual_result(pid, "up")
    pending2 = log.get_pending_predictions(symbol="btcusdt")
    assert all(p["id"]!=pid for p in pending2)
    acc = log.calculate_accuracy(mid)
    assert acc["total_predictions"]==1
    assert acc["accuracy"]==1.0
