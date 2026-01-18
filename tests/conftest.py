import pytest
import tempfile
from pathlib import Path
from src.database import init_db

@pytest.fixture
def tmp_db(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    # monkeypatch get_db_path for registry/logger/tracker
    import src.database as dbmod
    import src.model_registry as regmod
    import src.prediction_logger as logmod
    import src.performance_tracker as perfmod
    orig = dbmod.get_db_path
    dbmod.get_db_path = lambda: db
    regmod.get_db_path = lambda: db  # type: ignore
    # these modules import get_db_path directly, so patch their references
    import src.model_registry
    import src.prediction_logger
    import src.performance_tracker
    orig_reg = src.model_registry.get_db_path
    orig_log = src.prediction_logger.get_db_path
    orig_perf = src.performance_tracker.get_db_path
    src.model_registry.get_db_path = lambda: db
    src.prediction_logger.get_db_path = lambda: db
    src.performance_tracker.get_db_path = lambda: db
    # also patch auth token empty for tests
    import os
    os.environ.pop("API_TOKEN", None)
    yield db
    dbmod.get_db_path = orig
    src.model_registry.get_db_path = orig_reg
    src.prediction_logger.get_db_path = orig_log
    src.performance_tracker.get_db_path = orig_perf
