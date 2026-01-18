from src.database import init_db, get_connection
import sqlite3

def test_init_creates_tables(tmp_db):
    with get_connection(tmp_db) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r[0] for r in cur.fetchall()}
        assert "models" in names
        assert "predictions" in names
        assert "model_performance" in names
