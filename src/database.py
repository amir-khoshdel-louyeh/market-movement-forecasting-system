"""Database module for storing models, predictions, and performance metrics."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "mmfs.db"


def get_db_path() -> Path:
    """Return the database file path, creating parent directories if needed."""
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for database connections."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None):
    """Initialize database with tables and indexes."""
    if db_path is None:
        db_path = get_db_path()
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Create models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                version TEXT NOT NULL,
                file_path TEXT,
                hyperparameters TEXT,
                trained_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                features TEXT,
                actual_result TEXT,
                result_timestamp TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        
        # Create model_performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                market_condition TEXT,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_model_id ON model_performance(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_symbol ON model_performance(symbol)")
        
        print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("Database setup complete.")
