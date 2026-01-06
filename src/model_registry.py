"""Model registry for saving, loading, and managing trained models."""
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from .database import get_connection, get_db_path


DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


def get_models_dir() -> Path:
    """Return the models directory, creating it if needed."""
    models_dir = DEFAULT_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


class ModelRegistry:
    """Registry for managing trained models and their metadata."""
    
    def __init__(self, db_path: Optional[Path] = None, models_dir: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
        self.models_dir = models_dir or get_models_dir()
    
    def register_model(
        self,
        name: str,
        model_type: str,
        version: str,
        model_obj: Any,
        hyperparameters: Optional[Dict] = None,
        trained_at: Optional[datetime] = None
    ) -> int:
        """
        Register a new model in the database and save the model file.
        
        Args:
            name: Model name (e.g., "lstm_v1")
            model_type: Model type (e.g., "lstm", "transformer", "baseline")
            version: Version string
            model_obj: The trained model object to serialize
            hyperparameters: Dict of hyperparameters
            trained_at: When model was trained (defaults to now)
        
        Returns:
            The model ID from the database
        """
        if trained_at is None:
            trained_at = datetime.utcnow()
        
        # Generate filename and save model
        filename = f"{name}_{version}_{int(trained_at.timestamp())}.pkl"
        file_path = self.models_dir / filename
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_obj, f)
        
        # Insert into database
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO models (name, type, version, file_path, hyperparameters, trained_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                model_type,
                version,
                str(file_path),
                json.dumps(hyperparameters) if hyperparameters else None,
                trained_at.isoformat()
            ))
            model_id = cursor.lastrowid
        
        print(f"Registered model '{name}' (ID: {model_id}) at {file_path}")
        return model_id
    
    def load_model(self, model_id: int) -> tuple[Any, Dict]:
        """
        Load a model by ID.
        
        Args:
            model_id: The model ID from database
        
        Returns:
            Tuple of (model_object, metadata_dict)
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Model ID {model_id} not found")
        
        metadata = dict(row)
        file_path = Path(metadata['file_path'])
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            model_obj = pickle.load(f)
        
        return model_obj, metadata
    
    def get_model_by_name(self, name: str, version: Optional[str] = None) -> tuple[int, Any, Dict]:
        """
        Get model by name, optionally filtering by version.
        Returns the most recent if multiple matches.
        
        Returns:
            Tuple of (model_id, model_object, metadata_dict)
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if version:
                cursor.execute("""
                    SELECT * FROM models 
                    WHERE name = ? AND version = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (name, version))
            else:
                cursor.execute("""
                    SELECT * FROM models 
                    WHERE name = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (name,))
            row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Model '{name}' (version: {version}) not found")
        
        metadata = dict(row)
        model_id = metadata['id']
        model_obj, _ = self.load_model(model_id)
        
        return model_id, model_obj, metadata
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """
        List all registered models, optionally filtered by type.
        
        Returns:
            List of model metadata dicts
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if model_type:
                cursor.execute("""
                    SELECT * FROM models 
                    WHERE type = ?
                    ORDER BY created_at DESC
                """, (model_type,))
            else:
                cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def delete_model(self, model_id: int, remove_file: bool = True):
        """
        Delete a model from the registry.
        
        Args:
            model_id: The model ID to delete
            remove_file: Whether to also delete the model file
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Model ID {model_id} not found")
            
            file_path = Path(row['file_path'])
            
            # Delete from database
            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
            
            # Delete file if requested and exists
            if remove_file and file_path.exists():
                file_path.unlink()
                print(f"Deleted model file: {file_path}")
        
        print(f"Deleted model ID {model_id} from registry")


if __name__ == "__main__":
    # Demo: register a simple model
    from .database import init_db
    
    init_db()
    registry = ModelRegistry()
    
    # Create a dummy model (simple dict for demo)
    dummy_model = {"type": "baseline", "strategy": "moving_average", "window": 20}
    
    model_id = registry.register_model(
        name="ma_baseline",
        model_type="baseline",
        version="1.0.0",
        model_obj=dummy_model,
        hyperparameters={"window": 20}
    )
    
    print(f"\nRegistered model ID: {model_id}")
    
    # List all models
    models = registry.list_models()
    print(f"\nAll models: {len(models)}")
    for m in models:
        print(f"  - {m['name']} v{m['version']} (ID: {m['id']}, type: {m['type']})")
