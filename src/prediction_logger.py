"""Prediction logging system for storing and tracking model predictions."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .database import get_connection, get_db_path


class PredictionLogger:
    """Logger for storing and managing model predictions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
    
    def log_prediction(
        self,
        model_id: int,
        symbol: str,
        interval: str,
        prediction: str,
        confidence: float,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Log a new prediction to the database.
        
        Args:
            model_id: ID of the model making the prediction
            symbol: Trading pair (e.g., "btcusdt")
            interval: Timeframe (e.g., "1m", "5m")
            prediction: Predicted direction ("up", "down", "neutral")
            confidence: Confidence score (0-1)
            features: Dictionary of input features used
            timestamp: When prediction was made (defaults to now)
        
        Returns:
            The prediction ID
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions 
                (model_id, symbol, interval, timestamp, prediction, confidence, features)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                symbol.lower(),
                interval,
                timestamp.isoformat(),
                prediction,
                confidence,
                json.dumps(features)
            ))
            prediction_id = cursor.lastrowid
        
        return prediction_id
    
    def update_actual_result(
        self,
        prediction_id: int,
        actual_result: str,
        result_timestamp: Optional[datetime] = None
    ):
        """
        Update a prediction with the actual market result.
        
        Args:
            prediction_id: The prediction ID to update
            actual_result: Actual outcome ("up", "down", "neutral")
            result_timestamp: When the result was observed (defaults to now)
        """
        if result_timestamp is None:
            result_timestamp = datetime.utcnow()
        
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions
                SET actual_result = ?, result_timestamp = ?
                WHERE id = ?
            """, (actual_result, result_timestamp.isoformat(), prediction_id))
    
    def get_prediction(self, prediction_id: int) -> Optional[Dict]:
        """Get a single prediction by ID."""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
        
        return dict(row) if row else None
    
    def get_predictions_by_model(
        self,
        model_id: int,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get predictions for a specific model.
        
        Args:
            model_id: The model ID
            symbol: Optional symbol filter
            limit: Maximum number of results
        
        Returns:
            List of prediction dicts
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE model_id = ? AND symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (model_id, symbol.lower(), limit))
            else:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE model_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (model_id, limit))
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_pending_predictions(
        self,
        model_id: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get predictions that don't have actual results yet.
        
        Args:
            model_id: Optional model ID filter
            symbol: Optional symbol filter
        
        Returns:
            List of prediction dicts without actual_result
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM predictions WHERE actual_result IS NULL"
            params = []
            
            if model_id is not None:
                query += " AND model_id = ?"
                params.append(model_id)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.lower())
            
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def calculate_accuracy(
        self,
        model_id: int,
        symbol: Optional[str] = None,
        interval: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate accuracy metrics for a model.
        
        Args:
            model_id: The model ID
            symbol: Optional symbol filter
            interval: Optional interval filter
        
        Returns:
            Dict with accuracy metrics
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = actual_result THEN 1 ELSE 0 END) as correct,
                    AVG(confidence) as avg_confidence
                FROM predictions
                WHERE model_id = ? AND actual_result IS NOT NULL
            """
            params = [model_id]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.lower())
            
            if interval:
                query += " AND interval = ?"
                params.append(interval)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
        
        total = row['total'] or 0
        correct = row['correct'] or 0
        avg_confidence = row['avg_confidence'] or 0.0
        
        accuracy = (correct / total) if total > 0 else 0.0
        
        return {
            "model_id": model_id,
            "symbol": symbol,
            "interval": interval,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence
        }


if __name__ == "__main__":
    # Demo: log predictions and calculate accuracy
    from .database import init_db
    from .model_registry import ModelRegistry
    from .models.baseline import MovingAverageCrossoverModel
    import numpy as np
    
    init_db()
    registry = ModelRegistry()
    logger = PredictionLogger()
    
    # Register a model
    model = MovingAverageCrossoverModel()
    model_id = registry.register_model(
        name=model.name,
        model_type=model.model_type,
        version=model.version,
        model_obj=model,
        hyperparameters=model.get_hyperparameters()
    )
    
    print(f"Registered model ID: {model_id}\n")
    
    # Simulate predictions
    symbols = ["btcusdt", "ethusdt"]
    
    for i in range(10):
        symbol = np.random.choice(symbols)
        
        # Fake features
        features = {
            "ma5": 50000 + np.random.randn() * 1000,
            "ma20": 50000 + np.random.randn() * 500,
            "last_close": 50000 + np.random.randn() * 1000
        }
        
        prediction, confidence = model.predict(features)
        
        pred_id = logger.log_prediction(
            model_id=model_id,
            symbol=symbol,
            interval="1m",
            prediction=prediction,
            confidence=confidence,
            features=features
        )
        
        # Simulate actual result (random for demo)
        actual = np.random.choice(["up", "down", "neutral"])
        logger.update_actual_result(pred_id, actual)
        
        print(f"Prediction {pred_id}: {prediction} (confidence: {confidence:.2f}), Actual: {actual}")
    
    # Calculate accuracy
    print("\nAccuracy metrics:")
    metrics = logger.calculate_accuracy(model_id)
    print(f"  Total predictions: {metrics['total_predictions']}")
    print(f"  Correct: {metrics['correct_predictions']}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Avg confidence: {metrics['avg_confidence']:.2f}")
