"""Performance tracking system for model evaluation across market conditions."""
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .database import get_connection, get_db_path


def detect_market_condition(candles: np.ndarray) -> str:
    """
    Detect current market condition based on recent candle data.
    
    Args:
        candles: Array of shape (n, 5) with columns [open, high, low, close, volume]
    
    Returns:
        Market condition label: "volatile", "trending_up", "trending_down", "range", "unknown"
    """
    if len(candles) < 20:
        return "unknown"
    
    closes = candles[:, 3]
    highs = candles[:, 1]
    lows = candles[:, 2]
    
    # Calculate volatility (standard deviation of returns)
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns)
    
    # Calculate trend strength (linear regression slope)
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    slope_pct = (slope / np.mean(closes)) * 100
    
    # Calculate range (high-low relative to price)
    price_range = (np.max(highs) - np.min(lows)) / np.mean(closes)
    
    # Classify market condition
    # High volatility threshold
    if volatility > 0.02:
        return "volatile"
    
    # Trending conditions
    if abs(slope_pct) > 0.5:
        if slope_pct > 0:
            return "trending_up"
        else:
            return "trending_down"
    
    # Range-bound (sideways)
    if price_range < 0.05:
        return "range"
    
    return "unknown"


class PerformanceTracker:
    """Tracker for model performance across different market conditions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
    
    def update_performance(
        self,
        model_id: int,
        symbol: str,
        interval: str,
        market_condition: str
    ):
        """
        Update performance metrics for a model in a specific market condition.
        Calculates accuracy from predictions table and updates model_performance.
        
        Args:
            model_id: The model ID
            symbol: Trading pair
            interval: Timeframe
            market_condition: Market condition label
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate metrics from predictions
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = actual_result THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE model_id = ? 
                  AND symbol = ? 
                  AND interval = ?
                  AND actual_result IS NOT NULL
            """, (model_id, symbol.lower(), interval))
            
            row = cursor.fetchone()
            total = row['total'] or 0
            correct = row['correct'] or 0
            accuracy = (correct / total) if total > 0 else 0.0
            
            # Check if record exists
            cursor.execute("""
                SELECT id FROM model_performance
                WHERE model_id = ? AND symbol = ? AND interval = ? AND market_condition = ?
            """, (model_id, symbol.lower(), interval, market_condition))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE model_performance
                    SET total_predictions = ?,
                        correct_predictions = ?,
                        accuracy = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (total, correct, accuracy, datetime.utcnow().isoformat(), existing['id']))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO model_performance 
                    (model_id, symbol, interval, market_condition, total_predictions, correct_predictions, accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (model_id, symbol.lower(), interval, market_condition, total, correct, accuracy))
    
    def get_performance(
        self,
        model_id: int,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        market_condition: Optional[str] = None
    ) -> List[Dict]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: The model ID
            symbol: Optional symbol filter
            interval: Optional interval filter
            market_condition: Optional market condition filter
        
        Returns:
            List of performance metric dicts
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM model_performance WHERE model_id = ?"
            params = [model_id]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.lower())
            
            if interval:
                query += " AND interval = ?"
                params.append(interval)
            
            if market_condition:
                query += " AND market_condition = ?"
                params.append(market_condition)
            
            query += " ORDER BY accuracy DESC"
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_best_model_for_condition(
        self,
        symbol: str,
        interval: str,
        market_condition: str,
        min_predictions: int = 10
    ) -> Optional[Dict]:
        """
        Find the best performing model for a specific market condition.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            market_condition: Market condition label
            min_predictions: Minimum number of predictions required
        
        Returns:
            Dict with best model info or None if no models meet criteria
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mp.*, m.name, m.type, m.version
                FROM model_performance mp
                JOIN models m ON mp.model_id = m.id
                WHERE mp.symbol = ? 
                  AND mp.interval = ?
                  AND mp.market_condition = ?
                  AND mp.total_predictions >= ?
                ORDER BY mp.accuracy DESC
                LIMIT 1
            """, (symbol.lower(), interval, market_condition, min_predictions))
            
            row = cursor.fetchone()
        
        return dict(row) if row else None
    
    def get_model_comparison(
        self,
        symbol: str,
        interval: str,
        market_condition: Optional[str] = None
    ) -> List[Dict]:
        """
        Compare all models for a given symbol/interval.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            market_condition: Optional market condition filter
        
        Returns:
            List of model performance dicts sorted by accuracy
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT mp.*, m.name, m.type, m.version
                FROM model_performance mp
                JOIN models m ON mp.model_id = m.id
                WHERE mp.symbol = ? AND mp.interval = ?
            """
            params = [symbol.lower(), interval]
            
            if market_condition:
                query += " AND mp.market_condition = ?"
                params.append(market_condition)
            
            query += " ORDER BY mp.accuracy DESC"
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]


if __name__ == "__main__":
    # Demo: track performance across market conditions
    from .database import init_db
    from .model_registry import ModelRegistry
    from .prediction_logger import PredictionLogger
    from .models.baseline import MovingAverageCrossoverModel, MomentumModel
    import random
    
    init_db()
    registry = ModelRegistry()
    logger = PredictionLogger()
    tracker = PerformanceTracker()
    
    # Register two models
    models = [
        MovingAverageCrossoverModel(),
        MomentumModel()
    ]
    
    model_ids = []
    for model in models:
        model_id = registry.register_model(
            name=model.name,
            model_type=model.model_type,
            version=model.version,
            model_obj=model,
            hyperparameters=model.get_hyperparameters()
        )
        model_ids.append(model_id)
        print(f"Registered {model.name} (ID: {model_id})")
    
    # Simulate predictions in different market conditions
    print("\nSimulating predictions...")
    conditions = ["trending_up", "trending_down", "volatile", "range"]
    
    for condition in conditions:
        for model, model_id in zip(models, model_ids):
            # Simulate 20 predictions per condition
            for _ in range(20):
                features = {
                    "ma5": 50000 + np.random.randn() * 1000,
                    "ma20": 50000 + np.random.randn() * 500,
                    "price_change_pct": np.random.randn() * 2
                }
                
                prediction, confidence = model.predict(features)
                
                # Bias actual results based on condition to simulate performance differences
                if condition == "trending_up" and model.name == "momentum":
                    actual = "up" if random.random() > 0.3 else random.choice(["down", "neutral"])
                elif condition == "volatile" and model.name == "ma_crossover":
                    actual = prediction if random.random() > 0.4 else random.choice(["up", "down", "neutral"])
                else:
                    actual = random.choice(["up", "down", "neutral"])
                
                pred_id = logger.log_prediction(
                    model_id=model_id,
                    symbol="btcusdt",
                    interval="5m",
                    prediction=prediction,
                    confidence=confidence,
                    features=features
                )
                logger.update_actual_result(pred_id, actual)
        
        # Update performance for this condition
        for model_id in model_ids:
            tracker.update_performance(model_id, "btcusdt", "5m", condition)
    
    # Show performance comparison
    print("\nPerformance by market condition:")
    for condition in conditions:
        print(f"\n{condition.upper()}:")
        comparison = tracker.get_model_comparison("btcusdt", "5m", condition)
        for perf in comparison:
            print(f"  {perf['name']}: {perf['accuracy']:.2%} ({perf['correct_predictions']}/{perf['total_predictions']})")
        
        best = tracker.get_best_model_for_condition("btcusdt", "5m", condition, min_predictions=5)
        if best:
            print(f"  → Best: {best['name']}")
