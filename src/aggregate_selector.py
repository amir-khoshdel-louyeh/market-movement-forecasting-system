"""Aggregate model selector that chooses the best model based on market conditions."""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .database import get_db_path
from .model_registry import ModelRegistry
from .performance_tracker import PerformanceTracker, detect_market_condition
from .prediction_logger import PredictionLogger


class AggregateModelSelector:
    """
    Meta-model that selects the best performing model for current market conditions.
    
    This implements the core "learning" component that decides which model to use
    based on historical performance in similar market conditions.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        min_predictions: int = 10,
        fallback_model_id: Optional[int] = None
    ):
        """
        Initialize the aggregate model selector.
        
        Args:
            db_path: Database path
            min_predictions: Minimum predictions required to trust a model's performance
            fallback_model_id: Default model ID to use when no performance data exists
        """
        self.db_path = db_path or get_db_path()
        self.registry = ModelRegistry(db_path=self.db_path)
        self.tracker = PerformanceTracker(db_path=self.db_path)
        self.logger = PredictionLogger(db_path=self.db_path)
        self.min_predictions = min_predictions
        self.fallback_model_id = fallback_model_id
    
    def select_best_model(
        self,
        symbol: str,
        interval: str,
        candles: np.ndarray
    ) -> Tuple[int, str, Dict]:
        """
        Select the best model for current market conditions.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            candles: Recent candle data for market condition detection
        
        Returns:
            Tuple of (model_id, market_condition, model_metadata)
        """
        # Detect current market condition
        market_condition = detect_market_condition(candles)
        
        # Find best performing model for this condition
        best_perf = self.tracker.get_best_model_for_condition(
            symbol=symbol,
            interval=interval,
            market_condition=market_condition,
            min_predictions=self.min_predictions
        )
        
        if best_perf:
            model_id = best_perf['model_id']
            _, model_obj, metadata = self.registry.get_model_by_name(best_perf['name'])
            return model_id, market_condition, metadata
        
        # Fallback: if no performance data, use fallback or first available model
        if self.fallback_model_id:
            _, metadata = self.registry.load_model(self.fallback_model_id)
            return self.fallback_model_id, market_condition, metadata
        
        # Last resort: use any available model
        models = self.registry.list_models()
        if models:
            model_id = models[0]['id']
            _, metadata = self.registry.load_model(model_id)
            return model_id, market_condition, metadata
        
        raise ValueError("No models available in registry")
    
    def predict(
        self,
        symbol: str,
        interval: str,
        candles: np.ndarray,
        log_prediction: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction using the best model for current conditions.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            candles: Recent candle data
            log_prediction: Whether to log the prediction to database
        
        Returns:
            Dict with prediction details including selected model info
        """
        # Select best model
        model_id, market_condition, model_metadata = self.select_best_model(
            symbol=symbol,
            interval=interval,
            candles=candles
        )
        
        # Load model
        model_obj, _ = self.registry.load_model(model_id)
        
        # Prepare features
        features = model_obj.prepare_features(candles)
        
        # Make prediction
        prediction, confidence = model_obj.predict(features)
        
        # Log prediction if requested
        prediction_id = None
        if log_prediction:
            prediction_id = self.logger.log_prediction(
                model_id=model_id,
                symbol=symbol,
                interval=interval,
                prediction=prediction,
                confidence=confidence,
                features=features
            )
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_id": model_id,
            "model_name": model_metadata['name'],
            "model_type": model_metadata['type'],
            "market_condition": market_condition,
            "prediction_id": prediction_id,
            "features": features
        }
    
    def predict_with_model(
        self,
        model_id: int,
        symbol: str,
        interval: str,
        candles: np.ndarray,
        log_prediction: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction using a specific model.
        
        Args:
            model_id: The specific model ID to use
            symbol: Trading pair
            interval: Timeframe
            candles: Recent candle data
            log_prediction: Whether to log the prediction to database
        
        Returns:
            Dict with prediction details
        """
        # Detect current market condition
        market_condition = detect_market_condition(candles)
        
        # Load the specified model
        model_obj, model_metadata = self.registry.load_model(model_id)
        
        # Prepare features
        features = model_obj.prepare_features(candles)
        
        # Make prediction
        prediction, confidence = model_obj.predict(features)
        
        # Log prediction if requested
        prediction_id = None
        if log_prediction:
            prediction_id = self.logger.log_prediction(
                model_id=model_id,
                symbol=symbol,
                interval=interval,
                prediction=prediction,
                confidence=confidence,
                features=features
            )
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_id": model_id,
            "model_name": model_metadata['name'],
            "model_type": model_metadata['type'],
            "market_condition": market_condition,
            "prediction_id": prediction_id,
            "features": features
        }
    
    def get_ensemble_prediction(
        self,
        symbol: str,
        interval: str,
        candles: np.ndarray,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Make prediction using ensemble of top N models (weighted voting).
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            candles: Recent candle data
            top_n: Number of top models to include in ensemble
        
        Returns:
            Dict with ensemble prediction and contributing models
        """
        # Detect market condition
        market_condition = detect_market_condition(candles)
        
        # Get top N models for this condition
        comparison = self.tracker.get_model_comparison(symbol, interval, market_condition)
        top_models = comparison[:top_n] if len(comparison) >= top_n else comparison
        
        if not top_models:
            # No performance data, fall back to single model prediction
            return self.predict(symbol, interval, candles, log_prediction=False)
        
        # Collect predictions from each model
        votes = {"up": 0.0, "down": 0.0, "neutral": 0.0}
        model_predictions = []
        
        for perf in top_models:
            model_id = perf['model_id']
            model_obj, metadata = self.registry.load_model(model_id)
            features = model_obj.prepare_features(candles)
            prediction, confidence = model_obj.predict(features)
            
            # Weight by accuracy and confidence
            weight = perf['accuracy'] * confidence
            votes[prediction] += weight
            
            model_predictions.append({
                "model_id": model_id,
                "model_name": metadata['name'],
                "prediction": prediction,
                "confidence": confidence,
                "weight": weight
            })
        
        # Select prediction with highest weighted vote
        final_prediction = max(votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(votes.values())
        final_confidence = votes[final_prediction] / total_weight if total_weight > 0 else 0.5
        
        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "market_condition": market_condition,
            "ensemble_votes": votes,
            "contributing_models": model_predictions,
            "method": "ensemble"
        }


if __name__ == "__main__":
    # Demo: use aggregate selector
    from .database import init_db
    from .models.baseline import MovingAverageCrossoverModel, MomentumModel, VolumeWeightedModel
    
    init_db()
    
    # Register models
    registry = ModelRegistry()
    models = [
        MovingAverageCrossoverModel(),
        MomentumModel(),
        VolumeWeightedModel()
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
    
    # Generate sample candle data
    np.random.seed(42)
    base_price = 50000
    candles = []
    for i in range(30):
        trend = i * 50  # Add uptrend
        open_price = base_price + trend + np.random.randn() * 100
        close_price = open_price + np.random.randn() * 200 + 20
        high_price = max(open_price, close_price) + abs(np.random.randn() * 50)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 50)
        volume = abs(np.random.randn() * 1000000 + 5000000)
        candles.append([open_price, high_price, low_price, close_price, volume])
        base_price = close_price
    
    candles = np.array(candles)
    
    # Create aggregate selector
    selector = AggregateModelSelector(fallback_model_id=model_ids[0])
    
    print("\n--- Single Model Selection (Best for Condition) ---")
    result = selector.predict("btcusdt", "5m", candles, log_prediction=False)
    print(f"Market Condition: {result['market_condition']}")
    print(f"Selected Model: {result['model_name']} (ID: {result['model_id']})")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\n--- Ensemble Prediction (Top 3 Models) ---")
    ensemble = selector.get_ensemble_prediction("btcusdt", "5m", candles, top_n=3)
    print(f"Market Condition: {ensemble['market_condition']}")
    print(f"Ensemble Prediction: {ensemble['prediction']}")
    print(f"Ensemble Confidence: {ensemble['confidence']:.2f}")
    print(f"Votes: {ensemble['ensemble_votes']}")
    print(f"Contributing models: {len(ensemble['contributing_models'])}")
