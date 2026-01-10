"""Baseline prediction models for market movement forecasting."""
import random
import numpy as np
from typing import Dict, Any, Tuple

from .base import BaseModel


class MovingAverageCrossoverModel(BaseModel):
    """Predict based on moving average crossover strategy."""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, version: str = "1.0.0"):
        super().__init__(name="ma_crossover", version=version)
        self.model_type = "baseline"
        self.short_window = short_window
        self.long_window = long_window
    
    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict market direction based on MA crossover.
        - Up: if short MA > long MA
        - Down: if short MA < long MA
        - Neutral: if MAs are too close or not available
        """
        ma_short = features.get("ma5")
        ma_long = features.get("ma20")
        
        if ma_short is None or ma_long is None:
            return "neutral", 0.5
        
        diff_pct = abs(ma_short - ma_long) / ma_long * 100
        
        # If difference is very small, neutral
        if diff_pct < 0.1:
            return "neutral", 0.5
        
        # Predict based on crossover
        if ma_short > ma_long:
            # Confidence increases with larger gap
            confidence = min(0.6 + diff_pct * 0.1, 0.95)
            return "up", confidence
        else:
            confidence = min(0.6 + diff_pct * 0.1, 0.95)
            return "down", confidence
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window
        }


class MomentumModel(BaseModel):
    """Predict based on recent price momentum."""
    
    def __init__(self, lookback: int = 5, threshold: float = 0.5, version: str = "1.0.0"):
        super().__init__(name="momentum", version=version)
        self.model_type = "baseline"
        self.lookback = lookback
        self.threshold = threshold  # min % change to predict up/down
    
    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict based on recent price change percentage.
        - Up: if price increased > threshold
        - Down: if price decreased > threshold
        - Neutral: otherwise
        """
        price_change_pct = features.get("price_change_pct")
        
        if price_change_pct is None:
            return "neutral", 0.5
        
        abs_change = abs(price_change_pct)
        
        if abs_change < self.threshold:
            return "neutral", 0.5
        
        # Confidence increases with larger momentum
        confidence = min(0.55 + abs_change * 0.05, 0.9)
        
        if price_change_pct > 0:
            return "up", confidence
        else:
            return "down", confidence
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "threshold": self.threshold
        }


class RandomModel(BaseModel):
    """Random baseline for comparison (baseline of baselines)."""
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__(name="random", version=version)
        self.model_type = "baseline"
    
    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """Randomly predict up/down/neutral with equal probability."""
        prediction = random.choice(["up", "down", "neutral"])
        confidence = random.uniform(0.4, 0.6)
        return prediction, confidence
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {}


class VolumeWeightedModel(BaseModel):
    """Predict based on volume trends combined with price."""
    
    def __init__(self, volume_threshold: float = 1.5, version: str = "1.0.0"):
        super().__init__(name="volume_weighted", version=version)
        self.model_type = "baseline"
        self.volume_threshold = volume_threshold  # multiplier over avg volume
    
    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict based on volume spike + price direction.
        High volume adds confidence to price movement prediction.
        """
        last_volume = features.get("last_volume")
        volume_ma5 = features.get("volume_ma5")
        price_change_pct = features.get("price_change_pct")
        
        if last_volume is None or volume_ma5 is None or price_change_pct is None:
            return "neutral", 0.5
        
        volume_ratio = last_volume / volume_ma5 if volume_ma5 > 0 else 1.0
        
        # If volume is low, lower confidence
        if volume_ratio < 0.8:
            confidence = 0.45
            return "neutral", confidence
        
        # High volume increases confidence
        volume_boost = min((volume_ratio - 1.0) * 0.2, 0.3)
        
        if abs(price_change_pct) < 0.2:
            return "neutral", 0.5
        
        base_confidence = 0.55 + abs(price_change_pct) * 0.03
        confidence = min(base_confidence + volume_boost, 0.95)
        
        if price_change_pct > 0:
            return "up", confidence
        else:
            return "down", confidence
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "volume_threshold": self.volume_threshold
        }


if __name__ == "__main__":
    # Demo: test baseline models
    
    # Simulate some candle data (open, high, low, close, volume)
    np.random.seed(42)
    base_price = 50000
    candles = []
    for i in range(30):
        open_price = base_price + np.random.randn() * 100
        close_price = open_price + np.random.randn() * 200
        high_price = max(open_price, close_price) + abs(np.random.randn() * 50)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 50)
        volume = abs(np.random.randn() * 1000000 + 5000000)
        candles.append([open_price, high_price, low_price, close_price, volume])
        base_price = close_price
    
    candles = np.array(candles)
    
    # Test each model
    models = [
        MovingAverageCrossoverModel(),
        MomentumModel(),
        RandomModel(),
        VolumeWeightedModel()
    ]
    
    print("Testing baseline models with sample data:\n")
    
    for model in models:
        features = model.prepare_features(candles)
        prediction, confidence = model.predict(features)
        print(f"{model.name}:")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Hyperparameters: {model.get_hyperparameters()}")
        print()
