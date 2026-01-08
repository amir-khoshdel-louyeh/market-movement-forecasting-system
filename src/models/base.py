"""Base class for prediction models."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.model_type = "base"
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Make a prediction based on input features.
        
        Args:
            features: Dictionary of feature values (e.g., prices, volumes, indicators)
        
        Returns:
            Tuple of (prediction, confidence)
            - prediction: "up", "down", or "neutral"
            - confidence: float between 0 and 1
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return the model's hyperparameters as a dict."""
        pass
    
    def prepare_features(self, candles: np.ndarray) -> Dict[str, Any]:
        """
        Prepare features from raw candle data.
        
        Args:
            candles: Array of shape (n, 5) with columns [open, high, low, close, volume]
        
        Returns:
            Dictionary of computed features
        """
        if len(candles) == 0:
            return {}
        
        closes = candles[:, 3]
        volumes = candles[:, 4]
        
        features = {
            "last_close": float(closes[-1]),
            "last_volume": float(volumes[-1]),
        }
        
        if len(closes) >= 2:
            features["price_change"] = float(closes[-1] - closes[-2])
            features["price_change_pct"] = float((closes[-1] - closes[-2]) / closes[-2] * 100)
        
        if len(closes) >= 5:
            features["ma5"] = float(np.mean(closes[-5:]))
        
        if len(closes) >= 20:
            features["ma20"] = float(np.mean(closes[-20:]))
        
        if len(volumes) >= 5:
            features["volume_ma5"] = float(np.mean(volumes[-5:]))
        
        return features
