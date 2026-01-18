"""Backtest engine with labeling and train/test split."""
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from .prediction_logger import PredictionLogger
from .performance_tracker import PerformanceTracker, detect_market_condition
from .model_registry import ModelRegistry


DEFAULT_THRESHOLD = 0.2  # % move to be considered up/down


def label_next(current_close: float, next_close: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Classify next-candle move with neutral band."""
    if current_close == 0:
        return "neutral"
    pct = (next_close - current_close) / current_close * 100
    if pct > threshold:
        return "up"
    if pct < -threshold:
        return "down"
    return "neutral"


def backtest_model(
    model_obj,
    candles: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    window_size: int = 30,
    step: int = 1,
) -> Dict[str, Any]:
    """Run sliding-window backtest, return metrics without DB side-effects."""
    if len(candles) <= window_size:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "by_condition": {}}

    y_true, y_pred = [], []
    cond_stats: Dict[str, Dict[str, int]] = {}

    for start in range(0, len(candles) - window_size - 1, step):
        subset = candles[start : start + window_size]
        curr_close = float(subset[-1, 3])
        next_close = float(candles[start + window_size, 3])
        actual = label_next(curr_close, next_close, threshold)
        features = model_obj.prepare_features(subset)
        pred, _ = model_obj.predict(features)
        cond = detect_market_condition(subset)
        y_true.append(actual)
        y_pred.append(pred)
        if cond not in cond_stats:
            cond_stats[cond] = {"total": 0, "correct": 0}
        cond_stats[cond]["total"] += 1
        if pred == actual:
            cond_stats[cond]["correct"] += 1

    total = len(y_true)
    correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    acc = correct / total if total else 0.0

    by_cond = {
        k: {"total": v["total"], "correct": v["correct"], "accuracy": v["correct"] / v["total"] if v["total"] else 0}
        for k, v in cond_stats.items()
    }
    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "by_condition": by_cond,
        "threshold": threshold,
        "window_size": window_size,
    }


def run_backtest(
    candles: np.ndarray,
    symbol: str,
    interval: str,
    registry: Optional[ModelRegistry] = None,
    logger: Optional[PredictionLogger] = None,
    tracker: Optional[PerformanceTracker] = None,
    threshold: float = DEFAULT_THRESHOLD,
    window_size: int = 30,
    step: int = 5,
    train_ratio: float = 0.8,
    log_to_db: bool = True,
) -> Dict[str, Any]:
    """Train/test split backtest, optionally log to DB and update performance."""
    registry = registry or ModelRegistry()
    logger = logger or PredictionLogger()
    tracker = tracker or PerformanceTracker()

    models = registry.list_models()
    if not models:
        return {"ok": False, "error": "no models registered"}

    n = len(candles)
    split = int(n * train_ratio) if 0 < train_ratio < 1 else n
    train_candles = candles[:split]
    test_candles = candles[split - window_size :] if split > window_size else candles

    results = []
    predictions_generated = 0
    conditions_seen = set()

    # Fit deep models on train split only
    fit_results = {}
    for meta in models:
        if meta["type"] in ("lstm", "transformer"):
            try:
                obj, file_meta = registry.load_model(meta["id"])
                if hasattr(obj, "fit"):
                    res = obj.fit(train_candles, epochs=10)
                    fit_results[meta["name"]] = res
                    if res.get("ok"):
                        import pickle
                        from pathlib import Path
                        p = Path(file_meta["file_path"])
                        with open(p, "wb") as f:
                            pickle.dump(obj, f)
            except Exception as e:
                fit_results[meta["name"]] = {"ok": False, "error": str(e)}

    # Evaluate on test split (with optional DB logging)
    for meta in models:
        obj, _ = registry.load_model(meta["id"])
        # dry backtest metrics
        metrics = backtest_model(obj, test_candles, threshold=threshold, window_size=window_size, step=step)
        entry = {"model": meta, "metrics": metrics, "fit": fit_results.get(meta["name"])}
        results.append(entry)

        if log_to_db:
            for start in range(0, len(test_candles) - window_size - 1, step):
                subset = test_candles[start : start + window_size]
                cond = detect_market_condition(subset)
                conditions_seen.add(cond)
                features = obj.prepare_features(subset)
                pred, conf = obj.predict(features)
                curr_close = float(subset[-1, 3])
                next_close = float(test_candles[start + window_size, 3])
                actual = label_next(curr_close, next_close, threshold)
                # avoid serializing large arrays in features for DB; keep scalar features + threshold
                safe_features = {k: v for k, v in features.items() if k not in ("sequence", "raw_sequence")}
                safe_features["_threshold"] = threshold
                # truncate sequence to avoid huge json
                pid = logger.log_prediction(
                    model_id=meta["id"],
                    symbol=symbol,
                    interval=interval,
                    prediction=pred,
                    confidence=conf,
                    features=safe_features,
                )
                logger.update_actual_result(pid, actual)
                predictions_generated += 1

    if log_to_db:
        for meta in models:
            # update mixed + per-condition would be better, but update at least mixed bucket
            tracker.update_performance(meta["id"], symbol, interval, "mixed")
            # also show per-condition stats in response without writing extra rows

    return {
        "ok": True,
        "train_samples": int(len(train_candles)),
        "test_samples": int(len(test_candles)),
        "predictions_generated": predictions_generated,
        "conditions": sorted(list(conditions_seen)),
        "threshold": threshold,
        "window_size": window_size,
        "results": results,
        "fit_results": fit_results,
    }
