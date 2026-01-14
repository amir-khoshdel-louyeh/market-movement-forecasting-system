import json
import threading
import asyncio
import queue
import time
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, request, jsonify, render_template
import requests

from .mmfs_stream import StreamConfig, stream_kline, DEFAULT_SYMBOL
from .model_registry import ModelRegistry
from .prediction_logger import PredictionLogger
from .performance_tracker import PerformanceTracker
from .aggregate_selector import AggregateModelSelector
from .database import init_db


class WebState:
    def __init__(self):
        self.df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])  # Date index
        self.df.index.name = "Date"
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()
        self.stream_thread: threading.Thread | None = None
        self.stop_event: threading.Event = threading.Event()
        self.symbol: str = DEFAULT_SYMBOL
        self.interval: str = "1m"
        self.max_rows: int = 500

    def upsert_kline(self, dt: datetime, k):
        new = {
            "Open": float(k.o),
            "High": float(k.h),
            "Low": float(k.l),
            "Close": float(k.c),
            "Volume": float(k.v),
        }
        if dt in self.df.index:
            old = self.df.loc[dt]
            merged = {
                "Open": float(old["Open"]),
                "High": max(float(old["High"]), new["High"]),
                "Low": min(float(old["Low"]), new["Low"]),
                "Close": new["Close"],
                "Volume": new["Volume"],
            }
            self.df.loc[dt] = merged
        else:
            self.df.loc[dt] = new
        self.df = self.df.sort_index()
        if len(self.df) > self.max_rows:
            self.df = self.df.tail(self.max_rows)


state = WebState()
app = Flask(__name__, template_folder="templates", static_folder="static")


def _publish(event: dict):
    with state.lock:
        for q in list(state.subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass


def _start_stream(symbol: str, interval: str):
    if state.stream_thread and state.stream_thread.is_alive():
        # If config unchanged, keep running; else request stop and restart.
        if state.symbol == symbol and state.interval == interval:
            return
        state.stop_event.set()
        state.stream_thread.join(timeout=3)

    # Prepare fresh stop flag and config
    state.stop_event = threading.Event()
    state.symbol = symbol
    state.interval = interval

    # Seed history for the newly selected symbol/interval so the UI loads matching candles
    _seed_history(symbol, interval, limit=50)

    def on_kline(kmsg):
        k = kmsg.k
        # Use UTC to avoid local-time shifts; keep ms epoch for clients
        dt = pd.to_datetime(k.t, unit='ms', utc=True)
        state.upsert_kline(dt, k)
        event = {
            "t": int(k.t),
            "closed": bool(k.x),
            "o": float(k.o),
            "h": float(k.h),
            "l": float(k.l),
            "c": float(k.c),
            "v": float(k.v),
        }
        _publish(event)

    def run():
        cfg = StreamConfig(symbol=symbol)
        asyncio.run(stream_kline(cfg, interval=interval, on_kline=on_kline, stop_event=state.stop_event))

    state.stream_thread = threading.Thread(target=run, daemon=True)
    state.stream_thread.start()


def _seed_history(symbol: str, interval: str, limit: int = 50):
    """Fetch recent klines from Binance REST and seed server state.

    Uses open time as the index. Volume is cumulative for each candle.
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        for k in data:
            # [openTime, open, high, low, close, volume, closeTime, ...]
            ot, o, h, l, c, v = k[0], k[1], k[2], k[3], k[4], k[5]
            dt = pd.to_datetime(ot, unit='ms', utc=True)
            rows.append((dt, float(o), float(h), float(l), float(c), float(v)))
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
        with state.lock:
            state.df = df.tail(state.max_rows)
    except Exception as e:
        print(f"Seed history failed: {e}")


def _get_historical_data_cache_path(symbol: str, interval: str, days: int = 30) -> Path:
    """Get the cache file path for historical data."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"historical_{symbol.lower()}_{interval}_{days}d.csv"
    return data_dir / filename


def _download_historical_data(symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
    """
    Download historical kline data from Binance.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "1m", "5m", "1h")
        days: Number of days to download
        
    Returns:
        DataFrame with OHLCV data
    """
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    # Calculate number of candles needed (depends on interval)
    interval_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440
    }
    minutes_per_interval = interval_minutes.get(interval, 60)
    total_candles_needed = (days * 24 * 60) // minutes_per_interval
    
    # Binance API limit is 1000 candles per request
    max_per_request = 1000
    current_end_time = None
    
    remaining_candles = total_candles_needed
    
    while remaining_candles > 0:
        limit = min(max_per_request, remaining_candles)
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        if current_end_time:
            params["endTime"] = int(current_end_time)
        
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
            
            all_data = data + all_data  # Prepend to maintain chronological order
            
            # Set end time for next request (go back further)
            first_candle_time = data[0][0]
            current_end_time = first_candle_time - 1
            
            remaining_candles -= len(data)
            
            # Be respectful to the API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error downloading historical data: {e}")
            break
    
    # Convert to DataFrame
    rows = []
    for k in all_data:
        ot, o, h, l, c, v = k[0], k[1], k[2], k[3], k[4], k[5]
        dt = pd.to_datetime(ot, unit='ms', utc=True)
        rows.append((dt, float(o), float(h), float(l), float(c), float(v)))
    
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
    return df


def _get_historical_data(symbol: str, interval: str, days: int = 30, use_cache: bool = True) -> pd.DataFrame:
    """
    Get historical data, using cached version if available.
    
    Args:
        symbol: Trading pair
        interval: Kline interval
        days: Number of days to fetch
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with historical OHLCV data
    """
    cache_path = _get_historical_data_cache_path(symbol, interval, days)
    
    # Check cache
    if use_cache and cache_path.exists():
        try:
            print(f"Loading cached historical data from {cache_path}")
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    # Download fresh data
    print(f"Downloading {days} days of {interval} data for {symbol}...")
    df = _download_historical_data(symbol, interval, days)
    
    # Save to cache
    try:
        df.to_csv(cache_path)
        print(f"Cached historical data to {cache_path}")
    except Exception as e:
        print(f"Failed to cache data: {e}")
    
    return df


@app.route("/")
def index():
    return render_template("index.html", symbol=state.symbol, interval=state.interval)


@app.route("/ml")
def ml_dashboard():
    """ML dashboard for model management and predictions."""
    return render_template("ml_dashboard.html")


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or state.symbol).lower()
    interval = data.get("interval") or state.interval
    _start_stream(symbol, interval)
    return jsonify({"ok": True, "symbol": symbol, "interval": interval})


@app.route("/api/candles")
def api_candles():
    # Return full DF as array of dicts with iso datetime
    records = []
    for dt, row in state.df.iterrows():
        # dt is a pandas Timestamp (UTC). Use its ns value to avoid tz issues.
        t_ms = int(pd.Timestamp(dt).value // 1_000_000)
        records.append({
            "t": t_ms,
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": float(row["Volume"]),
        })
    return jsonify({"symbol": state.symbol, "interval": state.interval, "candles": records})


@app.route("/events")
def sse_events():
    q = queue.Queue()
    with state.lock:
        state.subscribers.append(q)

    def gen():
        try:
            # Send a small handshake message
            yield "data: {\"type\": \"hello\"}\n\n"
            while True:
                try:
                    event = q.get(timeout=15)
                    payload = json.dumps(event)
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    # keepalive
                    yield "data: {\"type\": \"keepalive\"}\n\n"
        except GeneratorExit:
            pass
        finally:
            with state.lock:
                if q in state.subscribers:
                    state.subscribers.remove(q)

    return Response(gen(), mimetype="text/event-stream")


# ML Prediction API Endpoints

@app.route("/api/models")
def api_models():
    """List all registered models."""
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Make a prediction using the best model for current conditions or a specific model."""
    try:
        data = request.get_json() or {}
        symbol = data.get("symbol", state.symbol).lower()
        interval = data.get("interval", state.interval)
        use_ensemble = data.get("ensemble", False)
        model_id = data.get("model_id")
        
        # Get recent candle data
        with state.lock:
            if state.df.empty:
                return jsonify({"ok": False, "error": "No candle data available"}), 400
            
            candles_array = state.df[["Open", "High", "Low", "Close", "Volume"]].values
        
        # Make prediction
        selector = AggregateModelSelector()
        
        if use_ensemble:
            result = selector.get_ensemble_prediction(symbol, interval, candles_array)
        elif model_id:
            result = selector.predict_with_model(model_id, symbol, interval, candles_array, log_prediction=True)
        else:
            result = selector.predict(symbol, interval, candles_array, log_prediction=True)
        
        return jsonify({"ok": True, "prediction": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/performance")
def api_performance():
    """Get performance metrics for all models."""
    try:
        symbol = request.args.get("symbol", state.symbol).lower()
        interval = request.args.get("interval", state.interval)
        
        tracker = PerformanceTracker()
        registry = ModelRegistry()
        
        # Get all models and their performance
        models = registry.list_models()
        results = []
        
        for model in models:
            model_id = model["id"]
            perf_data = tracker.get_performance(model_id, symbol, interval)
            
            results.append({
                "model": model,
                "performance": perf_data
            })
        
        return jsonify({"ok": True, "performance": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predictions")
def api_predictions():
    """Get recent predictions with optional filters."""
    try:
        model_id = request.args.get("model_id", type=int)
        symbol = request.args.get("symbol")
        limit = request.args.get("limit", 50, type=int)
        
        logger = PredictionLogger()
        
        if model_id:
            predictions = logger.get_predictions_by_model(model_id, symbol, limit)
        else:
            # Get all recent predictions
            with logger.get_connection(logger.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?"
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                predictions = [dict(row) for row in rows]
        
        return jsonify({"ok": True, "predictions": predictions})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/train/initialize", methods=["POST"])
def api_train_initialize():
    """Initialize and register baseline models."""
    try:
        from .models.baseline import (
            MovingAverageCrossoverModel,
            MomentumModel,
            VolumeWeightedModel,
            RandomModel
        )
        
        registry = ModelRegistry()
        
        # Check if models already exist
        existing_models = registry.list_models()
        if existing_models:
            return jsonify({
                "ok": False,
                "error": f"Cannot initialize baselines. {len(existing_models)} model(s) already registered. Delete existing models first."
            }), 400
        
        models_info = []
        
        # Initialize baseline models
        baselines = [
            MovingAverageCrossoverModel(),
            MomentumModel(),
            VolumeWeightedModel(),
            RandomModel()
        ]
        
        for model in baselines:
            model_id = registry.register_model(
                name=model.name,
                model_type=model.model_type,
                version=model.version,
                model_obj=model,
                hyperparameters=model.get_hyperparameters()
            )
            models_info.append({
                "id": model_id,
                "name": model.name,
                "type": model.model_type,
                "version": model.version
            })
        
        return jsonify({"ok": True, "models": models_info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/train/models", methods=["POST"])
def api_train_models():
    """Train models on one month of historical candle data and update performance metrics."""
    try:
        from .prediction_logger import PredictionLogger
        from .performance_tracker import PerformanceTracker, detect_market_condition
        import random
        
        # Get symbol and interval
        data = request.get_json() or {}
        symbol = data.get("symbol", state.symbol).lower()
        interval = data.get("interval", state.interval)
        
        logger = PredictionLogger()
        tracker = PerformanceTracker()
        registry = ModelRegistry()
        
        # Get all registered models
        models = registry.list_models()
        if not models:
            return jsonify({"ok": False, "error": "No models registered. Initialize baselines first."}), 400
        
        # Download historical data (30 days, cached)
        print(f"Fetching one month of historical data for {symbol}...")
        hist_df = _get_historical_data(symbol, interval, days=30, use_cache=True)
        
        if hist_df.empty or len(hist_df) < 50:
            return jsonify({
                "ok": False,
                "error": f"Insufficient historical data. Got {len(hist_df)} candles, need at least 50."
            }), 400
        
        candles_array = hist_df[["Open", "High", "Low", "Close", "Volume"]].values
        
        predictions_generated = 0
        conditions_seen = set()
        
        print(f"Training on {len(candles_array)} candles from {hist_df.index[0]} to {hist_df.index[-1]}")
        
        # Generate predictions for each model using historical data
        for model_meta in models:
            model_id = model_meta['id']
            try:
                model_obj, _ = registry.load_model(model_id)
            except Exception as e:
                return jsonify({
                    "ok": False,
                    "error": f"Failed to load model {model_meta['name']}: {str(e)}"
                }), 500
            
            # Make predictions on different windows of historical data
            # Use sliding windows to generate more varied training data
            window_size = 30
            step = 5  # Step through data with stride
            
            for start_idx in range(0, len(candles_array) - window_size, step):
                subset = candles_array[start_idx : start_idx + window_size]
                
                # Detect market condition for this window
                market_condition = detect_market_condition(subset)
                
                features = model_obj.prepare_features(subset)
                prediction, confidence = model_obj.predict(features)
                
                # Log prediction
                pred_id = logger.log_prediction(
                    model_id=model_id,
                    symbol=symbol,
                    interval=interval,
                    prediction=prediction,
                    confidence=confidence,
                    features=features
                )
                
                # Simulate actual result based on next candle (for training purposes)
                if start_idx + window_size < len(candles_array):
                    next_close = candles_array[start_idx + window_size][3]  # Close price
                    current_close = candles_array[start_idx + window_size - 1][3]
                    
                    if next_close > current_close:
                        actual = "up"
                    elif next_close < current_close:
                        actual = "down"
                    else:
                        actual = "neutral"
                else:
                    actual = random.choice(["up", "down", "neutral"])
                
                logger.update_actual_result(pred_id, actual)
                
                predictions_generated += 1
                conditions_seen.add(market_condition)
        
        # Update performance metrics
        for model_meta in models:
            tracker.update_performance(
                model_meta['id'],
                symbol,
                interval,
                "mixed"  # Models trained on mixed market conditions
            )
        
        return jsonify({
            "ok": True,
            "predictions_generated": predictions_generated,
            "conditions": list(conditions_seen),
            "candles_analyzed": len(candles_array),
            "symbol": symbol,
            "interval": interval,
            "data_range": {
                "start": hist_df.index[0].isoformat(),
                "end": hist_df.index[-1].isoformat()
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Training error: {str(e)}"}), 500


def run_web(host: str = "127.0.0.1", port: int = 5000):
    # Initialize database
    init_db()
    
    # Seed last 50 candles for default symbol/interval, then start stream
    _seed_history(state.symbol, state.interval, limit=50)
    _start_stream(state.symbol, state.interval)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_web()
