import json
import threading
import asyncio
import queue
import time
from datetime import datetime
from typing import List

import pandas as pd
from flask import Flask, Response, request, jsonify, render_template

from .mmfs_stream import StreamConfig, stream_kline, DEFAULT_SYMBOL


class WebState:
    def __init__(self):
        self.df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])  # Date index
        self.df.index.name = "Date"
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()
        self.stream_thread: threading.Thread | None = None
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
        # If same config, do nothing; else we could implement restart (simple: ignore for now)
        return

    state.symbol = symbol
    state.interval = interval

    def on_kline(kmsg):
        k = kmsg.k
        dt = datetime.fromtimestamp(k.t / 1000)
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
        asyncio.run(stream_kline(cfg, interval=interval, on_kline=on_kline))

    state.stream_thread = threading.Thread(target=run, daemon=True)
    state.stream_thread.start()


@app.route("/")
def index():
    return render_template("index.html", symbol=state.symbol, interval=state.interval)


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
        records.append({
            "t": int(dt.timestamp() * 1000),
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


def run_web(host: str = "127.0.0.1", port: int = 5000):
    # Start default stream on startup
    _start_stream(state.symbol, state.interval)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_web()
