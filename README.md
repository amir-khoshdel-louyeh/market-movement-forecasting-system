# Market Movement Forecasting System

Starter scaffolding to stream market data via websockets (Binance) and prepare for future LSTM/Transformer modeling and a GUI.

## Quick Start

1. Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r ./requirements.txt
```

2. Stream a ticker:

```bash
# Option A: via main.py (no install needed)
python3 main.py btcusdt

# Web UI: start Flask front-end
python3 main.py --mode web

# Option B: install console script, then run
pip install -e .
mmfs-stream btcusdt
# Or use default symbol from .env
```

Optional `.env`:
```
SYMBOL=btcusdt
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
```

## Notes
- Uses Binance public websocket `@ticker` endpoint (no API key).
- Outputs normalized JSON lines for easy logging or piping.
- Next steps: persistence, feature extraction, simple GUI (PyQt/Tkinter), model training.
