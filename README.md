# Market Movement Forecasting System

Real-time Binance kline stream + LSTM/Transformer ensemble with threshold backtest and auto-resolving predictions.

## Features
- **Streaming**: Binance `wss://stream.binance.com:9443/ws` `@ticker`/`@kline` via `src/mmfs_stream.py:74` (`stream_ticker`) and `src/mmfs_stream.py:112` (`stream_kline`), REST seeding `src/mmfs_web.py:149` (`_seed_history`) and 30-day history cache `src/mmfs_web.py:173` (`_get_historical_data`)
- **Persistence**: SQLite `data/mmfs.db` (`src/database.py:35` `init_db` creates `models`, `predictions`, `model_performance` with indexes), file store `data/models/*.pkl` via `src/model_registry.py:28` (`ModelRegistry`), `.gitignore:7` ignores `data/*.db`/`*.pkl` but keeps `data/.gitkeep`
- **Feature Extraction**: `src/models/base.py:35` (`BaseModel.prepare_features` → `last_close`, `price_change_pct`, `ma5`/`ma20`, `volume_ma5`), deep models add `src/models/lstm.py:87`/`src/models/transformer.py:107` sequence z-score normalization (`seq_len=30`) + `raw_sequence`
- **GUI**: Flask `src/mmfs_web.py:61` + Plotly candlesticks `src/templates/index.html:1`/`src/static/js/app.js:76` with SSE `src/mmfs_web.py:331` (`/events`), symbol/interval selector, and ML dashboard `src/templates/ml_dashboard.html:1` (init/train/predict/ensemble/leaderboard)
- **Model Training**: 4 baselines `src/models/baseline.py:9` (`MA Crossover`, `Momentum`, `VolumeWeighted`, `Random`) + `src/models/lstm.py:16` (`LSTMNet`/`LSTMModel`) + `src/models/transformer.py:15` (`TransformerNet`/`TransformerModel`) with `fit()`; registry `src/models/__init__.py:1` exports all; `POST /api/train/initialize` `src/mmfs_web.py:458` and `POST /api/train/models` `src/mmfs_web.py:513` run `src/backtest.py:45` (`run_backtest` `80/20` split, `threshold=0.2%` `label_next` `src/backtest.py:11`, dry `POST /api/backtest` `src/mmfs_web.py:589`, auto-resolve on closed candle `src/mmfs_web.py:89`)

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or pip install -e .[dev]
cp .env.example .env  # SYMBOL, BINANCE_WS_URL, API_TOKEN
python main.py --mode web   # http://127.0.0.1:5000  (or --mode cli)
mmfs-stream btcusdt
```

## Docker
```bash
docker build -t mmfs .              # Dockerfile:1 python:3.11-slim
docker compose up                   # docker-compose.yml:1
docker run -p 5000:5000 --env-file .env -v ./data:/app/data mmfs
```

## API
| Method | Path | Auth | Desc |
|---|---|---|---|
| `GET` | `/`, `/ml`, `/api/candles`, `/events` | - | UI + SSE |
| `GET` | `/api/models`, `/api/performance`, `/api/predictions` | - | read-only |
| `POST` | `/start`, `/api/predict`, `/api/train/initialize`, `/api/train/models`, `/api/backtest`, `/api/resolve` | `X-API-Token` or `Bearer` if `API_TOKEN` set | `src/auth.py:1` `require_auth`, `src/mmfs_web.py:304` |

## Architecture
`Binance WS/REST → WebState.df (500 rows) → SSE → Plotly` + `PredictionLogger` `src/prediction_logger.py:10` ↔ `PerformanceTracker` `src/performance_tracker.py:58` (`volatile`/`trending_up`/`trending_down`/`range`) ↔ `AggregateModelSelector` `src/aggregate_selector.py:12` (best + `get_ensemble_prediction` `src/aggregate_selector.py:202`).

## Project Structure
```
src/mmfs_stream.py  # websockets
src/mmfs_web.py     # Flask + SSE + backtest/resolve
src/database.py     # sqlite + get_connection
src/model_registry.py # pickle registry
src/backtest.py     # label_next + run_backtest
src/auth.py         # token gate
src/models/base.py, baseline.py, lstm.py, transformer.py
src/templates/ static/
tests/              # pytest 13 tests
```

## Tests & CI
```bash
pytest -q  # 13 tests: db, models, backtest, logger, api, auth
```
GitHub Actions `.github/workflows/ci.yml:1` runs `pytest` and `docker build` on push/PR.

## Env
`SYMBOL=btcusdt` `BINANCE_WS_URL=wss://stream.binance.com:9443/ws` `API_TOKEN=` (empty = no auth) `FLASK_SECRET` `DATABASE_PATH=data/mmfs.db` — see `.env.example:1`. If `API_TOKEN` set, all mutating `POST /api/*` require token.
