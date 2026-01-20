# Market Movement Forecasting System

Real-time Binance kline stream + LSTM/Transformer ensemble with auto-resolving predictions.

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or pip install -e .[dev] for tests
cp .env.example .env  # set SYMBOL, BINANCE_WS_URL, API_TOKEN (optional)
python main.py --mode web   # -> http://127.0.0.1:5000
python main.py --mode cli   # ticker stream
mmfs-stream btcusdt
```

## Docker
```bash
docker build -t mmfs .
docker compose up
# or: docker run -p 5000:5000 --env-file .env -v ./data:/app/data mmfs
```

## API
- `GET /` `GET /ml` `GET /api/candles` `GET /events` (SSE)
- `POST /start` `POST /api/predict` `POST /api/train/initialize` `POST /api/train/models` `POST /api/backtest` `POST /api/resolve` require `X-API-Token` or `Authorization: Bearer` if `API_TOKEN` set in `.env`
- `GET /api/models` `GET /api/performance` `GET /api/predictions`

## Training & Backtest
`POST /api/train/models` uses threshold `0.2%` labeling, `80/20` train/test split via `src/backtest.py:65` (`run_backtest`), fits deep models only on train. `POST /api/backtest` dry-run without DB writes. Live pending predictions auto-resolve on next closed candle `src/mmfs_web.py:89`.

## Tests & CI
```bash
pytest -q
```
CI: `.github/workflows/ci.yml` runs `pytest` and `docker build` on push/PR.

## Env
See `.env.example`: `SYMBOL`, `BINANCE_WS_URL`, `API_TOKEN`, `FLASK_SECRET`, `DATABASE_PATH`. `.gitignore:7` keeps `data/*.db`/`*.pkl` out but preserves `data/.gitkeep`.

## Auth
If `API_TOKEN` empty, no auth. If set, all mutating `POST /api/*` and `/start` require token.
