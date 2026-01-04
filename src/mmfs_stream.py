import asyncio
import json
import os
from dataclasses import dataclass

import websockets
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BINANCE_WS_URL = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws")
DEFAULT_SYMBOL = os.getenv("SYMBOL", "btcusdt")


class TickerMessage(BaseModel):
    e: str  # event type
    E: int  # event time
    s: str  # symbol
    p: str | None = None  # price change
    P: str | None = None  # price change percent
    w: str | None = None  # weighted avg price
    x: str | None = None  # first trade(F)-1 price (first trade before the 24hr rolling window)
    c: str | None = None  # last price
    Q: str | None = None  # last quantity
    b: str | None = None  # best bid price
    B: str | None = None  # best bid qty
    a: str | None = None  # best ask price
    A: str | None = None  # best ask qty
    o: str | None = None  # open price
    h: str | None = None  # high price
    l: str | None = None  # low price
    v: str | None = None  # total traded base asset volume
    q: str | None = None  # total traded quote asset volume
    O: int | None = None  # statistics open time
    C: int | None = None  # statistics close time
    F: int | None = None  # first trade id
    L: int | None = None  # last trade id
    n: int | None = None  # total number of trades
class KlinePayload(BaseModel):
    t: int  # kline start time (ms)
    T: int  # kline close time (ms)
    s: str  # symbol
    i: str  # interval, e.g. 1m
    f: int  # first trade ID
    L: int  # last trade ID
    o: str  # open price
    c: str  # close price
    h: str  # high price
    l: str  # low price
    v: str  # base asset volume
    n: int  # number of trades
    x: bool  # is this kline closed?


class KlineMessage(BaseModel):
    e: str  # event type, 'kline'
    E: int  # event time
    s: str  # symbol
    k: KlinePayload



@dataclass
class StreamConfig:
    symbol: str = DEFAULT_SYMBOL

    @property
    def url(self) -> str:
        # 24hr ticker stream for a symbol
        return f"{BINANCE_WS_URL}/{self.symbol}@ticker"


async def stream_ticker(cfg: StreamConfig):
    print(f"Connecting to {cfg.url}")
    async for ws in websockets.connect(cfg.url, ping_interval=20, ping_timeout=20):
        try:
            print("Connected. Streaming messages...")
            async for msg in ws:
                data = json.loads(msg)
                try:
                    tick = TickerMessage(**data)
                except Exception:
                    # Non-standard message (e.g., control) - print raw
                    print(f"Raw: {data}")
                    continue

                print(
                    json.dumps(
                        {
                            "time": tick.E,
                            "symbol": tick.s,
                            "last": tick.c,
                            "bid": tick.b,
                            "ask": tick.a,
                            "high": tick.h,
                            "low": tick.l,
                            "vol": tick.v,
                        }
                    )
                )
        except websockets.ConnectionClosedError as e:
            print(f"Connection closed, retrying: {e}")
            await asyncio.sleep(2)
            continue
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(2)
            continue


async def stream_kline(cfg: StreamConfig, interval: str = "1m", on_kline=None, stop_event=None):
    """Stream kline data, supporting a cooperative stop via `stop_event`.

    If `on_kline` is provided, it will be called with a `KlineMessage`; otherwise print normalized JSON lines.
    """
    url = f"{BINANCE_WS_URL}/{cfg.symbol}@kline_{interval}"
    print(f"Connecting to {url}")

    while True:
        if stop_event and stop_event.is_set():
            return
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                print("Connected. Streaming klines...")
                while True:
                    if stop_event and stop_event.is_set():
                        await ws.close()
                        return
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    data = json.loads(msg)
                    try:
                        kmsg = KlineMessage(**data)
                    except Exception:
                        print(f"Raw: {data}")
                        continue

                    if on_kline is not None:
                        try:
                            on_kline(kmsg)
                        except Exception as cb_err:
                            print(f"on_kline error: {cb_err}")
                    else:
                        print(
                            json.dumps(
                                {
                                    "event_time": kmsg.E,
                                    "symbol": kmsg.s,
                                    "interval": kmsg.k.i,
                                    "open_time": kmsg.k.t,
                                    "close_time": kmsg.k.T,
                                    "open": kmsg.k.o,
                                    "high": kmsg.k.h,
                                    "low": kmsg.k.l,
                                    "close": kmsg.k.c,
                                    "volume": kmsg.k.v,
                                    "closed": kmsg.k.x,
                                }
                            )
                        )
        except websockets.ConnectionClosedError as e:
            if stop_event and stop_event.is_set():
                return
            print(f"Connection closed, retrying: {e}")
            await asyncio.sleep(2)
            continue
        except Exception as e:
            if stop_event and stop_event.is_set():
                return
            print(f"Error: {e}")
            await asyncio.sleep(2)
            continue


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stream market ticker via Binance websockets")
    parser.add_argument("symbol", nargs="?", default=DEFAULT_SYMBOL, help="Symbol, e.g. btcusdt")
    args = parser.parse_args()

    cfg = StreamConfig(symbol=args.symbol.lower())
    asyncio.run(stream_ticker(cfg))


if __name__ == "__main__":
    main()
