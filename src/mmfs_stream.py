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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stream market ticker via Binance websockets")
    parser.add_argument("symbol", nargs="?", default=DEFAULT_SYMBOL, help="Symbol, e.g. btcusdt")
    args = parser.parse_args()

    cfg = StreamConfig(symbol=args.symbol.lower())
    asyncio.run(stream_ticker(cfg))


if __name__ == "__main__":
    main()
