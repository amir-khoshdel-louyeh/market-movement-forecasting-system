import asyncio
import argparse

from src.mmfs_stream import StreamConfig, stream_ticker, DEFAULT_SYMBOL
from src.mmfs_gui import launch_gui


def main():
    parser = argparse.ArgumentParser(
        description="Run MMFS in CLI (stream) or GUI (candles) mode"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "gui"],
        help="Choose 'cli' for streaming or 'gui' for candlestick viewer",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Symbol for streaming, e.g. btcusdt (default from .env)",
    )

    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        print("Select mode: [1] CLI stream  [2] GUI candles")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            mode = "cli"
        elif choice == "2":
            mode = "gui"
        else:
            print("Invalid choice. Defaulting to CLI stream.")
            mode = "cli"

    if mode == "cli":
        symbol = args.symbol.lower() if args.symbol else DEFAULT_SYMBOL
        cfg = StreamConfig(symbol=symbol)
        asyncio.run(stream_ticker(cfg))
    else:
        launch_gui()


if __name__ == "__main__":
    main()
