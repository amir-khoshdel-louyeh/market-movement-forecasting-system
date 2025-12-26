import asyncio
import argparse

from src.mmfs_stream import StreamConfig, stream_ticker, DEFAULT_SYMBOL
from src.mmfs_gui import launch_gui
from src.mmfs_web import run_web


def main():
    parser = argparse.ArgumentParser(
        description="Run MMFS in CLI (stream) or GUI (candles) mode"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "gui", "web"],
        help="Choose 'cli' for streaming, 'gui' for Tk viewer, or 'web' for Flask UI",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Symbol for streaming, e.g. btcusdt (default from .env)",
    )

    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        print("Select mode: [1] CLI stream  [2] Tk GUI  [3] Web (Flask)")
        choice = input("Enter 1, 2 or 3: ").strip()
        if choice == "1":
            mode = "cli"
        elif choice == "2":
            mode = "gui"
        elif choice == "3":
            mode = "web"
        else:
            print("Invalid choice. Defaulting to CLI stream.")
            mode = "cli"

    if mode == "cli":
        symbol = args.symbol.lower() if args.symbol else DEFAULT_SYMBOL
        cfg = StreamConfig(symbol=symbol)
        asyncio.run(stream_ticker(cfg))
    elif mode == "gui":
        launch_gui()
    else:
        print("Starting Flask web UI on http://127.0.0.1:5000 …")
        run_web()


if __name__ == "__main__":
    main()
