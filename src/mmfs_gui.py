import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import random
import threading
import asyncio
import queue
import time
import pandas as pd
import mplfinance as mpf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .mmfs_stream import StreamConfig, stream_kline, DEFAULT_SYMBOL


def _generate_sample_candles(n: int = 100) -> pd.DataFrame:
    base = datetime.now() - timedelta(minutes=n)
    dates = [base + timedelta(minutes=i) for i in range(n)]

    price = 40000.0
    rows = []
    for dt in dates:
        change = random.uniform(-50, 50)
        o = price
        h = o + abs(random.uniform(0, 60))
        l = o - abs(random.uniform(0, 60))
        c = o + change
        v = random.uniform(10, 100)
        price = c
        rows.append((dt, o, h, l, c, v))

    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
    return df


class CandlesApp(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("MMFS – Candlestick Viewer (stub)")
        self.pack(fill=tk.BOTH, expand=True)

        # DataFrame holds OHLCV indexed by kline open time
        self.df = _generate_sample_candles(60)
        self._queue: queue.Queue = queue.Queue()
        self._stream_thread: threading.Thread | None = None
        self.max_rows = 300
        self._last_plot_ms = 0
        self._plot_interval_ms = 1000

        # Top controls
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(controls, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_var = tk.StringVar(value="btcusdt")
        ttk.Entry(controls, textvariable=self.symbol_var, width=12).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(controls, text="Interval:").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value="1m")
        ttk.Combobox(controls, textvariable=self.interval_var, values=["1m", "5m", "15m", "1h"], width=6).pack(
            side=tk.LEFT, padx=(4, 12)
        )

        self.start_btn = ttk.Button(controls, text="Start", command=self._start_stream)
        self.start_btn.pack(side=tk.LEFT)

        # Chart area
        chart_frame = ttk.Frame(self)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Plot styling
        self._mc = mpf.make_marketcolors(
            up='tab:green', down='tab:red', edge='inherit', wick='inherit', volume='in'
        )
        self._style = mpf.make_mpf_style(
            marketcolors=self._mc,
            gridstyle='--',
            facecolor='white',
            figcolor='white',
            rc={'axes.grid': True}
        )

        # Initial plot with sample data
        self._replot()

    def _noop(self):
        pass

    def _replot(self):
        self.figure.clear()
        ax_price = self.figure.add_subplot(2, 1, 1)
        ax_vol = self.figure.add_subplot(2, 1, 2, sharex=ax_price)
        mpf.plot(
            self.df,
            type="candle",
            style=self._style,
            ax=ax_price,
            volume=ax_vol,
            warn_too_much_data=1000,
            datetime_format='%H:%M',
            xrotation=15,
            update_width_config={'candle_linewidth': 0.7, 'candle_width': 0.6, 'volume_width': 0.8},
        )
        # Autoscale with small margins for better readability
        ax_price.margins(y=0.05)
        ax_vol.margins(y=0.10)
        ax_price.relim(); ax_price.autoscale_view()
        ax_vol.relim(); ax_vol.autoscale_view()

        sym = (self.symbol_var.get() or DEFAULT_SYMBOL).upper()
        iv = self.interval_var.get() or '1m'
        ax_price.set_title(f"{sym} – {iv} Candlesticks")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _start_stream(self):
        # Disable button and start background streaming
        self.start_btn.config(text="Streaming…", state=tk.DISABLED)
        symbol = self.symbol_var.get().lower() or DEFAULT_SYMBOL
        interval = self.interval_var.get() or "1m"

        def on_kline(kmsg):
            self._queue.put(kmsg)

        def run():
            cfg = StreamConfig(symbol=symbol)
            asyncio.run(stream_kline(cfg, interval=interval, on_kline=on_kline))

        self._stream_thread = threading.Thread(target=run, daemon=True)
        self._stream_thread.start()
        self.after(500, self._drain_queue)

    def _drain_queue(self):
        updated = False
        closed_arrived = False
        try:
            while True:
                kmsg = self._queue.get_nowait()
                k = kmsg.k
                # Use kline open time as index
                dt = datetime.fromtimestamp(k.t / 1000)
                self._upsert_kline(dt, k)
                if k.x:
                    closed_arrived = True
                updated = True
        except queue.Empty:
            pass

        if updated:
            # Ensure datetime index is sorted
            self.df = self.df.sort_index()
            # Keep only most recent rows
            if len(self.df) > self.max_rows:
                self.df = self.df.tail(self.max_rows)
            now_ms = int(time.time() * 1000)
            # Replot when a candle closes or on throttle interval
            if closed_arrived or (now_ms - self._last_plot_ms) >= self._plot_interval_ms:
                self._replot()
                self._last_plot_ms = now_ms

        # Keep polling while streaming
        if self._stream_thread and self._stream_thread.is_alive():
            self.after(500, self._drain_queue)

    def _upsert_kline(self, dt: datetime, k):
        """Update or insert OHLCV for a given kline.

        - Preserve the original Open when updating an in-progress candle.
        - Update High/Low with extremes, Close with latest, Volume with latest cumulative.
        """
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
                "Volume": new["Volume"],  # Binance kline volume is cumulative within candle
            }
            self.df.loc[dt] = merged
        else:
            self.df.loc[dt] = new


def launch_gui():
    root = tk.Tk()
    root.geometry("900x600")
    app = CandlesApp(master=root)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
