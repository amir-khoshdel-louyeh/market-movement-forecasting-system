import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import random

import pandas as pd
import mplfinance as mpf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        self.start_btn = ttk.Button(controls, text="Start (coming next)", command=self._noop)
        self.start_btn.pack(side=tk.LEFT)

        # Chart area
        chart_frame = ttk.Frame(self)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial plot with sample data
        self._plot_sample()

    def _noop(self):
        pass

    def _plot_sample(self):
        df = _generate_sample_candles(120)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        mpf.plot(
            df,
            type="candle",
            style="yahoo",
            ax=ax,
            volume=True,
            warn_too_much_data=False,
        )
        ax.set_title("Sample Candlesticks – Live stream next step")
        self.canvas.draw_idle()


def launch_gui():
    root = tk.Tk()
    root.geometry("900x600")
    app = CandlesApp(master=root)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
