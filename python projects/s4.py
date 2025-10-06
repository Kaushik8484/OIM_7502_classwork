# in_class_4_str.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


plt.style.use("seaborn-v0_8")  # nicer defaults


@dataclass
class Stock:
    symbol: str
    start_date: Optional[str] = None   # ISO 'YYYY-MM-DD'; default = last 365 days
    end_date: Optional[str] = None
    data: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        today = pd.Timestamp.today().floor("D")
        if self.end_date is None:
            self.end_date = today.strftime("%Y-%m-%d")
        if self.start_date is None:
            self.start_date = (today - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        self.get_data()  # populate self.data on construction

    # 1) get_data -------------------------------------------------------------
    def get_data(self) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for {self.symbol} between "
                             f"{self.start_date} and {self.end_date}.")

        # Keep just what we need and standardize names
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns={"Close": "close"})
        df = df[["close"]].copy()

        self.data = df
        # enrich with return calculations
        self.calc_returns()
        return self.data

    # 2) calc_returns ---------------------------------------------------------
    def calc_returns(self) -> None:
        """
        Adds:
          - price_change: day-over-day change in close price (close_t - close_{t-1})
          - instant_return: log return ln(close_t) - ln(close_{t-1}), rounded to 4 d.p.
        """
        if self.data.empty:
            raise RuntimeError("Call get_data() before calc_returns().")

        s = self.data["close"]
        self.data["price_change"] = s.diff()
        self.data["instant_return"] = np.log(s).diff().round(4)

    # 3) plot_return_dist -----------------------------------------------------
    def plot_return_dist(self, bins: Optional[int] = None) -> None:
        """
        Histogram of instantaneous (log) returns with a KDE-like smoothed line.
        """
        r = self.data["instant_return"].dropna()
        if r.empty:
            raise RuntimeError("No returns to plot. Did you call get_data()?")

        # heuristic for a reasonable number of bins
        if bins is None:
            bins = max(15, int(np.sqrt(len(r))))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(r, bins=bins, color="#7fc8f8", edgecolor="#1f4e79", alpha=0.85, density=True)
        # simple smoothing via rolling mean on a fine histogram grid
        # (keeps us dependency-light vs seaborn)
        counts, edges = np.histogram(r, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        if len(counts) > 3:
            smooth = pd.Series(counts).rolling(3, center=True, min_periods=1).mean()
            ax.plot(centers, smooth, color="#0b6e4f", lw=2, label="smoothed density")

        mu, sigma = r.mean(), r.std(ddof=1)
        ax.set_title(f"{self.symbol} · Instantaneous Return Distribution\n"
                     f"μ={mu:.4f}, σ={sigma:.4f}")
        ax.set_xlabel("log return")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # 4) plot_performance -----------------------------------------------------
    def plot_performance(self) -> None:
        s = self.data["close"].dropna()
        if s.empty:
            raise RuntimeError("No prices to plot. Did you call get_data()?")

        perf_pct = (s / s.iloc[0] - 1.0) * 100.0

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(perf_pct.index, perf_pct.values, color="#2a9d8f", lw=2)
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_title(f"{self.symbol} · Performance (% gain/loss)\n"
                     f"{self.start_date} → {self.end_date}")
        ax.set_ylabel("performance (%)")
        ax.set_xlabel("date")
        ax.grid(True, alpha=0.3)
        # annotate final value
        ax.annotate(f"{perf_pct.iloc[-1]:.2f}%",
                    xy=(perf_pct.index[-1], perf_pct.iloc[-1]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2a9d8f"))
        plt.tight_layout()
        plt.show()


# quick self-test --------------------------------------------------------------
if __name__ == "__main__":
    # change symbol/dates as you like; defaults to the last 365 days ending today
    stk = Stock("AAPL")
    # access the data attribute
    print(stk.data.head())
    # generate the two plots
    stk.plot_return_dist()
    stk.plot_performance()
