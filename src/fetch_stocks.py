"""
fetch_stocks.py
---------------
Fetches stock price and volume data via yfinance (Yahoo Finance).

Returned columns per ticker:
  Close, Volume, returns (daily % change), volatility (21-day rolling std of returns)

Results are cached to disk to avoid redundant network calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_VOLATILITY_WINDOW = 21  # trading days (~1 calendar month)


def get_stock_data(
    ticker: str,
    date_range: tuple[str, str],
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for *ticker* and derive returns + volatility.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "NVO", "NVDA".
    date_range : tuple[str, str]
        (start_date, end_date) as "YYYY-MM-DD" strings.
    cache_path : Path | None
        If provided, save/load the raw CSV from this path.

    Returns
    -------
    pd.DataFrame
        Daily DatetimeIndex with columns:
        Open, High, Low, Close, Volume, returns, volatility
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading stock data from cache: %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    start, end = date_range
    logger.info("Downloading %s from %s to %s via yfinance", ticker, start, end)

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker={ticker!r} range={date_range}")

    # yfinance may return MultiIndex columns when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(_VOLATILITY_WINDOW).std()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info("Stock data cached to %s", cache_path)

    return df
