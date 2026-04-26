"""
align.py
--------
Merges Google Trends and stock DataFrames onto a common weekly frequency.

Output column naming convention
--------------------------------
  trend_<term>   : 0-100 normalised Google Trends interest for each search term
  price_raw      : week-end adjusted close price (original currency)
  price_norm     : price_raw min-max normalised to 0-100 over the full period
  volume         : sum of daily trading volume over the week
  returns        : mean of daily returns over the week
  volatility     : mean of daily realised volatility over the week
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_TREND_FILL_WEEKS = 2  # forward-fill at most 2 consecutive missing weeks


def align_series(
    trends_df: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align Google Trends (weekly) and stock (daily) data onto a shared weekly index.

    Parameters
    ----------
    trends_df : pd.DataFrame
        Output of ``fetch_trends.get_trends()``.
        Weekly DatetimeIndex, columns = search term names, values 0-100.
    stock_df : pd.DataFrame
        Output of ``fetch_stocks.get_stock_data()``.
        Daily DatetimeIndex with columns: Close, Volume, returns, volatility.

    Returns
    -------
    pd.DataFrame
        Weekly DatetimeIndex (Sunday week-start from pytrends convention).
        Columns: trend_<term>, price_raw, price_norm, volume, returns, volatility.
        Rows with no stock data at all are dropped.
    """
    # ------------------------------------------------------------------ #
    # 1.  Resample stock data to weekly                                    #
    # ------------------------------------------------------------------ #
    weekly_stock = stock_df.resample("W").agg(
        {
            "Close": "last",       # week-end close
            "Volume": "sum",
            "returns": "mean",
            "volatility": "mean",
        }
    )
    weekly_stock = weekly_stock.rename(
        columns={"Close": "price_raw", "Volume": "volume"}
    )

    # ------------------------------------------------------------------ #
    # 2.  Forward-fill gaps in trends (up to _MAX_TREND_FILL_WEEKS weeks) #
    # ------------------------------------------------------------------ #
    trends_filled = trends_df.ffill(limit=_MAX_TREND_FILL_WEEKS)

    # Rename trend columns to trend_<term>
    trend_cols = {col: f"trend_{col}" for col in trends_filled.columns}
    trends_filled = trends_filled.rename(columns=trend_cols)

    # ------------------------------------------------------------------ #
    # 3.  Align indexes                                                    #
    # ------------------------------------------------------------------ #
    # pytrends uses Sunday as the week label; resample("W") also uses Sunday.
    # Inner join: keep only weeks where both datasets have data.
    merged = trends_filled.join(weekly_stock, how="inner")

    # Drop weeks where price is missing (e.g. market closures spanning full week)
    merged = merged.dropna(subset=["price_raw"])

    if merged.empty:
        raise ValueError(
            "Aligned DataFrame is empty — check that the date ranges overlap "
            "between the trends and stock data."
        )

    # ------------------------------------------------------------------ #
    # 4.  Normalise price to 0-100 for visual overlay                     #
    # ------------------------------------------------------------------ #
    price_min = merged["price_raw"].min()
    price_max = merged["price_raw"].max()
    if price_max == price_min:
        merged["price_norm"] = 50.0
    else:
        merged["price_norm"] = (
            (merged["price_raw"] - price_min) / (price_max - price_min) * 100
        )

    # Reorder columns for clarity
    trend_columns = [c for c in merged.columns if c.startswith("trend_")]
    other_columns = ["price_raw", "price_norm", "volume", "returns", "volatility"]
    merged = merged[trend_columns + other_columns]

    logger.info(
        "Aligned dataset: %d weekly observations from %s to %s",
        len(merged),
        merged.index.min().date(),
        merged.index.max().date(),
    )
    return merged
