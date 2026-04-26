"""
analysis.py
-----------
Six statistical methods comparing a Google Trends series to stock metrics.

Every public function accepts the aligned DataFrame produced by align.py and
an optional ``term`` argument that selects which trend_<term> column to use
as the search-interest series.  When multiple trend columns are present, each
function runs against every trend column and returns a list of per-term results.

Return type convention
----------------------
Each function returns a list of dicts (one per trend term) with at minimum:
    {
        "term":           str,
        "result":         <method-specific payload>,
        "interpretation": str,   # plain-English summary
    }
Results are serialisable to JSON so they can be saved as artifacts and read
back by the Quarto report template.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, coint

logger = logging.getLogger(__name__)

_CCF_MAX_LAG = 12          # weeks either side of zero
_ROLLING_WINDOW = 52       # weeks (≈1 year)
_GRANGER_MAX_LAG = 8       # max lag orders to test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trend_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("trend_")]


def _term_name(col: str) -> str:
    return col[len("trend_"):]


# ---------------------------------------------------------------------------
# 1. Pearson & Spearman correlation (zero lag)
# ---------------------------------------------------------------------------

def zero_lag_correlation(
    df: pd.DataFrame,
    stock_col: str = "price_norm",
) -> list[dict[str, Any]]:
    """
    Pearson and Spearman correlations between each trend series and *stock_col*
    at zero lag.  Both the raw series and first-differenced (return) series are
    tested.
    """
    results = []
    for col in _trend_columns(df):
        combined = df[[col, stock_col]].dropna()
        x = combined[col].values
        y = combined[stock_col].values

        if x.std() == 0 or y.std() == 0:
            logger.warning("Constant series detected for '%s'; skipping correlation.", col)
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Also correlate with weekly returns
        if "returns" in df.columns:
            ret_combined = df[[col, "returns"]].dropna()
            ret_x = ret_combined[col].values
            ret_y = ret_combined["returns"].values
            if ret_x.std() > 0 and ret_y.std() > 0:
                pr_ret, pp_ret = stats.pearsonr(ret_x, ret_y)
                sr_ret, sp_ret = stats.spearmanr(ret_x, ret_y)
            else:
                pr_ret = pp_ret = sr_ret = sp_ret = None
        else:
            pr_ret = pp_ret = sr_ret = sp_ret = None

        term = _term_name(col)
        interp = (
            f"'{term}' vs {stock_col}: Pearson r={pearson_r:.3f} (p={pearson_p:.3e}), "
            f"Spearman r={spearman_r:.3f} (p={spearman_p:.3e})."
        )
        if pearson_p < 0.05:
            direction = "positive" if pearson_r > 0 else "negative"
            interp += f" The {direction} correlation is statistically significant."
        else:
            interp += " The correlation is not statistically significant at the 5% level."

        results.append(
            {
                "term": term,
                "result": {
                    "n": int(len(combined)),
                    "pearson_r": round(pearson_r, 4),
                    "pearson_p": round(pearson_p, 6),
                    "spearman_r": round(spearman_r, 4),
                    "spearman_p": round(spearman_p, 6),
                    "pearson_r_returns": round(pr_ret, 4) if pr_ret is not None else None,
                    "pearson_p_returns": round(pp_ret, 6) if pp_ret is not None else None,
                    "spearman_r_returns": round(sr_ret, 4) if sr_ret is not None else None,
                    "spearman_p_returns": round(sp_ret, 6) if sp_ret is not None else None,
                },
                "interpretation": interp,
            }
        )
    return results


# ---------------------------------------------------------------------------
# 2. Cross-Correlation Function (CCF)
# ---------------------------------------------------------------------------

def cross_correlation(
    df: pd.DataFrame,
    stock_col: str = "price_norm",
    max_lag: int = _CCF_MAX_LAG,
) -> list[dict[str, Any]]:
    """
    Compute cross-correlation between each trend series and *stock_col* at
    lags from -max_lag to +max_lag weeks.

    Positive lag k means the trend series *leads* the stock by k weeks.
    """
    results = []
    for col in _trend_columns(df):
        combined = df[[col, stock_col]].dropna()
        x = combined[col].values
        y = combined[stock_col].values

        # Standardise before computing CCF; skip constant series
        if x.std() == 0 or y.std() == 0:
            logger.warning("Constant series detected for '%s' in CCF; skipping.", col)
            continue
        x_std = (x - x.mean()) / x.std()
        y_std = (y - y.mean()) / y.std()
        n = len(x_std)

        lags = list(range(-max_lag, max_lag + 1))
        ccf_values = []
        for lag in lags:
            if lag >= 0:
                a, b = x_std[: n - lag] if lag > 0 else x_std, y_std[lag:] if lag > 0 else y_std
            else:
                a, b = x_std[-lag:], y_std[: n + lag]
            if len(a) < 10:
                ccf_values.append(0.0)
            else:
                ccf_values.append(float(np.corrcoef(a, b)[0, 1]))

        optimal_lag = lags[int(np.argmax(np.abs(ccf_values)))]
        optimal_corr = ccf_values[lags.index(optimal_lag)]

        term = _term_name(col)
        if optimal_lag > 0:
            lead_lag_str = f"searches lead price by {optimal_lag} week(s)"
        elif optimal_lag < 0:
            lead_lag_str = f"price leads searches by {abs(optimal_lag)} week(s)"
        else:
            lead_lag_str = "no lead/lag (peak at zero)"

        results.append(
            {
                "term": term,
                "result": {
                    "lags": lags,
                    "ccf_values": [round(v, 4) for v in ccf_values],
                    "optimal_lag": optimal_lag,
                    "optimal_corr": round(optimal_corr, 4),
                },
                "interpretation": (
                    f"'{term}': Peak CCF of {optimal_corr:.3f} at lag {optimal_lag} weeks "
                    f"({lead_lag_str})."
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 3. Granger Causality
# ---------------------------------------------------------------------------

def granger_causality(
    df: pd.DataFrame,
    stock_col: str = "returns",
    max_lag: int = _GRANGER_MAX_LAG,
) -> list[dict[str, Any]]:
    """
    Test whether each trend series Granger-causes *stock_col* (default: weekly
    returns).  Tests are run for lag orders 1 through *max_lag*.

    Note: Granger causality is a predictive concept — it does not imply true
    economic causation.
    """
    results = []
    for col in _trend_columns(df):
        combined = df[[stock_col, col]].dropna()
        if len(combined) < max_lag * 3:
            logger.warning("Too few observations for Granger test on '%s'; skipping.", col)
            continue

        per_lag: list[dict] = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_res = grangercausalitytests(
                    combined.values,
                    maxlag=max_lag,
                    verbose=False,
                )
            for lag_order, (test_dict, _) in gc_res.items():
                f_stat = test_dict["ssr_ftest"][0]
                p_val = test_dict["ssr_ftest"][1]
                per_lag.append(
                    {
                        "lag": lag_order,
                        "f_stat": round(f_stat, 4),
                        "p_value": round(p_val, 6),
                        "significant": p_val < 0.05,
                    }
                )
        except Exception as exc:
            logger.warning("Granger causality failed for '%s': %s", col, exc)
            continue

        best = min(per_lag, key=lambda d: d["p_value"])
        term = _term_name(col)
        sig_lags = [d["lag"] for d in per_lag if d["significant"]]

        if sig_lags:
            interp = (
                f"'{term}' Granger-causes {stock_col} at lag(s) {sig_lags} (best p={best['p_value']:.4f}). "
                "Search trends contain predictive information about future stock moves."
            )
        else:
            interp = (
                f"'{term}' does not Granger-cause {stock_col} at any tested lag "
                f"(best p={best['p_value']:.4f} at lag {best['lag']})."
            )

        results.append(
            {
                "term": term,
                "result": {"per_lag": per_lag, "best_lag": best},
                "interpretation": interp,
            }
        )
    return results


# ---------------------------------------------------------------------------
# 4. Rolling Correlation
# ---------------------------------------------------------------------------

def rolling_correlation(
    df: pd.DataFrame,
    stock_col: str = "price_norm",
    window: int = _ROLLING_WINDOW,
) -> list[dict[str, Any]]:
    """
    Compute a rolling Pearson correlation between each trend series and
    *stock_col* using a window of *window* weeks.

    Returns the full time series of rolling correlations so the Quarto report
    can plot them.
    """
    results = []
    for col in _trend_columns(df):
        combined = df[[col, stock_col]].dropna()
        rolling_r = combined[col].rolling(window).corr(combined[stock_col])

        recent_r = rolling_r.dropna().iloc[-4:].mean()  # average of last ~month
        term = _term_name(col)

        results.append(
            {
                "term": term,
                "result": {
                    "dates": rolling_r.index.strftime("%Y-%m-%d").tolist(),
                    "rolling_corr": [
                        round(v, 4) if pd.notna(v) else None
                        for v in rolling_r.values
                    ],
                    "window_weeks": window,
                    "recent_mean_corr": round(float(recent_r), 4) if pd.notna(recent_r) else None,
                },
                "interpretation": (
                    f"'{term}' rolling {window}-week correlation with {stock_col}: "
                    f"recent average = {recent_r:.3f}. "
                    + (
                        "Relationship has been strengthening recently."
                        if pd.notna(recent_r) and abs(recent_r) > 0.5
                        else "Relationship is moderate or weak in the recent window."
                    )
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 5. Cointegration (Engle-Granger)
# ---------------------------------------------------------------------------

def cointegration_test(
    df: pd.DataFrame,
    stock_col: str = "price_raw",
) -> list[dict[str, Any]]:
    """
    Engle-Granger cointegration test between each trend series and *stock_col*.

    Cointegration implies a long-run equilibrium relationship even when both
    series individually have unit roots (are non-stationary).
    """
    results = []
    for col in _trend_columns(df):
        combined = df[[col, stock_col]].dropna()
        if len(combined) < 30:
            logger.warning("Too few observations for cointegration test on '%s'; skipping.", col)
            continue

        try:
            coint_t, p_value, crit_values = coint(
                combined[stock_col].values,
                combined[col].values,
            )
        except Exception as exc:
            logger.warning("Cointegration test failed for '%s': %s", col, exc)
            continue

        term = _term_name(col)
        if p_value < 0.05:
            interp = (
                f"'{term}' and {stock_col} appear to be cointegrated "
                f"(p={p_value:.4f}). A long-run equilibrium relationship may exist."
            )
        else:
            interp = (
                f"No evidence of cointegration between '{term}' and {stock_col} "
                f"(p={p_value:.4f}). The series do not share a long-run equilibrium."
            )

        results.append(
            {
                "term": term,
                "result": {
                    "coint_t": round(float(coint_t), 4),
                    "p_value": round(float(p_value), 6),
                    "crit_1pct": round(float(crit_values[0]), 4),
                    "crit_5pct": round(float(crit_values[1]), 4),
                    "crit_10pct": round(float(crit_values[2]), 4),
                    "cointegrated": bool(p_value < 0.05),
                },
                "interpretation": interp,
            }
        )
    return results


# ---------------------------------------------------------------------------
# 6. Event Overlay data preparation
# ---------------------------------------------------------------------------

def event_overlay(
    df: pd.DataFrame,
    events: list[dict],
    stock_col: str = "price_raw",
) -> dict[str, Any]:
    """
    Prepare data for an event-annotated price chart.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned weekly DataFrame.
    events : list[dict]
        List of {"date": "YYYY-MM-DD", "label": "..."} dicts from the YAML config.
    stock_col : str
        Which stock column to annotate.

    Returns
    -------
    dict with keys:
        "dates"        : weekly date strings
        "price"        : stock_col values
        "events"       : annotated events with nearest weekly date and price level
    """
    series = df[stock_col].dropna()
    annotated_events = []
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        # Find the nearest available weekly date
        nearest_idx = int(pd.Series((series.index - ev_date).total_seconds().values).abs().argmin())
        nearest_date = series.index[nearest_idx]
        annotated_events.append(
            {
                "original_date": ev["date"],
                "nearest_weekly_date": nearest_date.strftime("%Y-%m-%d"),
                "price_level": round(float(series.iloc[nearest_idx]), 4),
                "label": ev["label"],
            }
        )

    return {
        "term": "event_overlay",
        "result": {
            "dates": series.index.strftime("%Y-%m-%d").tolist(),
            "price": [round(float(v), 4) for v in series.values],
            "events": annotated_events,
        },
        "interpretation": (
            f"Price series annotated with {len(events)} event(s). "
            "Use the chart to assess whether events coincide with trend + price inflection points."
        ),
    }


# ---------------------------------------------------------------------------
# Convenience: run all six methods and return a combined results dict
# ---------------------------------------------------------------------------

def run_all(
    df: pd.DataFrame,
    events: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run all six analyses and return a single serialisable dict.

    Keys: "zero_lag_correlation", "cross_correlation", "granger_causality",
          "rolling_correlation", "cointegration", "event_overlay".
    """
    return {
        "zero_lag_correlation": zero_lag_correlation(df),
        "cross_correlation": cross_correlation(df),
        "granger_causality": granger_causality(df),
        "rolling_correlation": rolling_correlation(df),
        "cointegration": cointegration_test(df),
        "event_overlay": event_overlay(df, events or []),
    }
