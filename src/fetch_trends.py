"""
fetch_trends.py
---------------
Fetches Google Trends data via the unofficial pytrends API.

Key behaviour:
- Returns weekly interest-over-time data normalised 0-100 by Google.
  The values are *relative* within the query window, not absolute search volumes.
- Up to 5 search terms can be compared in a single call because they share
  the same normalisation baseline.
- Results are cached to disk so re-running an analysis avoids re-fetching
  and respects Google's rate limits.
- Exponential backoff is applied on ResponseError / TooManyRequestsError.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError

logger = logging.getLogger(__name__)

_MAX_TERMS_PER_REQUEST = 5
_INITIAL_BACKOFF_S = 60  # Google rate-limits aggressively; start at 60 s
_MAX_RETRIES = 5


def get_trends(
    search_terms: list[str],
    date_range: tuple[str, str],
    geo: str = "",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch weekly Google Trends interest-over-time for *search_terms*.

    Parameters
    ----------
    search_terms : list[str]
        Up to 5 search terms.  All terms share the same 0-100 normalisation
        baseline, so they are comparable within the result.
    date_range : tuple[str, str]
        (start_date, end_date) as "YYYY-MM-DD" strings.
    geo : str
        ISO 3166-1 alpha-2 country code, e.g. "US", "GB".
        Empty string (default) means worldwide.
    cache_path : Path | None
        If provided, save/load the raw CSV from this path.

    Returns
    -------
    pd.DataFrame
        Weekly DatetimeIndex, one column per search term named after the term.
        Values are 0-100 relative interest.
    """
    if len(search_terms) > _MAX_TERMS_PER_REQUEST:
        raise ValueError(
            f"pytrends supports at most {_MAX_TERMS_PER_REQUEST} terms per request; "
            f"got {len(search_terms)}: {search_terms}"
        )

    if cache_path is not None and cache_path.exists():
        logger.info("Loading trends from cache: %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    start, end = date_range
    timeframe = f"{start} {end}"

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))

    df = _fetch_with_backoff(pytrends, search_terms, timeframe, geo)

    # Drop the isPartial column Google sometimes appends
    df = df.drop(columns=["isPartial"], errors="ignore")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info("Trends cached to %s", cache_path)

    return df


def _fetch_with_backoff(
    pytrends: TrendReq,
    kw_list: list[str],
    timeframe: str,
    geo: str,
) -> pd.DataFrame:
    backoff = _INITIAL_BACKOFF_S
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            pytrends.build_payload(kw_list, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df.empty:
                raise ValueError(
                    f"No Google Trends data returned for terms={kw_list} "
                    f"timeframe={timeframe} geo={geo!r}"
                )
            return df
        except ResponseError as exc:
            if attempt == _MAX_RETRIES:
                raise
            logger.warning(
                "Google Trends rate-limit hit (attempt %d/%d). "
                "Waiting %d s before retry. Error: %s",
                attempt,
                _MAX_RETRIES,
                backoff,
                exc,
            )
            time.sleep(backoff)
            backoff *= 2
