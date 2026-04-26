"""
fetch_trends.py
---------------
Fetches Google Trends data via the unofficial pytrends API.

Key behaviour:
- Returns weekly interest-over-time data normalised 0-100 by Google.
  The values are *relative* within the query window, not absolute search volumes.
- Up to 5 search terms can be compared in a single call because they share
  the same normalisation baseline.
- Google silently returns monthly data when the requested window exceeds ~270
  days.  To preserve weekly resolution across multi-year ranges this module
  splits the request into overlapping 6-month chunks and re-scales each chunk
  so the series are stitched on a common baseline (using a 4-week overlap
  period to compute a scaling factor between adjacent chunks).
- Results are cached to disk so re-running an analysis avoids re-fetching
  and respects Google's rate limits.
- Exponential backoff is applied on ResponseError / TooManyRequestsError.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError

logger = logging.getLogger(__name__)

_MAX_TERMS_PER_REQUEST = 5
_INITIAL_BACKOFF_S = 60      # Google rate-limits aggressively; start at 60 s
_MAX_RETRIES = 5
_CHUNK_MONTHS = 6            # fetch 6-month windows to guarantee weekly data
_OVERLAP_WEEKS = 4           # overlap between consecutive chunks for re-scaling
_INTER_REQUEST_SLEEP_S = 2   # polite pause between chunk requests


def get_trends(
    search_terms: list[str],
    date_range: tuple[str, str],
    geo: str = "",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch weekly Google Trends interest-over-time for *search_terms*.

    For ranges longer than ~270 days the fetch is split into overlapping
    6-month chunks and the chunks are re-scaled onto a single 0-100 baseline
    before being concatenated.  This ensures weekly resolution regardless of
    how long the overall date range is.

    Parameters
    ----------
    search_terms : list[str]
        Up to 5 search terms.  All terms share the same 0-100 normalisation
        baseline within each chunk; cross-chunk values are rescaled.
    date_range : tuple[str, str]
        (start_date, end_date) as "YYYY-MM-DD" strings.
    geo : str
        ISO 3166-1 alpha-2 country code, e.g. "US", "GB".
        Empty string (default) means worldwide.
    cache_path : Path | None
        If provided, save/load the stitched weekly CSV from this path.

    Returns
    -------
    pd.DataFrame
        Weekly DatetimeIndex, one column per search term named after the term.
        Values are 0-100 relative interest (rescaled across chunks).
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

    start = pd.Timestamp(date_range[0])
    end   = pd.Timestamp(date_range[1])

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))

    chunks = _build_chunks(start, end)
    logger.info(
        "Fetching Google Trends in %d chunk(s) of ~%d months each "
        "(weekly resolution requires chunking for long date ranges).",
        len(chunks), _CHUNK_MONTHS,
    )

    raw_chunks: list[pd.DataFrame] = []
    for i, (cs, ce) in enumerate(chunks):
        timeframe = f"{cs.strftime('%Y-%m-%d')} {ce.strftime('%Y-%m-%d')}"
        logger.info("  Chunk %d/%d: %s", i + 1, len(chunks), timeframe)
        chunk_df = _fetch_with_backoff(pytrends, search_terms, timeframe, geo)
        chunk_df = chunk_df.drop(columns=["isPartial"], errors="ignore")
        chunk_df.index = pd.to_datetime(chunk_df.index)
        chunk_df.index.name = "date"
        raw_chunks.append(chunk_df)
        if i < len(chunks) - 1:
            time.sleep(_INTER_REQUEST_SLEEP_S)

    df = _stitch_chunks(raw_chunks)

    # Clip to the originally requested range
    df = df.loc[start:end]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info("Trends cached to %s (%d weekly rows)", cache_path, len(df))

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split [start, end] into overlapping ~6-month windows.

    Each window overlaps the next by _OVERLAP_WEEKS weeks so we have a shared
    region to compute a rescaling factor.
    """
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    chunk_start = start
    overlap_delta = timedelta(weeks=_OVERLAP_WEEKS)
    chunk_delta   = pd.DateOffset(months=_CHUNK_MONTHS)

    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_delta, end)
        chunks.append((chunk_start, chunk_end))
        if chunk_end >= end:
            break
        # Next chunk starts _OVERLAP_WEEKS before the current end
        chunk_start = chunk_end - overlap_delta

    return chunks


def _stitch_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate overlapping weekly chunks onto a single rescaled 0-100 index.

    Strategy: treat the first chunk as the reference.  For each subsequent
    chunk, find the overlapping rows with the previous (already stitched)
    result, compute a per-column median scale factor, and rescale the new
    chunk before appending.  Finally clip everything to [0, 100].
    """
    if len(chunks) == 1:
        return chunks[0]

    stitched = chunks[0].copy()

    for chunk in chunks[1:]:
        overlap_idx = stitched.index.intersection(chunk.index)
        if len(overlap_idx) >= 2:
            ref  = stitched.loc[overlap_idx]
            new  = chunk.loc[overlap_idx]
            # Per-column scale factor: ref / new (avoid div-by-zero)
            scale = (ref + 1).div(new + 1).median()
            chunk = chunk.multiply(scale).clip(0, 100)
        else:
            logger.warning(
                "Chunk overlap too small (%d rows); rescaling may be inaccurate.",
                len(overlap_idx),
            )

        # Append only the non-overlapping tail of the new chunk
        new_rows = chunk.loc[chunk.index.difference(stitched.index)]
        stitched = pd.concat([stitched, new_rows]).sort_index()

    return stitched.clip(0, 100)


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

