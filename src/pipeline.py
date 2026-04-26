"""
pipeline.py
-----------
Orchestrates the full analysis workflow for a single entry from analyses.yaml:

  1. Fetch Google Trends (with cache)
  2. Fetch stock data (with cache)
  3. Align both series to weekly frequency
  4. Run all six statistical analyses
  5. Save artifacts to data/outputs/<name>/
  6. Render the Quarto HTML report

Artifacts written
-----------------
  data/outputs/<name>/raw_trends.csv
  data/outputs/<name>/raw_stock.csv
  data/outputs/<name>/aligned.csv
  data/outputs/<name>/analysis_results.json
  reports/outputs/<name>.html
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing the package
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fetch_trends import get_trends
from fetch_stocks import get_stock_data
from align import align_series
from analysis import run_all

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "outputs"
_REPORTS_OUT = _REPO_ROOT / "reports" / "outputs"
_TEMPLATE = _REPO_ROOT / "reports" / "template.qmd"


def run_pipeline(cfg: dict) -> None:
    """
    Execute the full pipeline for one analysis configuration dict.

    Parameters
    ----------
    cfg : dict
        One entry from config/analyses.yaml, with keys:
            name, search_terms, ticker, date_range, geo (optional), events (optional)
    """
    name: str = cfg["name"]
    search_terms: list[str] = cfg["search_terms"]
    ticker: str = cfg["ticker"]
    date_range: tuple[str, str] = tuple(cfg["date_range"])
    geo: str = cfg.get("geo", "")
    events: list[dict] = cfg.get("events", [])

    out_dir = _DATA_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting pipeline for '%s' ===", name)

    # ------------------------------------------------------------------ #
    # 1. Fetch Google Trends                                               #
    # ------------------------------------------------------------------ #
    trends_cache = out_dir / "raw_trends.csv"
    trends_df = get_trends(
        search_terms=search_terms,
        date_range=date_range,
        geo=geo,
        cache_path=trends_cache,
    )

    # ------------------------------------------------------------------ #
    # 2. Fetch stock data                                                  #
    # ------------------------------------------------------------------ #
    stock_cache = out_dir / "raw_stock.csv"
    stock_df = get_stock_data(
        ticker=ticker,
        date_range=date_range,
        cache_path=stock_cache,
    )

    # ------------------------------------------------------------------ #
    # 3. Align to weekly                                                   #
    # ------------------------------------------------------------------ #
    aligned = align_series(trends_df, stock_df)
    aligned.to_csv(out_dir / "aligned.csv")
    logger.info("Aligned data saved (%d rows)", len(aligned))

    # ------------------------------------------------------------------ #
    # 4. Run all six analyses                                              #
    # ------------------------------------------------------------------ #
    results = run_all(aligned, events=events)

    results_path = out_dir / "analysis_results.json"
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=_json_serialise)
    logger.info("Analysis results saved to %s", results_path)

    # ------------------------------------------------------------------ #
    # 5. Save analysis metadata (for Quarto header)                       #
    # ------------------------------------------------------------------ #
    meta = {
        "name": name,
        "search_terms": search_terms,
        "ticker": ticker,
        "date_range": list(date_range),
        "geo": geo or "Global",
        "n_weeks": len(aligned),
        "date_from": aligned.index.min().strftime("%Y-%m-%d"),
        "date_to": aligned.index.max().strftime("%Y-%m-%d"),
    }
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------ #
    # 6. Render Quarto report                                              #
    # ------------------------------------------------------------------ #
    _render_report(name)

    logger.info("=== Pipeline complete for '%s' ===", name)


def _render_report(analysis_name: str) -> None:
    _REPORTS_OUT.mkdir(parents=True, exist_ok=True)
    output_filename = f"{analysis_name}.html"

    # Quarto --output must be a bare filename (no path).
    # We run the command from _REPO_ROOT so relative paths in the .qmd resolve correctly,
    # and then move the rendered file to reports/outputs/.
    cmd = [
        "quarto",
        "render",
        str(_TEMPLATE),
        "-P",
        f"analysis_name:{analysis_name}",
        "--output",
        output_filename,
    ]
    logger.info("Rendering Quarto report: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))

    if result.returncode != 0:
        logger.error("Quarto render failed:\n%s\n%s", result.stdout, result.stderr)
        raise RuntimeError(
            f"Quarto render failed for '{analysis_name}'.\n{result.stderr}"
        )

    # Quarto places the output file relative to cwd (repo root) when --output
    # is a bare filename and cwd is different from the .qmd location.
    # Search both the repo root and the reports/ directory.
    rendered = None
    for candidate in [_REPO_ROOT / output_filename, _TEMPLATE.parent / output_filename]:
        if candidate.exists():
            rendered = candidate
            break

    dest = _REPORTS_OUT / output_filename
    if rendered is not None and rendered != dest:
        rendered.rename(dest)
    elif rendered is None:
        logger.warning(
            "Could not find rendered file '%s' in expected locations; "
            "it may already be at the correct path.",
            output_filename,
        )
    logger.info("Report written to %s", dest)


def _json_serialise(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (int,)):
        return int(obj)
    # numpy scalar types (int64, float64, bool_, etc.)
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
