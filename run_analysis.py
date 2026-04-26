#!/usr/bin/env python3
"""
run_analysis.py
---------------
CLI entry point for the stock_trends analysis pipeline.

Usage
-----
  # Run a single named analysis:
  python run_analysis.py --name ozempic_nvo

  # Run all analyses defined in config/analyses.yaml:
  python run_analysis.py --all

  # List available analyses:
  python run_analysis.py --list
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ------------------------------------------------------------------ #
# Logging setup                                                        #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #
_REPO_ROOT = Path(__file__).parent
_CONFIG_PATH = _REPO_ROOT / "config" / "analyses.yaml"
_SRC = _REPO_ROOT / "src"

# Ensure src/ is importable
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import run_pipeline  # noqa: E402 (import after path setup)


# ------------------------------------------------------------------ #
# Config loading                                                       #
# ------------------------------------------------------------------ #

def load_config(path: Path = _CONFIG_PATH) -> list[dict]:
    if not path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    analyses = data.get("analyses", [])
    if not analyses:
        logger.error("No analyses defined in %s", path)
        sys.exit(1)
    return analyses


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Google Trends vs stock analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --name ozempic_nvo
  python run_analysis.py --all
  python run_analysis.py --list
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--name",
        metavar="ANALYSIS_NAME",
        help="Run a single named analysis from analyses.yaml.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run every analysis defined in analyses.yaml.",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available analysis names and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    analyses = load_config()

    if args.list:
        print("Available analyses:")
        for cfg in analyses:
            print(f"  {cfg['name']:30s}  ticker={cfg['ticker']}  terms={cfg['search_terms']}")
        return

    if args.all:
        targets = analyses
    else:
        targets = [cfg for cfg in analyses if cfg["name"] == args.name]
        if not targets:
            logger.error(
                "Analysis '%s' not found in config. Use --list to see available names.",
                args.name,
            )
            sys.exit(1)

    errors: list[str] = []
    for cfg in targets:
        try:
            run_pipeline(cfg)
        except Exception as exc:
            logger.error("Pipeline failed for '%s': %s", cfg["name"], exc, exc_info=True)
            errors.append(cfg["name"])

    if errors:
        logger.error("The following analyses failed: %s", errors)
        sys.exit(1)

    logger.info("All done.")


if __name__ == "__main__":
    main()
