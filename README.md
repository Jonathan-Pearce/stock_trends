# stock_trends

Explores the relationship between Google search interest (via Google Trends) and stock prices, trading volume, and returns. For each search-term / ticker pair the pipeline fetches data, runs six statistical analyses, and renders a self-contained interactive HTML report.

---

## How it works

```
config/analyses.yaml          ← define any number of analyses here
        ↓
run_analysis.py --name <name> ← one command to run a full analysis
        ↓
src/fetch_trends.py           ← Google Trends via pytrends (cached)
src/fetch_stocks.py           ← Yahoo Finance via yfinance (cached)
src/align.py                  ← resample both to weekly, normalise
src/analysis.py               ← six statistical methods (see below)
src/pipeline.py               ← orchestrates all steps + Quarto render
        ↓
data/outputs/<name>/          ← aligned.csv, analysis_results.json, meta.json
reports/outputs/<name>.html   ← self-contained interactive HTML report
```

---

## Setup

**Prerequisites:** Python 3.10+, [Quarto](https://quarto.org/docs/get-started/) (system install)

```bash
# Install Quarto (Linux)
curl -fsSL https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-amd64.deb -o /tmp/quarto.deb
sudo dpkg -i /tmp/quarto.deb

# Install Python dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# List all configured analyses
python run_analysis.py --list

# Run a single analysis (fetches data, computes stats, renders report)
python run_analysis.py --name ozempic_nvo

# Run every analysis in config/analyses.yaml
python run_analysis.py --all
```

Output is written to:
- `data/outputs/<name>/` — raw + aligned CSVs, JSON results
- `reports/outputs/<name>.html` — self-contained interactive HTML report

### Viewing the report

The HTML report is fully self-contained (no internet connection needed once built). Three ways to open it:

**In a Codespace / dev container:**
```bash
cd reports/outputs && python -m http.server 8080
# Then open the forwarded port 8080 in your browser
```

**Locally:** Download `reports/outputs/<name>.html` and open in any browser.

**VS Code:** Right-click the file → Open With → Simple Browser.

---

## Adding a new analysis

Add one block to `config/analyses.yaml`:

```yaml
- name: my_analysis
  search_terms: ["search term 1", "search term 2"]   # up to 5 terms
  ticker: AAPL                                        # Yahoo Finance ticker
  date_range: ["2020-01-01", "2026-04-26"]
  geo: ""                                             # "" = worldwide, or e.g. "US"
  events:                                             # optional annotations
    - {date: "2022-06-01", label: "Key event"}
```

Then run:
```bash
python run_analysis.py --name my_analysis
```

No code changes needed.

---

## Statistical methods

Every analysis runs all six methods:

| # | Method | Question answered |
|---|--------|------------------|
| 1 | Pearson & Spearman correlation (zero lag) | Are they correlated at all? |
| 2 | Cross-correlation function (CCF), lags −12 to +12 weeks | Do searches *lead* price by N weeks? |
| 3 | Granger causality (F-test, up to 8 lag orders) | Does search volume help predict future returns? |
| 4 | Rolling 52-week correlation | Does the relationship strengthen or weaken over time? |
| 5 | Engle-Granger cointegration | Is there a long-run equilibrium between the series? |
| 6 | Event overlay | Annotate key events to explain regime changes |

---

## Pre-configured analyses

| Name | Search terms | Ticker | Notes |
|------|-------------|--------|-------|
| `ozempic_nvo` | Ozempic, semaglutide, Novo Nordisk, weight loss drug | NVO | Flagship example — tracks semaglutide's cultural rise |
| `chatgpt_nvda` | ChatGPT, artificial intelligence, NVIDIA GPU, large language model | NVDA | AI search frenzy vs NVIDIA's price appreciation |

---

## Data sources & caveats

- **Google Trends (pytrends):** Unofficial API. Values are *relative* interest (0–100, normalised within the query window), not absolute search volumes. Rate limits apply — the pipeline retries with exponential backoff and caches all raw fetches to disk.
- **Stock data (yfinance / Yahoo Finance):** Free, adjusted daily OHLCV. Resampled to weekly for alignment with Trends.
- **Caching:** Raw fetch results are saved to `data/outputs/<name>/raw_trends.csv` and `raw_stock.csv`. Delete either file to force a fresh fetch on the next run.

