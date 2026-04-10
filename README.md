# Detecting Stock Market Crashes with Topological Data Analysis

Uses [Topological Data Analysis (TDA)](https://en.wikipedia.org/wiki/Topological_data_analysis) to build a crash indicator for the S&P 500. The core idea is to embed the price time series into higher-dimensional space via Takens' embedding, compute persistence diagrams using the Vietoris-Rips filtration, and then measure how rapidly the topological signature changes between consecutive windows — the *homological derivative*. Large values of this derivative coincide with historical crash periods.

For a detailed walkthrough, see the accompanying [blog post](https://towardsdatascience.com/detecting-stock-market-crashes-with-topological-data-analysis-7d5dd98abe42).

## Project structure

```
.
├── main.py                    # Full pipeline: data → embeddings → TDA → plots
├── homological_derivative.py  # Custom sklearn-compatible transformer
├── plotting.py                # Crash detection and comparison plots
├── requirements.txt
└── images/                    # Output plots (created on first run)
```

## Method overview

1. **Data** — Daily or hourly price data from Yahoo Finance, converted to log-returns.
2. **Takens embedding** — The scalar log-return series is embedded into a higher-dimensional point cloud using a time-delay embedding (auto-searched or fixed at `dimension=3`, `time_delay=2`).
3. **Sliding windows** — A sliding window (size 31 days, stride 1) turns the embedded series into a time series of point clouds at full temporal resolution.
4. **Persistence diagrams** — Vietoris-Rips persistent homology is applied to each point cloud, tracking $H_0$ and $H_1$ generators.
5. **Homological derivatives** — Four complementary metrics measure topological change between consecutive diagrams:
   - **Landscape distance** — L2 norm of persistence landscape differences
   - **Betti curve distance** — L2 norm of Betti curve differences
   - **Persistent entropy distance** — L2 norm of entropy vector differences
   - **Wasserstein distance** — optimal-transport distance between consecutive diagrams (no vectorisation step)
6. **Crash probability** — Each metric is normalised to $[0, 1]$ using a causal rolling window (252-day trailing min/max, no look-ahead bias). Values above the threshold flag potential crash periods.
7. **CSD indicators** — Rolling variance and lag-1 autocorrelation (Critical Slowing Down indicators) are computed alongside TDA and plotted for comparison.
8. **Evaluation** — Precision, recall, and F1 are printed against known historical crashes using a configurable lead window (default 60 days).

## Getting started

```bash
# Install dependencies and run
uv run python main.py
```

Or with a traditional virtualenv:

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
python main.py
```

Output plots are written to `./images/`. Downloaded price data is cached to `./cache/` so subsequent runs skip the network request.

## Usage

```
python main.py [--ticker TICKER [TICKER ...]] [--start-year YEAR] [--threshold FLOAT]
               [--interval {1d,1h}] [--stride INT] [--rolling-window INT]
               [--output-dir DIR] [--cache-dir DIR]
               [--no-cache] [--fixed-embedding] [--no-crash-annotations]
```

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `^GSPC` | Yahoo Finance ticker(s) — space-separated for multiple |
| `--start-year` | `1980` | Earliest year of data to analyse |
| `--threshold` | `0.3` | Crash probability cutoff in \[0, 1\] |
| `--interval` | `1d` | Bar interval: `1d` daily or `1h` hourly (last 730 days only) |
| `--stride` | `1` | Sliding window stride in bars. Use `4` for faster runs at lower resolution |
| `--rolling-window` | `10` | Smoothing window applied before normalisation |
| `--output-dir` | `./images` | Directory for output plots |
| `--cache-dir` | `./cache` | Directory for cached price CSV files |
| `--no-cache` | flag | Force a fresh download and overwrite the cache |
| `--fixed-embedding` | flag | Skip Takens parameter search; use `dim=3, delay=2` (much faster) |
| `--no-crash-annotations` | flag | Hide historical crash shading on plots |

Examples:

```bash
# Quick validation run (~5–10 min)
python main.py --fixed-embedding --stride 4 --start-year 2010

# Good quality run covering all major crashes (~20–40 min)
python main.py --fixed-embedding --stride 2 --start-year 2000

# Analyse Bitcoin with a tighter threshold
python main.py --ticker BTC-USD --threshold 0.2 --start-year 2015 --fixed-embedding

# Multiple tickers
python main.py --ticker QQQ SPY IWM --start-year 2000 --fixed-embedding
```

## Output files

Each run produces the following in `--output-dir`:

| File | Description |
|---|---|
| `{ticker}_{interval}_{metric}_detector.png` | Price chart with crash-signal dots for each of the 4 metrics |
| `{ticker}_metric_comparison.png` | 2×2 grid comparing raw distances across all 4 metrics |
| `{ticker}_{interval}_csd_indicators.png` | Rolling variance and AC1 alongside price |

Precision/recall/F1 against known crashes is printed to stdout at the end of each run.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_SIZE_DAILY` | `31` | Sliding window size in trading days |
| `WINDOW_SIZE_HOURLY` | `65` | Sliding window size in trading hours |
| `WINDOW_STRIDE` | `1` | Default stride (use `--stride 4` for speed) |
| `CAUSAL_NORM_WINDOW_DAILY` | `252` | Trailing bars for causal min/max normalisation (≈1 year) |
| `HOMOLOGY_DIMENSIONS` | `(0, 1)` | Homology groups tracked ($H_0$ and $H_1$) |

## Accuracy improvements vs original

- **Causal normalisation** — rolling 252-bar trailing window replaces global min/max, removing look-ahead bias
- **Stride = 1** — full temporal resolution (was stride = 4)
- **Wasserstein distance** — fourth TDA metric via optimal transport, complementing landscape/Betti/entropy
- **CSD indicators** — rolling variance and lag-1 AC validated as additive signals in the literature
- **Quantitative evaluation** — precision/recall/F1 printed against labelled crash windows

## Dependencies

- [giotto-tda](https://giotto-ai.github.io/gtda-docs/) — TDA pipeline (Takens embedding, Vietoris-Rips, persistence diagrams)
- [yfinance](https://github.com/ranaroussi/yfinance) — market data
- numpy, pandas, matplotlib, seaborn, scikit-learn
