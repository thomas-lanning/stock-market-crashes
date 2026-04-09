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

1. **Data** — Daily S&P 500 close prices from Yahoo Finance, resampled to fill weekends.
2. **Takens embedding** — The scalar price series is embedded into 3D point clouds using a time-delay embedding (`dimension=3`, `time_delay=2`).
3. **Sliding windows** — A sliding window (size 31, stride 4) turns the embedded series into a time series of point clouds.
4. **Persistence diagrams** — Vietoris-Rips persistent homology is applied to each point cloud, tracking $H_0$ and $H_1$ generators.
5. **Homological derivative** — Successive landscape or Betti-curve distances between diagrams measure topological change over time.
6. **Crash detection** — A rolling-mean normalisation produces a crash probability in $[0, 1]$; values above 0.3 flag potential crash periods.

The topological indicator is compared against a simple baseline: the absolute value of the first derivative averaged over the same sliding windows.

## Getting started

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
python main.py
```

Output plots are written to `./images/`. Downloaded price data is cached to `./cache/` so subsequent runs skip the network request.

## Usage

```
python main.py [--ticker TICKER] [--start-year YEAR] [--threshold FLOAT]
               [--output-dir DIR] [--cache-dir DIR] [--no-cache]
```

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `^GSPC` | Any Yahoo Finance ticker (e.g. `QQQ`, `BTC-USD`, `AAPL`) |
| `--start-year` | `1980` | Earliest year of data to analyse |
| `--threshold` | `0.3` | Normalised crash probability cutoff in \[0, 1\] |
| `--output-dir` | `./images` | Directory for output plots |
| `--cache-dir` | `./cache` | Directory for cached price CSV files |
| `--no-cache` | flag | Force a fresh download and overwrite the cache |

Examples:

```bash
# Analyse the Nasdaq-100 since 2000
python main.py --ticker QQQ --start-year 2000

# Analyse Bitcoin with a tighter crash threshold
python main.py --ticker BTC-USD --threshold 0.2 --output-dir ./btc-images

# Force a fresh S&P 500 download
python main.py --no-cache
```

## Key parameters

The TDA pipeline parameters are constants at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_DIMENSION` | `3` | Takens embedding dimension |
| `EMBEDDING_TIME_DELAY` | `2` | Takens time delay $\tau$ |
| `WINDOW_SIZE` | `31` | Sliding window size (trading days) |
| `WINDOW_STRIDE` | `4` | Stride between consecutive windows |
| `HOMOLOGY_DIMENSIONS` | `(0, 1)` | Homology groups to track |

## Dependencies

- [giotto-tda](https://giotto-ai.github.io/gtda-docs/) — TDA pipeline (Takens embedding, Vietoris-Rips, persistence diagrams)
- [yfinance](https://github.com/ranaroussi/yfinance) — S&P 500 data
- numpy, pandas, matplotlib, seaborn
