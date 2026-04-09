"""Detecting stock market crashes with Topological Data Analysis.

Applies Takens' embedding and persistent homology to a price time series to
compute a topological indicator of market stress, then compares it against a
simple first-derivative baseline.

Usage
-----
    python main.py                           # S&P 500 with defaults
    python main.py --ticker QQQ --start-year 2000
    python main.py --ticker BTC-USD --threshold 0.25 --output-dir ./out
    python main.py --no-cache               # force fresh download
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

import gtda.time_series as ts
import gtda.homology as hl

from homological_derivative import HomologicalDerivative
from plotting import plot_crash_detections, plot_crash_comparisons

# ---------------------------------------------------------------------------
# Default configuration (used as argparse defaults)
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "^GSPC"
DEFAULT_START_YEAR = "1980"
DEFAULT_THRESHOLD = 0.3
DEFAULT_OUTPUT_DIR = "./images"
DEFAULT_CACHE_DIR = "./cache"

EMBEDDING_DIMENSION = 3
EMBEDDING_TIME_DELAY = 2
WINDOW_SIZE = 31
WINDOW_STRIDE = 4
HOMOLOGY_DIMENSIONS = (0, 1)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set(color_codes=True, rc={"figure.figsize": (12, 4)})
    sns.set_palette(sns.color_palette("muted"))


# ---------------------------------------------------------------------------
# Data loading (with cache)
# ---------------------------------------------------------------------------
def load_price_series(ticker: str, start_year: str, cache_dir: str, no_cache: bool) -> pd.Series:
    """Download (or load from cache) daily close prices for a Yahoo Finance ticker.

    The cache is stored as a CSV at ``{cache_dir}/{slug}_from_{start_year}.csv``.
    Pass ``no_cache=True`` to force a fresh download and overwrite the cache.
    """
    slug = ticker.replace("^", "").replace("-", "_")
    cache_path = os.path.join(cache_dir, f"{slug}_from_{start_year}.csv")

    if not no_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        price = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
    else:
        print(f"Downloading {ticker} from Yahoo Finance...")
        raw = yf.Ticker(ticker).history(period="max")
        price_full = raw["Close"]
        price_resampled = price_full.resample("24H").pad()[start_year:]
        os.makedirs(cache_dir, exist_ok=True)
        price_resampled.to_csv(cache_path)
        print(f"Cached to {cache_path}")
        price = price_resampled

    print(f"Loaded {len(price)} daily observations from {start_year} ({ticker})")
    return price


# ---------------------------------------------------------------------------
# Takens embedding
# ---------------------------------------------------------------------------
def embed_price_series(price: pd.Series, dimension: int, time_delay: int,
                       window_size: int, window_stride: int):
    """Embed a price series via Takens' time-delay embedding then apply a sliding window.

    Returns
    -------
    price_embedded_windows : np.ndarray, shape (n_windows, window_size, dimension)
    embedder_time_delay    : int
    embedder_dimension     : int
    """
    price_values = price.values

    embedder = ts.SingleTakensEmbedding(
        parameters_type="fixed",
        dimension=dimension,
        time_delay=time_delay,
        n_jobs=-1,
    )
    price_embedded = embedder.fit_transform(price_values)

    sliding_window = ts.SlidingWindow(size=window_size, stride=window_stride)
    price_embedded_windows = sliding_window.fit_transform(price_embedded)

    print(
        f"Takens embedding: dimension={embedder.dimension_}, "
        f"time_delay={embedder.time_delay_}"
    )
    print(f"Sliding window: {price_embedded_windows.shape[0]} windows")
    return price_embedded_windows, embedder.time_delay_, embedder.dimension_


# ---------------------------------------------------------------------------
# Baseline: first derivative
# ---------------------------------------------------------------------------
def compute_baseline(price: pd.Series, embedder_dimension: int, embedder_time_delay: int,
                     window_size: int, window_stride: int):
    """Compute the absolute first-derivative baseline over sliding windows.

    Returns
    -------
    abs_derivative_of_means : np.ndarray
    time_index              : pd.DatetimeIndex
    price_at_index          : pd.Series
    """
    price_values = price.values
    window_size_price = window_size + embedder_dimension * embedder_time_delay - 2

    sliding_window_price = ts.SlidingWindow(size=window_size_price, stride=window_stride)
    window_indices = sliding_window_price.slice_windows(price_values)
    price_windows = sliding_window_price.fit_transform(price_values)

    abs_derivative_of_means = np.abs(np.mean(np.diff(price_windows, axis=0), axis=1))

    indices = [win[1] - 1 for win in window_indices[1:]]
    time_index = price.iloc[indices].index
    price_at_index = price.loc[time_index]

    return abs_derivative_of_means, time_index, price_at_index


# ---------------------------------------------------------------------------
# Persistence diagrams
# ---------------------------------------------------------------------------
def compute_persistence_diagrams(price_embedded_windows: np.ndarray, homology_dimensions):
    print("Computing Vietoris-Rips persistence diagrams...")
    vr = hl.VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=-1)
    diagrams = vr.fit_transform(price_embedded_windows)
    print(f"Diagrams shape: {diagrams.shape}")
    return diagrams


# ---------------------------------------------------------------------------
# Homological derivatives
# ---------------------------------------------------------------------------
def compute_landscape_distances(diagrams: np.ndarray) -> np.ndarray:
    print("Computing landscape distances...")
    hom_der = HomologicalDerivative(
        metric="landscape",
        metric_params={"p": 2, "n_layers": 10, "n_bins": 1000},
        order=2,
        n_jobs=-1,
    )
    return hom_der.fit_transform(diagrams)


def compute_betti_distances(diagrams: np.ndarray) -> np.ndarray:
    print("Computing Betti curve distances...")
    hom_der = HomologicalDerivative(
        metric="betti",
        metric_params={"p": 2, "n_bins": 1000},
        order=2,
        n_jobs=-1,
    )
    return hom_der.fit_transform(diagrams)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _ticker_slug(ticker: str) -> str:
    return ticker.replace("^", "").replace("-", "_")


def plot_price_series(price: pd.Series, ticker: str, output_dir: str):
    plt.figure()
    plt.plot(price)
    plt.title(f"{ticker} Close Price")
    plt.savefig(os.path.join(output_dir, f"{_ticker_slug(ticker)}_close_price.png"), bbox_inches="tight")
    plt.close()


def plot_metric(time_index, values, title: str, output_dir: str, filename: str):
    plt.figure(figsize=(15, 5))
    plt.plot(time_index, values, color="#1f77b4")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def plot_landscape_vs_betti(time_index, landscape_dists, betti_dists, ticker: str, output_dir: str):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_index, landscape_dists, "#1f77b4")
    plt.title(f"Landscape Distances — {ticker}")

    plt.subplot(1, 2, 2)
    plt.plot(time_index, betti_dists, "#1f77b4")
    plt.title(f"Betti Curve Distances — {ticker}")

    slug = _ticker_slug(ticker)
    plt.savefig(os.path.join(output_dir, f"{slug}_metric_landscape_betti.png"), bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(
    ticker: str = DEFAULT_TICKER,
    start_year: str = DEFAULT_START_YEAR,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    no_cache: bool = False,
):
    setup(output_dir)
    slug = _ticker_slug(ticker)

    # 1. Load data
    price = load_price_series(ticker, start_year, cache_dir, no_cache)
    plot_price_series(price, ticker, output_dir)

    # 2. Takens embedding + sliding windows (point clouds)
    price_embedded_windows, embedder_time_delay, embedder_dimension = embed_price_series(
        price,
        dimension=EMBEDDING_DIMENSION,
        time_delay=EMBEDDING_TIME_DELAY,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
    )

    # 3. Baseline: absolute first derivative
    abs_derivative_of_means, time_index, price_at_index = compute_baseline(
        price,
        embedder_dimension=embedder_dimension,
        embedder_time_delay=embedder_time_delay,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
    )
    plot_metric(
        time_index,
        abs_derivative_of_means,
        f"First Derivative — {ticker}",
        output_dir,
        f"{slug}_metric_first_derivative.png",
    )

    # 4. Persistence diagrams via Vietoris-Rips
    diagrams = compute_persistence_diagrams(price_embedded_windows, HOMOLOGY_DIMENSIONS)

    # 5. Homological derivatives
    landscape_dists = compute_landscape_distances(diagrams)
    betti_dists = compute_betti_distances(diagrams)
    plot_landscape_vs_betti(time_index, landscape_dists, betti_dists, ticker, output_dir)

    # 6. Crash detection — baseline
    plot_crash_detections(
        start_date=f"{start_year}-01-01",
        end_date="2005-01-01",
        threshold=threshold,
        distances=abs_derivative_of_means,
        time_index_derivs=time_index,
        price_resampled_derivs=price_at_index,
        metric_name=f"First_Derivative_{slug}",
        output_dir=output_dir,
    )

    # 7. Crash detection — landscape distance (dot-com and 2008 crisis)
    plot_crash_detections(
        start_date="1990-01-01",
        end_date="2005-01-01",
        threshold=threshold,
        distances=landscape_dists,
        time_index_derivs=time_index,
        price_resampled_derivs=price_at_index,
        metric_name=f"Landscape_dot-com_{slug}",
        output_dir=output_dir,
    )
    plot_crash_detections(
        start_date="2005-01-01",
        end_date="2012-01-01",
        threshold=threshold,
        distances=landscape_dists,
        time_index_derivs=time_index,
        price_resampled_derivs=price_at_index,
        metric_name=f"Landscape_2008_{slug}",
        output_dir=output_dir,
    )

    # 8. Head-to-head comparison: baseline vs topological
    plot_crash_comparisons(
        start_date=f"{start_year}-01-01",
        end_date="2012-01-01",
        threshold=threshold,
        distances_1=abs_derivative_of_means,
        distances_2=landscape_dists,
        time_index_derivs=time_index,
        price_resampled_derivs=price_at_index,
        output_dir=output_dir,
    )

    print(f"\nAll plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect market crashes using Topological Data Analysis."
    )
    parser.add_argument(
        "--ticker", default=DEFAULT_TICKER,
        help=f"Yahoo Finance ticker symbol (default: {DEFAULT_TICKER})",
    )
    parser.add_argument(
        "--start-year", default=DEFAULT_START_YEAR, dest="start_year",
        help=f"Earliest year of data to analyse (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Normalised crash probability cutoff in [0, 1] (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir",
        help=f"Directory for output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR, dest="cache_dir",
        help=f"Directory for cached price data (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-cache", action="store_true", dest="no_cache",
        help="Force a fresh download even if cached data exists",
    )

    args = parser.parse_args()
    main(
        ticker=args.ticker,
        start_year=args.start_year,
        threshold=args.threshold,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )
