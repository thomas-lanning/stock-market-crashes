"""Detecting stock market crashes with Topological Data Analysis.

Applies Takens' embedding and persistent homology to a log-return time series to
compute a topological indicator of market stress.

Usage
-----
    python main.py                           # S&P 500 with defaults
    python main.py --ticker QQQ --start-year 2000
    python main.py --ticker BTC-USD --threshold 0.25 --output-dir ./out
    python main.py --no-cache               # force fresh download
    python main.py --fixed-embedding        # skip auto parameter search (faster)
    python main.py --no-crash-annotations  # hide historical crash shading
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
from gtda.diagrams import PersistenceEntropy

from homological_derivative import HomologicalDerivative
from plotting import (
    CAUSAL_NORM_WINDOW,
    KNOWN_CRASHES,
    _normalize_crash_probability,
    plot_csd_indicators,
    plot_topological_detector,
)

# ---------------------------------------------------------------------------
# Default configuration (used as argparse defaults)
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "^GSPC"
DEFAULT_START_YEAR = "1980"
DEFAULT_THRESHOLD = 0.3
DEFAULT_OUTPUT_DIR = "./images"
DEFAULT_CACHE_DIR = "./cache"
DEFAULT_INTERVAL = "1d"
DEFAULT_ROLLING_WINDOW = 10

# Takens embedding upper bounds (used when parameters_type="search")
# and fixed values (used when --fixed-embedding is passed)
EMBEDDING_DIMENSION_UPPER = 8
EMBEDDING_TIME_DELAY_UPPER = 20
EMBEDDING_DIMENSION_FIXED = 3
EMBEDDING_TIME_DELAY_FIXED = 2

WINDOW_SIZE_DAILY = 31    # ~6 weeks of trading days
WINDOW_SIZE_HOURLY = 65   # ~2 weeks of trading hours (6.5h/day × 10 days)
WINDOW_STRIDE = 1         # stride=1 gives full temporal resolution (use --stride 4 for speed)
HOMOLOGY_DIMENSIONS = (0, 1)

# Causal normalization: how many TDA windows look back for rolling min/max reference.
# At stride=1 this equals the number of bars, so 252 ≈ 1 year daily, 1764 ≈ 1 year hourly.
CAUSAL_NORM_WINDOW_DAILY = 252
CAUSAL_NORM_WINDOW_HOURLY = 252 * 7


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
def load_price_series(ticker: str, start_year: str, cache_dir: str, no_cache: bool,
                      interval: str = "1d") -> pd.Series:
    """Download (or load from cache) close prices for a Yahoo Finance ticker.

    interval: "1d" for daily (full history), "1h" for hourly (last 730 days max).
    Cache stored at ``{cache_dir}/{slug}_{interval}_from_{start_year}.csv``.
    """
    slug = ticker.replace("^", "").replace("-", "_")
    cache_path = os.path.join(cache_dir, f"{slug}_{interval}_from_{start_year}.csv")

    if not no_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        price = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze().dropna()
    else:
        print(f"Downloading {ticker} from Yahoo Finance (interval={interval})...")
        if interval == "1h":
            raw = yf.Ticker(ticker).history(period="730d", interval="1h")
        else:
            raw = yf.Ticker(ticker).history(period="max", interval="1d")
        price_resampled = raw["Close"][start_year:].dropna()
        os.makedirs(cache_dir, exist_ok=True)
        price_resampled.to_csv(cache_path)
        print(f"Cached to {cache_path}")
        price = price_resampled

    label = "hourly" if interval == "1h" else "daily"
    print(f"Loaded {len(price)} {label} observations from {start_year} ({ticker})")
    return price


# ---------------------------------------------------------------------------
# Log-returns
# ---------------------------------------------------------------------------
def _to_log_returns(price: pd.Series) -> pd.Series:
    """Convert a price series to log-returns.

    Length is N-1. Index is price.index[1:] so each return is stamped at the
    day it becomes observable. Using log-returns makes the Takens embedding
    stationary and scale-invariant across different price levels and decades.
    """
    return pd.Series(np.diff(np.log(price.values)), index=price.index[1:], name="log_return")


# ---------------------------------------------------------------------------
# Takens embedding
# ---------------------------------------------------------------------------
def embed_price_series(series: pd.Series, dimension: int, time_delay: int,
                       window_size: int, window_stride: int,
                       parameters_type: str = "search"):
    """Embed a series via Takens' time-delay embedding then apply a sliding window.

    Parameters
    ----------
    series          : pd.Series — log-returns (or any stationary series)
    dimension       : int — upper bound for search, or exact value for fixed mode
    time_delay      : int — upper bound for search, or exact value for fixed mode
    parameters_type : "search" (auto, default) or "fixed"

    Returns
    -------
    embedded_windows   : np.ndarray, shape (n_windows, window_size, dimension)
    embedder_time_delay : int — fitted value
    embedder_dimension  : int — fitted value
    """
    embedder = ts.SingleTakensEmbedding(
        parameters_type=parameters_type,
        dimension=dimension,
        time_delay=time_delay,
        n_jobs=-1,
    )
    series_embedded = embedder.fit_transform(series.values)

    sliding_window = ts.SlidingWindow(size=window_size, stride=window_stride)
    embedded_windows = sliding_window.fit_transform(series_embedded)

    print(
        f"Takens embedding: dimension={embedder.dimension_}, "
        f"time_delay={embedder.time_delay_}"
    )
    print(f"Sliding window: {embedded_windows.shape[0]} windows")
    return embedded_windows, embedder.time_delay_, embedder.dimension_


# ---------------------------------------------------------------------------
# Baseline: first derivative
# ---------------------------------------------------------------------------
def compute_baseline(price: pd.Series, log_returns: pd.Series,
                     embedder_dimension: int, embedder_time_delay: int,
                     window_size: int, window_stride: int):
    """Compute the absolute first-derivative baseline over sliding windows.

    Uses log_returns for windowing (stationary, scale-invariant).
    Maps window endpoints back into price.index via the +1 offset:
      log_returns[i] corresponds to price[i+1].

    Returns
    -------
    abs_derivative_of_means : np.ndarray
    time_index              : pd.DatetimeIndex
    price_at_index          : pd.Series
    """
    lr_values = log_returns.values
    window_size_price = window_size + (embedder_dimension - 1) * embedder_time_delay

    sliding_window_lr = ts.SlidingWindow(size=window_size_price, stride=window_stride)
    window_indices = sliding_window_lr.slice_windows(lr_values)
    price_windows = sliding_window_lr.fit_transform(lr_values)

    abs_derivative_of_means = np.abs(np.mean(np.diff(price_windows, axis=0), axis=1))

    # log_returns[j] -> price[j+1], so win[1]-1 (last LR index) + 1 = win[1]
    indices = [win[1] for win in window_indices[1:]]
    time_index = price.iloc[indices].index
    price_at_index = price.loc[time_index]

    return abs_derivative_of_means, time_index, price_at_index


# ---------------------------------------------------------------------------
# Persistence diagrams
# ---------------------------------------------------------------------------
def compute_persistence_diagrams(embedded_windows: np.ndarray, homology_dimensions):
    print("Computing Vietoris-Rips persistence diagrams...")
    vr = hl.VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=-1)
    diagrams = vr.fit_transform(embedded_windows)
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


def compute_entropy_distances(diagrams: np.ndarray) -> np.ndarray:
    """Compute successive L2 distances between persistent entropy vectors.

    PersistenceEntropy maps each diagram to a vector of length n_homology_dims.
    Taking the L2 norm of consecutive differences gives a scalar per window
    transition — the same output shape as landscape/betti distances.
    """
    print("Computing persistent entropy distances...")
    pe = PersistenceEntropy(normalize=True, n_jobs=-1)
    entropy_vals = pe.fit_transform(diagrams)       # (n_windows, n_dims)
    diffs = np.diff(entropy_vals, axis=0)            # (n_windows-1, n_dims)
    return np.linalg.norm(diffs, axis=1, ord=2)     # (n_windows-1,)


def compute_wasserstein_distances(diagrams: np.ndarray) -> np.ndarray:
    """Compute successive Wasserstein distances between consecutive persistence diagrams.

    Wasserstein distance captures diagram-to-diagram change directly (via optimal
    transport matching) without going through a vectorization step. It serves as a
    fourth complementary signal alongside landscape, Betti, and entropy.

    With multiple homology dimensions, fit_transform returns shape (n-1, n_dims);
    take the L2 norm across dimensions to get a scalar per window transition.
    """
    print("Computing Wasserstein distances...")
    hom_der = HomologicalDerivative(
        metric="wasserstein",
        metric_params={"p": 2, "delta": 0.3},
        order=None,
        n_jobs=-1,
    )
    result = hom_der.fit_transform(diagrams)
    if result.ndim == 2:
        return np.linalg.norm(result, axis=1)
    return result


# ---------------------------------------------------------------------------
# Critical Slowing Down indicators
# ---------------------------------------------------------------------------
def compute_csd_indicators(log_returns: pd.Series, window_size: int) -> pd.DataFrame:
    """Compute Critical Slowing Down indicators: rolling variance and lag-1 autocorrelation.

    These classical early-warning signals have been shown to add value when combined
    with TDA landscape norms (Frontiers 2022). Rolling window matches the TDA sliding
    window size for consistency.
    """
    print("Computing CSD indicators (rolling variance + AC1)...")
    min_periods = max(window_size // 2, 2)
    rolling_var = log_returns.rolling(window_size, min_periods=min_periods).var()
    rolling_ac1 = log_returns.rolling(window_size, min_periods=min_periods).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    return pd.DataFrame({"variance": rolling_var, "ac1": rolling_ac1})


# ---------------------------------------------------------------------------
# Precision / recall evaluation
# ---------------------------------------------------------------------------
def evaluate_precision_recall(
    probability: np.ndarray,
    time_index: pd.DatetimeIndex,
    threshold: float,
    lead_days: int = 60,
) -> dict:
    """Evaluate crash-detector precision/recall against KNOWN_CRASHES.

    True positive  : at least one signal fires within lead_days before a crash start.
    False positive : a signal fires with no known crash within lead_days.
    False negative : a known crash has no preceding signal within lead_days.

    Parameters
    ----------
    lead_days : how many calendar days before a crash onset count as a valid early warning.
    """
    t = pd.to_datetime(time_index, utc=True).tz_localize(None)
    prob = probability.values if hasattr(probability, "values") else np.asarray(probability)
    signals = t[prob > threshold]

    tp_crashes, fn_crashes = set(), set()
    for name, start_str, _ in KNOWN_CRASHES:
        crash_start = pd.Timestamp(start_str)
        lead_start = crash_start - pd.Timedelta(days=lead_days)
        if any(lead_start <= s <= crash_start for s in signals):
            tp_crashes.add(name)
        else:
            fn_crashes.add(name)

    fp_count = 0
    for sig in signals:
        near = False
        for _, start_str, end_str in KNOWN_CRASHES:
            crash_start = pd.Timestamp(start_str)
            crash_end = pd.Timestamp(end_str)
            lead_start = crash_start - pd.Timedelta(days=lead_days)
            if lead_start <= sig <= crash_end:
                near = True
                break
        if not near:
            fp_count += 1

    tp, fn, fp = len(tp_crashes), len(fn_crashes), fp_count
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "detected":  sorted(tp_crashes),
        "missed":    sorted(fn_crashes),
    }


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


def plot_metric_comparison(time_index, landscape_dists, betti_dists, entropy_dists,
                           wasserstein_dists, ticker: str, output_dir: str):
    metrics = [
        (landscape_dists,    "Landscape",          "#1f77b4"),
        (betti_dists,        "Betti Curve",         "#ff7f0e"),
        (entropy_dists,      "Persistent Entropy",  "#2ca02c"),
        (wasserstein_dists,  "Wasserstein",         "#d62728"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    for ax, (values, label, color) in zip(axes.flat, metrics):
        ax.plot(time_index, values, color, linewidth=0.8)
        ax.set_title(f"{label} Distances — {ticker}")

    slug = _ticker_slug(ticker)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_metric_comparison.png"), bbox_inches="tight", dpi=150)
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
    interval: str = DEFAULT_INTERVAL,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    fixed_embedding: bool = False,
    show_crashes: bool = True,
    stride: int = WINDOW_STRIDE,
):
    setup(output_dir)
    slug = _ticker_slug(ticker)
    window_size = WINDOW_SIZE_HOURLY if interval == "1h" else WINDOW_SIZE_DAILY
    causal_norm_window = CAUSAL_NORM_WINDOW_HOURLY if interval == "1h" else CAUSAL_NORM_WINDOW_DAILY

    parameters_type = "fixed" if fixed_embedding else "search"
    embedding_dimension = EMBEDDING_DIMENSION_FIXED if fixed_embedding else EMBEDDING_DIMENSION_UPPER
    embedding_time_delay = EMBEDDING_TIME_DELAY_FIXED if fixed_embedding else EMBEDDING_TIME_DELAY_UPPER

    # 1. Load data
    price = load_price_series(ticker, start_year, cache_dir, no_cache, interval)

    # 2. Convert to log-returns (stationary, scale-invariant)
    log_returns = _to_log_returns(price)

    # 3. Takens embedding + sliding windows on log-returns
    embedded_windows, embedder_time_delay, embedder_dimension = embed_price_series(
        log_returns,
        dimension=embedding_dimension,
        time_delay=embedding_time_delay,
        window_size=window_size,
        window_stride=stride,
        parameters_type=parameters_type,
    )

    # 4. Baseline: absolute first derivative of log-returns over sliding windows
    _, time_index, price_at_index = compute_baseline(
        price,
        log_returns,
        embedder_dimension=embedder_dimension,
        embedder_time_delay=embedder_time_delay,
        window_size=window_size,
        window_stride=stride,
    )

    # 5. Persistence diagrams via Vietoris-Rips
    diagrams = compute_persistence_diagrams(embedded_windows, HOMOLOGY_DIMENSIONS)

    # 6. Homological derivatives (4 metrics)
    landscape_dists    = compute_landscape_distances(diagrams)
    betti_dists        = compute_betti_distances(diagrams)
    entropy_dists      = compute_entropy_distances(diagrams)
    wasserstein_dists  = compute_wasserstein_distances(diagrams)

    # 7. CSD indicators (rolling variance + AC1) — classical early-warning signals
    csd_df = compute_csd_indicators(log_returns, window_size)

    # Sanity check: all TDA distance arrays must align with time_index
    assert len(time_index) == len(landscape_dists), (
        f"Index length mismatch: time_index={len(time_index)}, landscape_dists={len(landscape_dists)}"
    )
    for name, arr in [("betti", betti_dists), ("entropy", entropy_dists), ("wasserstein", wasserstein_dists)]:
        assert len(arr) == len(landscape_dists), (
            f"{name} length mismatch: {len(arr)} vs {len(landscape_dists)}"
        )

    # 8. Topological detector plots + precision/recall evaluation
    metrics = [
        ("landscape",   landscape_dists),
        ("betti",       betti_dists),
        ("entropy",     entropy_dists),
        ("wasserstein", wasserstein_dists),
    ]

    print("\n=== Detector Evaluation (against known crashes, 60-day lead window) ===")
    for metric_name, dists in metrics:
        plot_topological_detector(
            distances=dists,
            time_index_derivs=time_index,
            price=price,
            threshold=threshold,
            ticker=ticker,
            metric_name=metric_name,
            interval=interval,
            output_dir=output_dir,
            rolling_window=rolling_window,
            causal_window=causal_norm_window,
            show_crashes=show_crashes,
        )
        prob = _normalize_crash_probability(dists, rolling_window, causal_norm_window)
        results = evaluate_precision_recall(prob, time_index, threshold)
        print(
            f"  {metric_name:12s}  P={results['precision']:.3f}  R={results['recall']:.3f}  "
            f"F1={results['f1']:.3f}  TP={results['tp']}  FP={results['fp']}  FN={results['fn']}"
        )
        print(f"               detected={results['detected']}  missed={results['missed']}")

    # 9. CSD indicator plot
    plot_csd_indicators(csd_df, price, ticker, interval, output_dir, show_crashes)

    # 10. Side-by-side metric comparison (2×2 grid)
    plot_metric_comparison(
        time_index, landscape_dists, betti_dists, entropy_dists, wasserstein_dists,
        ticker, output_dir,
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
        "--ticker", default=[DEFAULT_TICKER], nargs="+",
        help=f"Yahoo Finance ticker symbol(s) (default: {DEFAULT_TICKER})",
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
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL, choices=["1d", "1h"],
        help="Price bar interval: '1d' daily (default) or '1h' hourly (last 730 days only)",
    )
    parser.add_argument(
        "--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, dest="rolling_window",
        help=f"Rolling window size for smoothing crash probability (default: {DEFAULT_ROLLING_WINDOW})",
    )
    parser.add_argument(
        "--fixed-embedding", action="store_true", dest="fixed_embedding",
        help=f"Use fixed Takens parameters (dim={EMBEDDING_DIMENSION_FIXED}, delay={EMBEDDING_TIME_DELAY_FIXED}) "
             "instead of auto-search (faster but less optimal)",
    )
    parser.add_argument(
        "--no-crash-annotations", action="store_false", dest="show_crashes",
        help="Hide historical crash shading on detector plots",
    )
    parser.add_argument(
        "--stride", type=int, default=WINDOW_STRIDE,
        help=f"Sliding window stride in bars (default: {WINDOW_STRIDE}). "
             "Use 4 for faster computation at reduced temporal resolution.",
    )

    args = parser.parse_args()
    for ticker in args.ticker:
        main(
            ticker=ticker,
            start_year=args.start_year,
            threshold=args.threshold,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
            interval=args.interval,
            rolling_window=args.rolling_window,
            fixed_embedding=args.fixed_embedding,
            show_crashes=args.show_crashes,
            stride=args.stride,
        )
