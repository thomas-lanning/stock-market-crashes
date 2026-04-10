"""Plot functions"""

import os
import pandas as pd
import matplotlib.pyplot as plt


KNOWN_CRASHES = [
    ("Black Monday",  "1987-10-14", "1987-10-19"),
    ("Dot-com",       "2000-03-01", "2002-10-09"),
    ("GFC",           "2007-10-09", "2009-03-09"),
    ("COVID",         "2020-02-19", "2020-03-23"),
    ("2022 bear",     "2022-01-03", "2022-10-12"),
]

# 1 year of trading days — causal normalization reference window for daily data.
# For hourly pass causal_window=252*7 (~1 year of trading hours).
CAUSAL_NORM_WINDOW = 252


def _normalize_crash_probability(distances, rolling_window: int = 10,
                                  causal_window: int = CAUSAL_NORM_WINDOW):
    """Normalize distances to [0, 1] crash probability using causal rolling statistics.

    Uses a trailing causal_window for min/max reference so each value only sees
    data available at that point in time — no look-ahead bias.
    """
    s = pd.Series(distances)
    smoothed = s.rolling(rolling_window, min_periods=1).mean()
    rolled_min = smoothed.rolling(causal_window, min_periods=rolling_window).min()
    rolled_max = smoothed.rolling(causal_window, min_periods=rolling_window).max()
    denom = rolled_max - rolled_min
    normalized = (smoothed - rolled_min) / denom.where(denom != 0, other=1e-10)
    return normalized.clip(0, 1).fillna(0)


def _strip_tz(index):
    return pd.to_datetime(index, utc=True).tz_localize(None)


def _add_crash_annotations(ax, price_index, show_crashes: bool = True):
    """Shade known historical crash windows on an existing axis."""
    if not show_crashes:
        return
    plot_start = price_index.min()
    plot_end = price_index.max()
    first = True
    for label, start, end in KNOWN_CRASHES:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if e < plot_start or s > plot_end:
            continue
        ax.axvspan(
            max(s, plot_start), min(e, plot_end),
            alpha=0.12, color="#d62728", zorder=0,
            label="Known crash" if first else "_nolegend_",
        )
        first = False


def plot_topological_detector(
    distances,
    time_index_derivs,
    price,
    threshold,
    ticker,
    metric_name="landscape",
    interval="1d",
    output_dir="./images",
    rolling_window: int = 10,
    causal_window: int = CAUSAL_NORM_WINDOW,
    show_crashes: bool = True,
):
    """Plot price with crash-signal dots overlaid on the same axis.

    The time_index_derivs already points to the last day of each sliding window,
    so each probability value is correctly stamped at the date when the signal
    was first available (no look-ahead).
    """
    probability = _normalize_crash_probability(distances, rolling_window, causal_window)
    t = _strip_tz(time_index_derivs)

    # Full price series on a clean index for the background line
    price_plot = price.copy()
    price_plot.index = _strip_tz(price_plot.index)

    # Prices at the window dates, used for dot placement
    price_at_t = price_plot.reindex(t, method="nearest")

    above = (probability > threshold).values
    crash_t = t[above]
    crash_price = price_at_t.values[above]

    fig, ax = plt.subplots(figsize=(15, 5))

    _add_crash_annotations(ax, price_plot.index, show_crashes)
    ax.plot(price_plot.index, price_plot.values, color="#333333", linewidth=1, label="Close Price")
    ax.scatter(crash_t, crash_price, color="#ff7f0e", s=12, zorder=3,
               label=f"Crash signal (prob > {int(threshold*100)}%)")

    ax.set_title(f"Topological Crash Detector ({metric_name}) — {ticker}  [threshold={threshold}]")
    ax.set_ylabel("Close Price")
    ax.set_xlabel("Date")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", prop={"size": 9})

    slug = ticker.replace("^", "").replace("-", "_")
    plt.savefig(os.path.join(output_dir, f"{slug}_{interval}_{metric_name}_detector.png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_csd_indicators(
    csd_df: "pd.DataFrame",
    price: "pd.Series",
    ticker: str,
    interval: str = "1d",
    output_dir: str = "./images",
    show_crashes: bool = True,
):
    """Plot Critical Slowing Down indicators (rolling variance + AC1) alongside price.

    These classical early-warning signals are plotted on shared x-axis for direct
    comparison with TDA crash signals.
    """
    price_plot = price.copy()
    price_plot.index = _strip_tz(price_plot.index)
    csd_plot = csd_df.copy()
    csd_plot.index = _strip_tz(csd_plot.index)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    _add_crash_annotations(axes[0], price_plot.index, show_crashes)
    axes[0].plot(price_plot.index, price_plot.values, color="#333333", linewidth=1)
    axes[0].set_ylabel("Close Price")
    axes[0].set_title(f"Critical Slowing Down Indicators — {ticker}")

    _add_crash_annotations(axes[1], price_plot.index, show_crashes)
    axes[1].plot(csd_plot.index, csd_plot["variance"], color="#1f77b4", linewidth=1)
    axes[1].set_ylabel("Rolling Variance")

    _add_crash_annotations(axes[2], price_plot.index, show_crashes)
    axes[2].plot(csd_plot.index, csd_plot["ac1"], color="#ff7f0e", linewidth=1)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.5)
    axes[2].set_ylabel("Rolling AC1 (lag-1)")
    axes[2].set_xlabel("Date")

    slug = ticker.replace("^", "").replace("-", "_")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_{interval}_csd_indicators.png"), bbox_inches="tight", dpi=150)
    plt.close()
