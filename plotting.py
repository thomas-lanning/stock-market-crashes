"""Plot functions"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def _normalize_crash_probability(distances):
    """Normalize distances to [0, 1] crash probability via rolling statistics."""
    rolled_mean = pd.Series(distances).rolling(20, min_periods=1).mean()
    rolled_min = pd.Series(distances).rolling(len(distances), min_periods=1).min()
    rolled_max = pd.Series(distances).rolling(len(distances), min_periods=1).max()
    denom = rolled_max - rolled_min
    return (rolled_mean - rolled_min) / denom.where(denom != 0, other=1e-10)


def plot_crash_detections(
    start_date,
    end_date,
    threshold,
    distances,
    time_index_derivs,
    price_resampled_derivs,
    metric_name,
    output_dir="./images",
):
    probability_of_crash = _normalize_crash_probability(distances)

    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
        time_index_derivs < pd.Timestamp(end_date)
    )
    probability_region = probability_of_crash[is_date_in_interval]
    time_index_region = time_index_derivs[is_date_in_interval]
    price_region = price_resampled_derivs.loc[is_date_in_interval]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(time_index_region, probability_region, color="#1f77b4")
    plt.axhline(y=threshold, linewidth=2, color="#ff7f0e", linestyle="--", label="Threshold")
    plt.title(f"Crash Probability Based on {metric_name}")
    plt.legend(loc="best", prop={"size": 10})

    plt.subplot(1, 2, 2)
    plt.plot(
        price_region[probability_region.values > threshold],
        "#ff7f0e", marker=".", linestyle="None", markersize=4,
    )
    plt.plot(
        price_region[probability_region.values <= threshold],
        color="#1f77b4", marker=".", linestyle="None", markersize=4,
    )
    plt.title("Close Price")
    plt.legend(
        [
            f"Crash probability > {int(threshold * 100)}%",
            f"Crash probability \u2264 {int(threshold * 100)}%",
        ],
        loc="best",
        prop={"size": 10},
    )

    slug = metric_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(output_dir, f"crash_{slug}.png"), bbox_inches="tight")
    plt.close()


def plot_crash_comparisons(
    start_date,
    end_date,
    threshold,
    distances_1,
    distances_2,
    time_index_derivs,
    price_resampled_derivs,
    output_dir="./images",
):
    probability_1 = _normalize_crash_probability(distances_1)
    probability_2 = _normalize_crash_probability(distances_2)

    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
        time_index_derivs < pd.Timestamp(end_date)
    )
    prob_1_region = probability_1[is_date_in_interval]
    prob_2_region = probability_2[is_date_in_interval]
    price_region = price_resampled_derivs.loc[is_date_in_interval]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        price_region[prob_1_region.values > threshold],
        "#ff7f0e", marker=".", linestyle="None", markersize=4,
    )
    plt.plot(
        price_region[prob_1_region.values <= threshold],
        "#1f77b4", marker=".", linestyle="None", markersize=4,
    )
    plt.title("Baseline Detector")
    plt.ylabel("Close Price", fontsize=12)
    plt.legend(
        [
            f"Crash probability > {int(threshold * 100)}%",
            f"Crash probability \u2264 {int(threshold * 100)}%",
        ],
        loc="best",
        prop={"size": 10},
    )

    plt.subplot(1, 2, 2)
    plt.plot(
        price_region[prob_2_region.values > threshold],
        "#ff7f0e", marker=".", linestyle="None", markersize=4,
    )
    plt.plot(
        price_region[prob_2_region.values <= threshold],
        "#1f77b4", marker=".", linestyle="None", markersize=4,
    )
    plt.title("Topological Detector")
    plt.legend(
        [
            f"Crash probability > {int(threshold * 100)}%",
            f"Crash probability \u2264 {int(threshold * 100)}%",
        ],
        loc="best",
        prop={"size": 10},
    )

    plt.savefig(os.path.join(output_dir, "crash_comparison.png"), bbox_inches="tight")
    plt.close()
