import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "neutral": "#CFCECE",
    "dark": "#272727",
}


def load_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def annotate_bar(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        color=PALETTE["dark"],
    )


def main() -> None:
    v1 = load_first_row(Path("results/summary/v1_plm_multiseed_summary.csv"))
    v2 = load_first_row(Path("results/summary/v2_c1_disentangled_multiseed_summary.csv"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Panel A: clean multi-seed performance.
    clean_labels = ["V1\nMiniLM", "V2/C1\nDisentangled"]
    clean_means = [
        as_float(v1, "mape_mean"),
        as_float(v2, "mape_mean"),
    ]
    clean_stds = [
        as_float(v1, "mape_std"),
        as_float(v2, "mape_std"),
    ]
    x = np.arange(len(clean_labels))
    axes[0].bar(
        x,
        clean_means,
        yerr=clean_stds,
        capsize=5,
        color=[PALETTE["neutral"], PALETTE["blue_main"]],
        edgecolor="black",
        linewidth=1.2,
    )
    axes[0].set_xticks(x, clean_labels)
    axes[0].set_ylabel("Test MAPE")
    axes[0].set_title("A  Multi-seed clean performance", loc="left", fontweight="bold")
    y_min = min(clean_means) - max(clean_stds) * 2.2
    y_max = max(clean_means) + max(clean_stds) * 2.5
    axes[0].set_ylim(y_min, y_max)
    for xi, mean in zip(x, clean_means):
        annotate_bar(axes[0], xi, mean + max(clean_stds) * 0.35, f"{mean:.4f}")

    # Panel B: V2 robustness under text perturbation and matched text swap.
    robust_labels = ["Clean", "Text noise\n0.3", "Matched\nswap"]
    robust_means = [
        as_float(v2, "mape_mean"),
        as_float(v2, "text_noise_0.3_mape_mean"),
        as_float(v2, "matched_text_swap_mape_mean"),
    ]
    robust_stds = [
        as_float(v2, "mape_std"),
        as_float(v2, "text_noise_0.3_mape_std"),
        as_float(v2, "matched_text_swap_mape_std"),
    ]
    xr = np.arange(len(robust_labels))
    axes[1].bar(
        xr,
        robust_means,
        yerr=robust_stds,
        capsize=5,
        color=[PALETTE["blue_main"], PALETTE["green"], PALETTE["red"]],
        edgecolor="black",
        linewidth=1.2,
    )
    axes[1].set_xticks(xr, robust_labels)
    axes[1].set_ylabel("Test MAPE")
    axes[1].set_title("B  V2/C1 robustness checks", loc="left", fontweight="bold")
    r_min = min(robust_means) - max(robust_stds) * 2.2
    r_max = max(robust_means) + max(robust_stds) * 2.8
    axes[1].set_ylim(r_min, r_max)
    for xi, mean in zip(xr, robust_means):
        annotate_bar(axes[1], xi, mean + max(robust_stds) * 0.35, f"{mean:.4f}")

    axes[1].text(
        2,
        robust_means[2] + robust_stds[2] + max(robust_stds) * 0.65,
        f"+{as_float(v2, 'matched_text_swap_delta_mape_mean'):.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color=PALETTE["red"],
    )

    for ax in axes:
        ax.tick_params(axis="both", width=1.4, length=4)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=1.2)
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig8_v2_disentangled_multiseed.png", dpi=300)
    fig.savefig(output_dir / "fig8_v2_disentangled_multiseed.svg")


if __name__ == "__main__":
    main()
