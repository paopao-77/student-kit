import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_DIR = Path("results/summary")
OUTPUT_DIR = Path("results/figures")
COLORS = ["#0F4D92", "#B64342", "#42949E", "#6AAA64"]


def read_csv(name: str) -> list[dict[str, str]]:
    with (SUMMARY_DIR / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    k_rows = read_csv("v2_c1_k_sensitivity.csv")
    kl_rows = read_csv("v2_c1_kl_sensitivity.csv")
    cf_rows = [
        row for row in read_csv("v2_c1_counterfactual_summary.csv") if row["split"] == "test"
    ]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    k = [int(row["latent_dim"]) for row in k_rows]
    axes[0].plot(k, [100 * float(row["best_val_mape"]) for row in k_rows], "o-", label="Validation", color=COLORS[0])
    axes[0].plot(k, [100 * float(row["test_mape"]) for row in k_rows], "s--", label="Test", color=COLORS[1])
    axes[0].set_xticks(k)
    axes[0].set_xlabel("Latent dimension K")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("a  Latent dimension", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#E5E5E5")

    kl = [float(row["kl_weight"]) for row in kl_rows]
    axes[1].plot(kl, [100 * float(row["best_val_mape"]) for row in kl_rows], "o-", label="Validation", color=COLORS[0])
    axes[1].plot(kl, [100 * float(row["test_mape"]) for row in kl_rows], "s--", label="Test", color=COLORS[1])
    axes[1].set_xscale("log")
    axes[1].set_xticks(kl, [f"{value:g}" for value in kl])
    axes[1].set_xlabel("KL weight")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("b  KL regularization", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#E5E5E5")

    noise = [0.0, 0.1, 0.2, 0.3]
    for index, row in enumerate(sorted(cf_rows, key=lambda value: float(value["counterfactual_weight"]))):
        values = [
            float(row["mape_clean"]),
            float(row["mape_noise_0.1"]),
            float(row["mape_noise_0.2"]),
            float(row["mape_noise_0.3"]),
        ]
        weight = float(row["counterfactual_weight"])
        axes[2].plot(
            [100 * value for value in noise],
            [100 * value for value in values],
            marker="o",
            linewidth=1.6,
            color=COLORS[index],
            label=f"lambda_CF={weight:g}",
        )
    axes[2].set_xlabel("Text feature masking (%)")
    axes[2].set_ylabel("Test MAPE (%)")
    axes[2].set_title("c  Counterfactual robustness", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", color="#E5E5E5")

    fig.subplots_adjust(wspace=0.35)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = OUTPUT_DIR / "fig7_v2_c1_sensitivity_and_counterfactual"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(basename.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(str(basename.with_suffix(".png")))


if __name__ == "__main__":
    main()
