import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_DIR = Path("results/summary")
OUTPUT_DIR = Path("results/figures")
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#6AAA64",
    "red": "#B64342",
    "neutral": "#8B8B8B",
    "teal": "#42949E",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def dynamics_series(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    model_prefixes = {
        "Early-count": "early_count_w",
        "SIR": "sir_w",
        "SEIR": "seir_w",
    }
    windows = [60, 180, 360]
    result = {}
    for label, prefix in model_prefixes.items():
        values = []
        for window in windows:
            model = f"{prefix}{window}m"
            match = next(
                row
                for row in rows
                if row["dataset"] == "pheme"
                and row["split_strategy"] == "stratified"
                and row["model"] == model
            )
            values.append(float(match["mape"]) * 100.0)
        result[label] = values
    return result


def v1_series(rows: list[dict[str, str]]) -> list[float]:
    ordered = sorted(rows, key=lambda row: float(row["observation_window_minutes"]))
    return [float(row["mape"]) * 100.0 for row in ordered]


def ablation_values(rows: list[dict[str, str]]) -> tuple[list[str], list[float]]:
    labels = {
        "heterorumor_v1_hurdle_wo_text": "No text",
        "heterorumor_v1_hurdle_wo_topology": "No topology",
        "heterorumor_v1_hurdle_wo_temporal": "No temporal",
        "heterorumor_v1_hurdle_wo_user": "No user",
    }
    ordered_names = [
        "heterorumor_v1_hurdle_wo_text",
        "heterorumor_v1_hurdle_wo_topology",
        "heterorumor_v1_hurdle_wo_temporal",
        "heterorumor_v1_hurdle_wo_user",
    ]
    by_model = {row["model"]: row for row in rows}
    return (
        [labels[name] for name in ordered_names],
        [float(by_model[name]["mape_delta_vs_full"]) * 100.0 for name in ordered_names],
    )


def save_figure(fig: plt.Figure, basename: str) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = OUTPUT_DIR / f"{basename}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        outputs.append(str(path))
    return outputs


def main() -> None:
    publication_style()
    paper_rows = read_csv(SUMMARY_DIR / "paper_test_metrics.csv")
    v1_window_rows = read_csv(SUMMARY_DIR / "v1_pheme_window_comparison.csv")
    ablation_rows = read_csv(SUMMARY_DIR / "v1_pheme_ablation.csv")

    windows = np.asarray([60, 180, 360])
    dynamics = dynamics_series(paper_rows)
    v1_values = v1_series(v1_window_rows)
    ablation_labels, ablation_delta = ablation_values(ablation_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    styles = {
        "Early-count": (PALETTE["neutral"], "o", "--"),
        "SIR": (PALETTE["red"], "s", "-."),
        "SEIR": (PALETTE["teal"], "D", ":"),
        "HeteroRumorDyn V1": (PALETTE["blue_main"], "o", "-"),
    }
    series = {**dynamics, "HeteroRumorDyn V1": v1_values}
    for label, values in series.items():
        color, marker, linestyle = styles[label]
        ax.plot(
            windows,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0 if label == "HeteroRumorDyn V1" else 1.5,
            markersize=5.5,
        )
    ax.set_xticks(windows)
    ax.set_xlabel("Observation window (min)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("a  Early cascade-size prediction", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[1]
    y = np.arange(len(ablation_labels))
    colors = [PALETTE["neutral"], PALETTE["neutral"], PALETTE["red"], PALETTE["green"]]
    bars = ax.barh(y, ablation_delta, color=colors, edgecolor="black", linewidth=0.7, height=0.62)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y, ablation_labels)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE change vs full model (pp)")
    ax.set_title("b  Modality ablation at 180 min", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    for bar, value in zip(bars, ablation_delta):
        offset = 0.025 if value >= 0 else -0.025
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
        )
    extent = max(abs(value) for value in ablation_delta)
    ax.set_xlim(-max(0.18, extent * 0.35), extent * 1.25)

    fig.subplots_adjust(wspace=0.38)
    outputs = save_figure(fig, "fig5_v1_window_and_ablation")
    plt.close(fig)
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
