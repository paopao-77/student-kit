import csv
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY = Path("results/summary")
FIGURES = Path("results/figures")
DRAFTS = Path("results/drafts")

PALETTE = {
    "blue": "#0F4D92",
    "teal": "#42949E",
    "green": "#6AAA64",
    "orange": "#D9853B",
    "red": "#B64342",
    "gray": "#8B8B8B",
    "light_gray": "#E5E5E5",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        number = float(value)
    except ValueError:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 9.5,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, basename: str) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = FIGURES / f"{basename}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        outputs.append(str(path))
    return outputs


def plot_holdout(ax: plt.Axes, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    modules = [row["module"] for row in rows]
    stratified = [f(row, "stratified_preferred") for row in rows]
    holdout = [f(row, "external_holdout") for row in rows]
    x = np.arange(len(modules))
    width = 0.36
    ax.bar(x - width / 2, stratified, width, label="Preferred", color=PALETTE["blue"])
    ax.bar(x + width / 2, holdout, width, label="External holdout", color=PALETTE["teal"])
    ax.set_xticks(x, modules)
    ax.set_ylabel("Metric value")
    ax.set_title("a  Preferred vs external holdout", loc="left", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)
    return [
        {
            "panel": "holdout",
            "item": row["module"],
            "metric": row["metric"],
            "preferred": row["stratified_preferred"],
            "external_holdout": row["external_holdout"],
            "delta": row["absolute_delta_holdout_minus_stratified"],
        }
        for row in rows
    ]


def plot_order_window(ax: plt.Axes, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    windows = [int(f(row, "order_window_size")) for row in rows]
    c2_auc = [f(row, "c2_auc_mean") for row in rows]
    c3_supp = [f(row, "c3_event_suppression_mean") for row in rows]
    random_supp = [f(row, "c3_random_suppression_mean") for row in rows]
    ax.plot(windows, c2_auc, marker="o", color=PALETTE["blue"], label="C2 AUC")
    ax.set_xlabel("Order-window size")
    ax.set_ylabel("C2 AUC")
    ax.set_xticks(windows)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
    twin = ax.twinx()
    twin.plot(windows, c3_supp, marker="s", color=PALETTE["green"], label="C3 suppression")
    twin.plot(windows, random_supp, marker="^", color=PALETTE["gray"], linestyle="--", label="Random same-budget")
    twin.set_ylabel("Suppression rate")
    ax.set_title("b  Order-window sensitivity", loc="left", fontweight="bold")
    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="center right", fontsize=8)
    return [
        {
            "panel": "order_window",
            "item": row["order_window_size"],
            "c2_auc": row["c2_auc_mean"],
            "c3_suppression": row["c3_event_suppression_mean"],
            "random_suppression": row["c3_random_suppression_mean"],
        }
        for row in rows
    ]


def plot_c3_strategies(ax: plt.Axes, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keep = {
        "heterorumor_c3_event_pulse": "HeteroRumor C3",
        "random_same_budget": "Random budget",
        "fixed_same_budget": "Fixed budget",
        "ed_id_adapted_same_budget": "ED-ID budget",
        "heterorumor_c3_no_game": "No game",
    }
    selected = [row for row in rows if row["strategy"] in keep]
    selected.sort(key=lambda row: f(row, "mean_suppression_rate_mean"))
    labels = [keep[row["strategy"]] for row in selected]
    values = [f(row, "mean_suppression_rate_mean") for row in selected]
    colors = [
        PALETTE["blue"] if row["strategy"] == "heterorumor_c3_event_pulse" else PALETTE["gray"]
        for row in selected
    ]
    y = np.arange(len(selected))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean suppression rate")
    ax.set_title("c  Preferred C3 strategy comparison", loc="left", fontweight="bold")
    ax.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.8)
    return [
        {
            "panel": "c3_strategy",
            "item": row["strategy"],
            "suppression": row["mean_suppression_rate_mean"],
            "cost": row["mean_cost_mean"],
            "benefit_cost_ratio": row["mean_benefit_cost_ratio_mean"],
        }
        for row in selected
    ]


def plot_efficiency(ax: plt.Axes, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = [row["module"].replace(" raw Weibo", "").replace(" preferred benchmark", "") for row in rows]
    values = [f(row, "seconds_median") for row in rows]
    colors = [PALETTE["orange"], PALETTE["orange"], PALETTE["green"], PALETTE["green"]]
    x = np.arange(len(rows))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Median seconds, log scale")
    ax.set_title("d  Runtime diagnostic", loc="left", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
    for idx, (row, value) in enumerate(zip(rows, values)):
        suffix = "*" if int(float(row.get("num_outliers", "0") or 0)) else ""
        ax.text(idx, value * 1.08, f"{value:.1f}s{suffix}", ha="center", va="bottom", fontsize=7.5)
    return [
        {
            "panel": "efficiency",
            "item": row["module"],
            "median_seconds": row["seconds_median"],
            "mean_seconds": row["seconds_mean"],
            "num_outliers": row["num_outliers"],
        }
        for row in rows
    ]


def write_note(outputs: list[str]) -> None:
    lines = [
        "# Raw Weibo Experiment Diagnostics Figure",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- {path}" for path in outputs)
    lines.extend(
        [
            "",
            "## Contents",
            "",
            "- Panel a checks whether the preferred raw-Weibo results survive the non-stratified fixed-seed holdout.",
            "- Panel b shows why `order_window_size=50` is kept as the preferred C2/C3 setting.",
            "- Panel c compares the preferred C3 controller with same-budget and ablated strategies.",
            "- Panel d summarizes runtime diagnostics; V1 uses the median because one recorded run is a long outlier.",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_e9_visual_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    publication_style()
    holdout_rows = read_csv(SUMMARY / "weibo_raw_external_holdout_comparison.csv")
    order_rows = read_csv(SUMMARY / "weibo_raw_c2_c3_order_window_sensitivity.csv")
    c3_rows = read_csv(SUMMARY / "c3_control_weibo_raw_preferred_summary.csv")
    efficiency_rows = read_csv(SUMMARY / "weibo_raw_efficiency_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    plot_rows: list[dict[str, Any]] = []
    plot_rows.extend(plot_holdout(axes[0, 0], holdout_rows))
    plot_rows.extend(plot_order_window(axes[0, 1], order_rows))
    plot_rows.extend(plot_c3_strategies(axes[1, 0], c3_rows))
    plot_rows.extend(plot_efficiency(axes[1, 1], efficiency_rows))
    fig.tight_layout(w_pad=2.5, h_pad=2.4)
    outputs = save_figure(fig, "fig_weibo_raw_e9_diagnostics")
    plt.close(fig)

    write_csv(
        FIGURES / "plot_data_weibo_raw_e9_diagnostics.csv",
        plot_rows,
        [
            "panel",
            "item",
            "metric",
            "preferred",
            "external_holdout",
            "delta",
            "c2_auc",
            "c3_suppression",
            "random_suppression",
            "suppression",
            "cost",
            "benefit_cost_ratio",
            "median_seconds",
            "mean_seconds",
            "num_outliers",
        ],
    )
    write_note(outputs)
    print({"outputs": outputs, "plot_data": str(FIGURES / "plot_data_weibo_raw_e9_diagnostics.csv")})


if __name__ == "__main__":
    main()
