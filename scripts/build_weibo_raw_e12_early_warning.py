import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFERRED_C2_DIR = Path("results/c2_breakout_weibo_raw_ow50")
SUMMARY = Path("results/summary")
FIGURES = Path("results/figures")
DRAFTS = Path("results/drafts")
SEEDS = [7, 21, 42, 84, 2024]
LEAD_THRESHOLDS = [0, 50, 100, 200, 500, 1000]

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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def as_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_int(value: Any, default: int = -1) -> int:
    number = as_float(value)
    if not math.isfinite(number):
        return default
    return int(round(number))


def fmt(value: float, digits: int = 6) -> float | str:
    return round(value, digits) if math.isfinite(value) else ""


def prediction_path(seed: int) -> Path:
    return PREFERRED_C2_DIR / f"weibo_breakout_stratified_seed{seed}_predictions.csv"


def load_seed_rows(seed: int) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(prediction_path(seed)):
        if row.get("split") != "test" or row.get("model") != "heterorumor_c2":
            continue
        label = as_int(row.get("label_id"), 0)
        pred = as_int(row.get("pred_label_id"), 0)
        lead = max(0.0, as_float(row.get("lead_time_minutes"), 0.0))
        has_warning = pred == 1 and as_int(row.get("first_warning_window")) >= 0
        rows.append(
            {
                "seed": seed,
                "sample_id": row["sample_id"],
                "label_id": label,
                "pred_label_id": pred,
                "score_label_1": as_float(row.get("score_label_1")),
                "first_warning_window": as_int(row.get("first_warning_window")),
                "breakout_window": as_int(row.get("breakout_window")),
                "lead_time": lead if label == 1 and has_warning else 0.0,
                "has_warning": int(has_warning),
                "true_positive_warning": int(label == 1 and has_warning),
                "false_alarm": int(label == 0 and has_warning),
            }
        )
    return rows


def seed_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    positives = [row for row in rows if row["label_id"] == 1]
    negatives = [row for row in rows if row["label_id"] == 0]
    warned_positives = [row for row in positives if row["true_positive_warning"]]
    false_alarms = [row for row in negatives if row["false_alarm"]]
    lead_values = [float(row["lead_time"]) for row in warned_positives]
    result: dict[str, Any] = {
        "seed": seed,
        "num_test_samples": len(rows),
        "num_positive_samples": len(positives),
        "num_negative_samples": len(negatives),
        "warning_rate": fmt(sum(row["has_warning"] for row in rows) / max(len(rows), 1)),
        "positive_warning_recall": fmt(len(warned_positives) / max(len(positives), 1)),
        "false_alarm_rate": fmt(len(false_alarms) / max(len(negatives), 1)),
        "mean_lead_time": fmt(mean(lead_values) if lead_values else 0.0),
        "median_lead_time": fmt(median(lead_values) if lead_values else 0.0),
        "p25_lead_time": fmt(float(np.percentile(lead_values, 25)) if lead_values else 0.0),
        "p75_lead_time": fmt(float(np.percentile(lead_values, 75)) if lead_values else 0.0),
    }
    for threshold in LEAD_THRESHOLDS:
        count = sum(row["true_positive_warning"] and row["lead_time"] >= threshold for row in positives)
        result[f"recall_lead_ge_{threshold}"] = fmt(count / max(len(positives), 1))
    return result


def aggregate_seed_summaries(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        key
        for key in seed_rows[0]
        if key not in {"seed"} and not key.startswith("num_")
    ]
    output = []
    for metric in metrics:
        values = [as_float(row[metric], 0.0) for row in seed_rows]
        sd = stdev(values) if len(values) > 1 else 0.0
        output.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": fmt(mean(values)),
                "std": fmt(sd),
                "ci95": fmt(1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0),
                "min": fmt(min(values)),
                "max": fmt(max(values)),
            }
        )
    return output


def lead_curve(seed_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for threshold in LEAD_THRESHOLDS:
        key = f"recall_lead_ge_{threshold}"
        values = [as_float(row[key], 0.0) for row in seed_summaries]
        sd = stdev(values) if len(values) > 1 else 0.0
        rows.append(
            {
                "lead_threshold": threshold,
                "recall_mean": fmt(mean(values)),
                "recall_std": fmt(sd),
                "recall_ci95": fmt(1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0),
            }
        )
    return rows


def window_coverage(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in all_rows if row["label_id"] == 1]
    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        breakout_window = int(row["breakout_window"])
        if breakout_window >= 0:
            by_window[breakout_window].append(row)
    output = []
    for window, rows in sorted(by_window.items()):
        if len(rows) < 5:
            continue
        warned = [row for row in rows if row["true_positive_warning"]]
        output.append(
            {
                "breakout_window": window,
                "num_positive_seed_samples": len(rows),
                "warning_recall": fmt(len(warned) / len(rows)),
                "mean_lead_time": fmt(mean([row["lead_time"] for row in warned]) if warned else 0.0),
            }
        )
    return output


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


def plot_outputs(
    seed_summaries: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> list[str]:
    publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    ax = axes[0, 0]
    thresholds = [row["lead_threshold"] for row in curve_rows]
    recalls = [as_float(row["recall_mean"]) for row in curve_rows]
    ci95 = [as_float(row["recall_ci95"]) for row in curve_rows]
    ax.errorbar(thresholds, recalls, yerr=ci95, marker="o", color=PALETTE["blue"], linewidth=1.8, capsize=3)
    ax.set_xlabel("Required lead time (event-order units)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("a  Early-warning recall by lead threshold", loc="left", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)

    ax = axes[0, 1]
    warned_leads = [row["lead_time"] for row in all_rows if row["true_positive_warning"]]
    bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, max(max(warned_leads or [5000]), 5001)]
    ax.hist(warned_leads, bins=sorted(set(bins)), color=PALETTE["teal"], edgecolor="black", linewidth=0.6)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Lead time (event-order units)")
    ax.set_ylabel("Count across seeds")
    ax.set_title("b  Lead-time distribution", loc="left", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)

    ax = axes[1, 0]
    labels = ["warning_rate", "positive_warning_recall", "false_alarm_rate"]
    means = [mean([as_float(row[label], 0.0) for row in seed_summaries]) for label in labels]
    stds = [stdev([as_float(row[label], 0.0) for row in seed_summaries]) for label in labels]
    ax.bar(np.arange(len(labels)), means, yerr=stds, color=[PALETTE["gray"], PALETTE["green"], PALETTE["red"]], capsize=3)
    ax.set_xticks(np.arange(len(labels)), ["Warning rate", "Positive recall", "False alarm"], rotation=12, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("c  Seed-level warning rates", loc="left", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)

    ax = axes[1, 1]
    if coverage_rows:
        windows = [row["breakout_window"] for row in coverage_rows]
        recall = [as_float(row["warning_recall"], 0.0) for row in coverage_rows]
        sizes = [max(20, as_float(row["num_positive_seed_samples"], 1.0) * 2.5) for row in coverage_rows]
        ax.scatter(windows, recall, s=sizes, color=PALETTE["orange"], edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xscale("symlog", linthresh=20)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Breakout window")
    ax.set_ylabel("Warning recall")
    ax.set_title("d  Recall by breakout timing", loc="left", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
    fig.tight_layout(w_pad=2.2, h_pad=2.4)
    outputs = save_figure(fig, "fig_weibo_raw_e12_early_warning")
    plt.close(fig)
    return outputs


def write_note(
    aggregate_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    outputs: list[str],
) -> None:
    by_metric = {row["metric"]: row for row in aggregate_rows}
    lines = [
        "# Raw Weibo E12 Early-Warning Validation",
        "",
        "## Design",
        "",
        "- Source: preferred raw-Weibo C2 outputs under `order_window_size=50`.",
        "- Model: `heterorumor_c2`; split: stratified test split; seeds: `7, 21, 42, 84, 2024`.",
        "- Lead time is measured in event-order units because raw Weibo uses event-order windows rather than reliable wall-clock timestamps.",
        "",
        "## Headline Metrics",
        "",
        f"- Positive warning recall: {by_metric['positive_warning_recall']['mean']} +/- {by_metric['positive_warning_recall']['std']}.",
        f"- Mean lead time among warned positives: {by_metric['mean_lead_time']['mean']} +/- {by_metric['mean_lead_time']['std']} event-order units.",
        f"- Median lead time among warned positives: {by_metric['median_lead_time']['mean']} event-order units.",
        f"- False-alarm rate: {by_metric['false_alarm_rate']['mean']} +/- {by_metric['false_alarm_rate']['std']}.",
        "",
        "## Recall by Required Lead Time",
        "",
        "| required lead | recall mean | recall std |",
        "|---:|---:|---:|",
    ]
    for row in curve_rows:
        lines.append(f"| {row['lead_threshold']} | {row['recall_mean']} | {row['recall_std']} |")
    lines.extend(["", "## Outputs", ""])
    lines.extend(f"- {path}" for path in outputs)
    lines.extend(
        [
            "- results/summary/weibo_raw_e12_early_warning_seed_summary.csv",
            "- results/summary/weibo_raw_e12_early_warning_summary.csv",
            "- results/summary/weibo_raw_e12_early_warning_recall_curve.csv",
            "- results/summary/weibo_raw_e12_early_warning_window_coverage.csv",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_e12_early_warning.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    seed_summaries = []
    all_rows = []
    for seed in SEEDS:
        rows = load_seed_rows(seed)
        all_rows.extend(rows)
        seed_summaries.append(seed_summary(rows, seed))
    aggregate_rows = aggregate_seed_summaries(seed_summaries)
    curve_rows = lead_curve(seed_summaries)
    coverage_rows = window_coverage(all_rows)

    write_csv(SUMMARY / "weibo_raw_e12_early_warning_seed_summary.csv", seed_summaries)
    write_csv(SUMMARY / "weibo_raw_e12_early_warning_summary.csv", aggregate_rows)
    write_csv(SUMMARY / "weibo_raw_e12_early_warning_recall_curve.csv", curve_rows)
    if coverage_rows:
        write_csv(SUMMARY / "weibo_raw_e12_early_warning_window_coverage.csv", coverage_rows)
    outputs = plot_outputs(seed_summaries, curve_rows, all_rows, coverage_rows)
    write_note(aggregate_rows, curve_rows, outputs)
    print(
        {
            "seed_summary": str(SUMMARY / "weibo_raw_e12_early_warning_seed_summary.csv"),
            "summary": str(SUMMARY / "weibo_raw_e12_early_warning_summary.csv"),
            "recall_curve": str(SUMMARY / "weibo_raw_e12_early_warning_recall_curve.csv"),
            "window_coverage": str(SUMMARY / "weibo_raw_e12_early_warning_window_coverage.csv"),
            "figures": outputs,
            "note": str(DRAFTS / "weibo_raw_e12_early_warning.md"),
        }
    )


if __name__ == "__main__":
    main()
