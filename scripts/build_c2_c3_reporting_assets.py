import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


RESULTS_ROOT = Path("results")
SUMMARY_DIR = RESULTS_ROOT / "summary"
FIGURE_DIR = RESULTS_ROOT / "figures"
DRAFT_DIR = RESULTS_ROOT / "drafts"

DATASETS = ["pheme", "twitter15", "twitter16", "weibo"]
SEEDS = [7, 21, 42, 84, 2024]

C2_MODELS = [
    ("static_random_forest", "Static graph"),
    ("dynamic_random_forest", "Dynamic trend"),
    ("heterorumor_c2_dynamic_only", "Dynamic-only"),
    ("heterorumor_c2_community_only", "Community-only"),
    ("heterorumor_c2_no_temporal_trend", "w/o temporal trend"),
    ("heterorumor_c2_no_lowfreq", "w/o low-freq"),
    ("heterorumor_c2_no_cross", "w/o cross-comm"),
    ("heterorumor_c2", "HeteroRumorDyn-C2"),
]
C2_FIGURE_MODELS = [
    ("static_random_forest", "Static graph"),
    ("dynamic_random_forest", "Dynamic trend"),
    ("heterorumor_c2_dynamic_only", "Dynamic-only"),
    ("heterorumor_c2_community_only", "Community-only"),
    ("heterorumor_c2", "HeteroRumorDyn-C2"),
]
C3_STRATEGIES = [
    ("fixed_intervention", "Fixed"),
    ("influence_blocking", "Influence blocking"),
    ("random_same_budget", "Random same-budget"),
    ("fixed_same_budget", "Fixed same-budget"),
    ("ed_id_adapted", "ED-ID-adapted"),
    ("ed_id_adapted_same_budget", "ED-ID same-budget"),
    ("heterorumor_c3_no_game", "w/o game"),
    ("heterorumor_c3_no_event_trigger", "w/o trigger"),
    ("heterorumor_c3_event_pulse", "HeteroRumorDyn-C3"),
]
C3_FIGURE_STRATEGIES = [
    ("fixed_intervention", "Fixed"),
    ("influence_blocking", "Influence blocking"),
    ("random_same_budget", "Random same-budget"),
    ("fixed_same_budget", "Fixed same-budget"),
    ("ed_id_adapted_same_budget", "ED-ID same-budget"),
    ("heterorumor_c3_event_pulse", "HeteroRumorDyn-C3"),
]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "orange": "#D9853B",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{" + "l" * len(fieldnames) + "}",
        "\\toprule",
        " & ".join(fieldnames) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("_", "\\_") for field in fieldnames]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pm(mean_value: float, std_value: float, digits: int = 4) -> str:
    if not math.isfinite(mean_value):
        return ""
    if std_value and math.isfinite(std_value):
        return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"
    return f"{mean_value:.{digits}f}"


def load_c2_long() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULTS_ROOT / "c2_breakout").glob("*_metrics.json")):
        payload = read_json(path)
        dataset = payload.get("dataset", "")
        split_strategy = payload.get("split_strategy", "")
        seed = int(payload.get("seed", 0))
        for model, split_payload in payload.get("models", {}).items():
            test = split_payload.get("test", {})
            rows.append(
                {
                    "dataset": dataset,
                    "split_strategy": split_strategy,
                    "seed": seed,
                    "model": model,
                    "auc": test.get("auc"),
                    "f1": test.get("f1"),
                    "precision": test.get("precision"),
                    "recall": test.get("recall"),
                    "macro_f1": test.get("macro_f1"),
                    "precision_at_10pct": test.get("precision_at_10pct"),
                    "recall_at_10pct": test.get("recall_at_10pct"),
                    "mean_lead_time_minutes": test.get("mean_lead_time_minutes"),
                    "median_lead_time_minutes": test.get("median_lead_time_minutes"),
                    "warning_rate": test.get("warning_rate"),
                    "threshold": test.get("threshold"),
                    "num_samples": test.get("num_samples"),
                    "source_file": str(path),
                }
            )
    return rows


def load_c3_long() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULTS_ROOT / "c3_control").glob("*_metrics.json")):
        payload = read_json(path)
        dataset = payload.get("dataset", "")
        split_strategy = payload.get("split_strategy", "")
        seed = int(payload.get("seed", 0))
        for strategy, metrics in payload.get("strategies", {}).items():
            rows.append(
                {
                    "dataset": dataset,
                    "split_strategy": split_strategy,
                    "seed": seed,
                    "strategy": strategy,
                    "trigger_rate": metrics.get("trigger_rate"),
                    "mean_baseline_size": metrics.get("mean_baseline_size"),
                    "mean_controlled_size": metrics.get("mean_controlled_size"),
                    "mean_suppression_rate": metrics.get("mean_suppression_rate"),
                    "median_suppression_rate": metrics.get("median_suppression_rate"),
                    "mean_cost": metrics.get("mean_cost"),
                    "median_cost": metrics.get("median_cost"),
                    "mean_benefit_cost_ratio": metrics.get("mean_benefit_cost_ratio"),
                    "num_samples": metrics.get("num_samples"),
                    "source_file": str(path),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, items in sorted(grouped.items()):
        row = dict(zip(keys, key_values))
        row["n_seeds"] = len({item["seed"] for item in items})
        for metric in metrics:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) not in (None, "") and math.isfinite(float(item[metric]))
            ]
            mean_value, std_value = mean_std(values)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_report"] = fmt_pm(mean_value, std_value)
        output.append(row)
    return output


def display_name(mapping: list[tuple[str, str]], key: str) -> str:
    return dict(mapping).get(key, key)


def c2_paper_table(c2_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        subset = [
            row
            for row in c2_agg
            if row["dataset"] == dataset
            and row["split_strategy"] == "stratified"
            and row["model"] in dict(C2_MODELS)
        ]
        best_auc = max((row["auc_mean"] for row in subset), default=float("-inf"))
        for model, name in C2_MODELS:
            row = next((item for item in subset if item["model"] == model), None)
            if row is None:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": name,
                    "Seeds": row["n_seeds"],
                    "AUC": row["auc_report"],
                    "F1": row["f1_report"],
                    "Recall@10%": row["recall_at_10pct_report"],
                    "Lead time (min)": row["mean_lead_time_minutes_report"],
                    "Best AUC": "yes" if abs(row["auc_mean"] - best_auc) < 1e-12 else "",
                }
            )
    return rows


def c3_paper_table(c3_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        subset = [
            row
            for row in c3_agg
            if row["dataset"] == dataset
            and row["split_strategy"] == "stratified"
            and row["strategy"] in dict(C3_STRATEGIES)
        ]
        best_supp = max((row["mean_suppression_rate_mean"] for row in subset), default=float("-inf"))
        for strategy, name in C3_STRATEGIES:
            row = next((item for item in subset if item["strategy"] == strategy), None)
            if row is None:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Strategy": name,
                    "Seeds": row["n_seeds"],
                    "Suppression": row["mean_suppression_rate_report"],
                    "Cost": row["mean_cost_report"],
                    "Benefit/Cost": row["mean_benefit_cost_ratio_report"],
                    "Trigger rate": row["trigger_rate_report"],
                    "Best suppression": "yes" if abs(row["mean_suppression_rate_mean"] - best_supp) < 1e-12 else "",
                }
            )
    return rows


def temporal_table(c2_long: list[dict[str, Any]], c3_long: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        c2 = next(
            (
                row
                for row in c2_long
                if row["dataset"] == dataset
                and row["split_strategy"] == "temporal"
                and row["seed"] == 42
                and row["model"] == "heterorumor_c2"
            ),
            None,
        )
        c3 = next(
            (
                row
                for row in c3_long
                if row["dataset"] == dataset
                and row["split_strategy"] == "temporal"
                and row["seed"] == 42
                and row["strategy"] == "heterorumor_c3_event_pulse"
            ),
            None,
        )
        if c2 and c3:
            rows.append(
                {
                    "Dataset": dataset,
                    "C2 AUC": fmt_float(float(c2["auc"])),
                    "C2 F1": fmt_float(float(c2["f1"])),
                    "Lead time (min)": fmt_float(float(c2["mean_lead_time_minutes"]), 2),
                    "C3 suppression": fmt_float(float(c3["mean_suppression_rate"])),
                    "C3 cost": fmt_float(float(c3["mean_cost"])),
                    "C3 benefit/cost": fmt_float(float(c3["mean_benefit_cost_ratio"]), 2),
                }
            )
    return rows


def apply_style() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Re-run with --skip-plots to update tables and drafts only.")
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.4,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def get_agg(
    rows: list[dict[str, Any]],
    dataset: str,
    key_name: str,
    key_value: str,
    metric: str,
    split_strategy: str = "stratified",
) -> tuple[float, float]:
    row = next(
        (
            item
            for item in rows
            if item["dataset"] == dataset
            and item["split_strategy"] == split_strategy
            and item[key_name] == key_value
        ),
        None,
    )
    if row is None:
        return float("nan"), 0.0
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def grouped_bar(
    ax,
    datasets: list[str],
    series: list[tuple[str, list[float], list[float], str]],
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(datasets))
    width = 0.78 / max(len(series), 1)
    offset_start = -0.39 + width / 2
    for idx, (label, values, errors, color) in enumerate(series):
        positions = x + offset_start + idx * width
        ax.bar(
            positions,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            yerr=errors,
            capsize=3,
            error_kw={"linewidth": 1.0},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)


def save_figure(fig, basename: str) -> list[str]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Re-run with --skip-plots to update tables and drafts only.")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = FIGURE_DIR / f"{basename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_c2(c2_agg: list[dict[str, Any]]) -> list[str]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Re-run with --skip-plots to update tables and drafts only.")
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 7.5))
    axes = axes.ravel()
    colors = [PALETTE["neutral"], PALETTE["teal"], PALETTE["violet"], PALETTE["orange"], PALETTE["blue_main"]]

    for ax, metric, ylabel, title in [
        (axes[0], "auc", "AUC", "A. Breakout prediction AUC"),
        (axes[1], "f1", "F1", "B. Breakout prediction F1"),
    ]:
        series = []
        for (model, label), color in zip(C2_FIGURE_MODELS, colors):
            values, errors = [], []
            for dataset in DATASETS:
                mean_value, std_value = get_agg(c2_agg, dataset, "model", model, metric)
                values.append(mean_value)
                errors.append(std_value)
            series.append((label, values, errors, color))
        grouped_bar(ax, DATASETS, series, ylabel, title, ylim=(0.0, 1.05))

    lead_values, lead_errors = [], []
    for dataset in DATASETS:
        mean_value, std_value = get_agg(c2_agg, dataset, "model", "heterorumor_c2", "mean_lead_time_minutes")
        lead_values.append(mean_value)
        lead_errors.append(std_value)
    grouped_bar(
        axes[2],
        DATASETS,
        [("HeteroRumorDyn-C2", lead_values, lead_errors, PALETTE["blue_main"])],
        "Minutes",
        "C. Mean lead time",
    )

    drop_series = []
    for ablation, label, color in [
        ("heterorumor_c2_no_temporal_trend", "Full - w/o temporal trend", PALETTE["violet"]),
        ("heterorumor_c2_no_lowfreq", "Full - w/o low-freq", PALETTE["green_3"]),
        ("heterorumor_c2_no_cross", "Full - w/o cross-comm", PALETTE["red_2"]),
    ]:
        values, errors = [], []
        for dataset in DATASETS:
            full_mean, full_std = get_agg(c2_agg, dataset, "model", "heterorumor_c2", "auc")
            abl_mean, abl_std = get_agg(c2_agg, dataset, "model", ablation, "auc")
            values.append(full_mean - abl_mean)
            errors.append(math.sqrt(full_std**2 + abl_std**2))
        drop_series.append((label, values, errors, color))
    grouped_bar(axes[3], DATASETS, drop_series, "AUC delta", "D. Ablation effect on AUC")
    axes[3].axhline(0, color="#333333", linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[3].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=1.2)
    return save_figure(fig, "fig10_c2_breakout_multiseed")


def plot_c3(c3_agg: list[dict[str, Any]]) -> list[str]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Re-run with --skip-plots to update tables and drafts only.")
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 7.5))
    axes = axes.ravel()
    colors = [
        PALETTE["neutral"],
        PALETTE["teal"],
        PALETTE["red_2"],
        PALETTE["orange"],
        PALETTE["violet"],
        PALETTE["blue_main"],
    ]

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "mean_suppression_rate", "Suppression rate", "A. Control effectiveness", None),
        (axes[1], "mean_cost", "Intervention cost", "B. Control cost", None),
        (axes[2], "mean_benefit_cost_ratio", "Benefit / cost", "C. Cost-effectiveness (log scale)", None),
    ]:
        series = []
        for (strategy, label), color in zip(C3_FIGURE_STRATEGIES, colors):
            values, errors = [], []
            for dataset in DATASETS:
                mean_value, std_value = get_agg(c3_agg, dataset, "strategy", strategy, metric)
                values.append(mean_value)
                errors.append(std_value)
            series.append((label, values, errors, color))
        grouped_bar(ax, DATASETS, series, ylabel, title, ylim=ylim)
        if metric == "mean_benefit_cost_ratio":
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)

    drop_series = []
    for ablation, label, color in [
        ("heterorumor_c3_no_game", "Full - w/o game", PALETTE["green_3"]),
        ("heterorumor_c3_no_event_trigger", "Full - w/o trigger", PALETTE["red_2"]),
    ]:
        values, errors = [], []
        for dataset in DATASETS:
            full_mean, full_std = get_agg(c3_agg, dataset, "strategy", "heterorumor_c3_event_pulse", "mean_suppression_rate")
            abl_mean, abl_std = get_agg(c3_agg, dataset, "strategy", ablation, "mean_suppression_rate")
            values.append(full_mean - abl_mean)
            errors.append(math.sqrt(full_std**2 + abl_std**2))
        drop_series.append((label, values, errors, color))
    grouped_bar(axes[3], DATASETS, drop_series, "Suppression delta", "D. Ablation effect on suppression")
    axes[3].axhline(0, color="#333333", linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[3].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=1.2)
    return save_figure(fig, "fig11_c3_control_multiseed")


def result_paragraph(c2_paper: list[dict[str, Any]], c3_paper: list[dict[str, Any]], temporal_rows: list[dict[str, Any]]) -> str:
    def row_for(rows: list[dict[str, Any]], dataset: str, key: str, value: str) -> dict[str, Any]:
        return next(row for row in rows if row["Dataset"] == dataset and row[key] == value)

    pheme_c2 = row_for(c2_paper, "pheme", "Method", "HeteroRumorDyn-C2")
    twitter15_c2 = row_for(c2_paper, "twitter15", "Method", "HeteroRumorDyn-C2")
    twitter16_c2 = row_for(c2_paper, "twitter16", "Method", "HeteroRumorDyn-C2")
    pheme_no_cross = row_for(c2_paper, "pheme", "Method", "w/o cross-comm")
    twitter15_no_cross = row_for(c2_paper, "twitter15", "Method", "w/o cross-comm")
    twitter16_dynamic = row_for(c2_paper, "twitter16", "Method", "Dynamic trend")
    twitter16_no_cross = row_for(c2_paper, "twitter16", "Method", "w/o cross-comm")
    pheme_c3 = row_for(c3_paper, "pheme", "Strategy", "HeteroRumorDyn-C3")
    twitter15_c3 = row_for(c3_paper, "twitter15", "Strategy", "HeteroRumorDyn-C3")
    twitter16_c3 = row_for(c3_paper, "twitter16", "Strategy", "HeteroRumorDyn-C3")
    twitter15_random = row_for(c3_paper, "twitter15", "Strategy", "Random same-budget")
    twitter15_fixed_budget = row_for(c3_paper, "twitter15", "Strategy", "Fixed same-budget")
    twitter16_random = row_for(c3_paper, "twitter16", "Strategy", "Random same-budget")
    twitter16_fixed_budget = row_for(c3_paper, "twitter16", "Strategy", "Fixed same-budget")

    lines = [
        "# C2/C3 Results Explanation",
        "",
        "## Paper-ready paragraph",
        "",
        (
            "For the breakout forecasting task, HeteroRumorDyn-C2 consistently achieved strong early-warning "
            f"performance under five random seeds. On PHEME, Twitter15 and Twitter16, the model obtained AUC values of "
            f"{pheme_c2['AUC']}, {twitter15_c2['AUC']} and {twitter16_c2['AUC']}, respectively, while providing mean "
            f"lead times of {pheme_c2['Lead time (min)']}, {twitter15_c2['Lead time (min)']} and "
            f"{twitter16_c2['Lead time (min)']} minutes. The ablation rows show whether low-frequency energy and "
            "cross-community propagation signals contribute beyond static topology and short-term temporal trends. "
            "The dynamic-only and community-only rows further separate the explanatory power of temporal evolution "
            "and community-bridging cues."
        ),
        "",
        "## C2 ablation interpretation",
        "",
        (
            "The C2 ablation should be read as a feature trade-off rather than a strictly monotonic hierarchy. "
            f"On PHEME, removing the cross-community cue reduced AUC from {pheme_c2['AUC']} to "
            f"{pheme_no_cross['AUC']}, supporting the usefulness of community-bridging signals. On Twitter15, "
            f"the no-cross variant reached {twitter15_no_cross['AUC']}, slightly higher than the full model "
            f"({twitter15_c2['AUC']}); this suggests that the proxy community signal in the public Twitter15 "
            "tree is noisy and can be partly replaced by dynamic trend and structural statistics. On Twitter16, "
            f"the dynamic-trend baseline achieved {twitter16_dynamic['AUC']}, while removing cross-community "
            f"features dropped to {twitter16_no_cross['AUC']}; thus the temporal trend dominates AUC, but the "
            "cross-community cue still prevents a larger degradation and gives a useful early-warning trade-off. "
            "Weibo should be reported as a proxy sanity check because the current BiGCN preprocessed version lacks "
            "real timestamps and real social-community labels."
        ),
        "",
        (
            "For closed-loop control, the risk-triggered HeteroRumorDyn-C3 strategy improved the cost-effectiveness "
            "of intervention compared with fixed intervention and influence blocking. Its mean suppression rates on "
            f"PHEME, Twitter15 and Twitter16 were {pheme_c3['Suppression']}, {twitter15_c3['Suppression']} and "
            f"{twitter16_c3['Suppression']}, with intervention costs of {pheme_c3['Cost']}, {twitter15_c3['Cost']} "
            f"and {twitter16_c3['Cost']}. The no-game and no-trigger ablations separate the benefits of adaptive "
            "leader-follower response from those of event-triggered timing, while the random and fixed same-budget "
            "controls test whether the gains come from risk-aware targeting rather than simply spending more budget."
        ),
        "",
        "## C3 fair-control interpretation",
        "",
        (
            "The same-budget controls are the key fairness checks for C3. On Twitter15, HeteroRumorDyn-C3 achieved "
            f"{twitter15_c3['Suppression']} suppression at {twitter15_c3['Cost']} cost, compared with "
            f"{twitter15_random['Suppression']} for random same-budget and {twitter15_fixed_budget['Suppression']} "
            "for fixed same-budget intervention. On Twitter16, the corresponding values were "
            f"{twitter16_c3['Suppression']}, {twitter16_random['Suppression']} and "
            f"{twitter16_fixed_budget['Suppression']}. Therefore the reported control gain should be described as "
            "risk-aware timing and targeting under a matched budget, not as an artifact of spending more intervention "
            "cost."
        ),
        "",
        "## Notes for reporting",
        "",
        "- Weibo results should be described as heuristic/proxy evidence because the current BiGCN Weibo files lack real timestamps and real social-community labels.",
        "- Stratified results are the main tuning/stability setting; temporal/proxy split rows are leakage-safe stress tests.",
        "- C3 is currently a simulation driven by C2 risk scores and observed cascade snapshots, not a real platform intervention.",
        "- The case-study figure should be used to show both successful early warning and the false-alarm boundary.",
        "",
        "## Temporal/proxy seed42 snapshot",
        "",
    ]
    for row in temporal_rows:
        lines.append(
            f"- {row['Dataset']}: C2 AUC={row['C2 AUC']}, F1={row['C2 F1']}, "
            f"C3 suppression={row['C3 suppression']}, cost={row['C3 cost']}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Update summary tables and result drafts without regenerating matplotlib figures.",
    )
    args = parser.parse_args()

    c2_long = load_c2_long()
    c3_long = load_c3_long()

    c2_metrics = [
        "auc",
        "f1",
        "precision",
        "recall",
        "recall_at_10pct",
        "mean_lead_time_minutes",
        "warning_rate",
    ]
    c3_metrics = [
        "trigger_rate",
        "mean_suppression_rate",
        "mean_cost",
        "mean_benefit_cost_ratio",
        "mean_controlled_size",
    ]
    c2_agg = aggregate(c2_long, ["dataset", "split_strategy", "model"], c2_metrics)
    c3_agg = aggregate(c3_long, ["dataset", "split_strategy", "strategy"], c3_metrics)
    c2_paper = c2_paper_table(c2_agg)
    c3_paper = c3_paper_table(c3_agg)
    temporal_rows = temporal_table(c2_long, c3_long)

    write_csv(SUMMARY_DIR / "c2_breakout_all_runs.csv", c2_long, list(c2_long[0].keys()))
    write_csv(SUMMARY_DIR / "c3_control_all_runs.csv", c3_long, list(c3_long[0].keys()))
    write_csv(SUMMARY_DIR / "c2_breakout_multiseed_summary.csv", c2_agg, list(c2_agg[0].keys()))
    write_csv(SUMMARY_DIR / "c3_control_multiseed_summary.csv", c3_agg, list(c3_agg[0].keys()))

    c2_fields = ["Dataset", "Method", "Seeds", "AUC", "F1", "Recall@10%", "Lead time (min)", "Best AUC"]
    c3_fields = ["Dataset", "Strategy", "Seeds", "Suppression", "Cost", "Benefit/Cost", "Trigger rate", "Best suppression"]
    temporal_fields = ["Dataset", "C2 AUC", "C2 F1", "Lead time (min)", "C3 suppression", "C3 cost", "C3 benefit/cost"]
    for basename, rows, fields in [
        ("c2_breakout_paper_table", c2_paper, c2_fields),
        ("c3_control_paper_table", c3_paper, c3_fields),
        ("c2_c3_temporal_seed42_table", temporal_rows, temporal_fields),
    ]:
        write_csv(SUMMARY_DIR / f"{basename}.csv", rows, fields)
        write_markdown(SUMMARY_DIR / f"{basename}.md", rows, fields)
        write_latex(SUMMARY_DIR / f"{basename}.tex", rows, fields)

    c2_figures: list[str] = []
    c3_figures: list[str] = []
    if not args.skip_plots:
        c2_figures = plot_c2(c2_agg)
        c3_figures = plot_c3(c3_agg)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    (DRAFT_DIR / "c2_c3_results_explanation.md").write_text(
        result_paragraph(c2_paper, c3_paper, temporal_rows),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "c2_table": str(SUMMARY_DIR / "c2_breakout_paper_table.csv"),
                "c3_table": str(SUMMARY_DIR / "c3_control_paper_table.csv"),
                "temporal_table": str(SUMMARY_DIR / "c2_c3_temporal_seed42_table.csv"),
                "figures": c2_figures + c3_figures,
                "draft": str(DRAFT_DIR / "c2_c3_results_explanation.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
