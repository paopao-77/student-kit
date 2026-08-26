import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("data/processed/weibo")
CASE_DIR = Path("results/case_studies")
FIGURE_DIR = Path("results/figures")
DRAFTS = Path("results/drafts")
C2_PREDICTIONS = Path(
    "results/c2_breakout_weibo_raw_ow50/weibo_breakout_stratified_seed42_predictions.csv"
)
C3_SIMULATIONS = Path(
    "results/c3_control_weibo_raw_ow50/"
    "weibo_control_stratified_heterorumor_c2_seed42_simulations.csv"
)

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


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def load_c2_rows() -> dict[str, dict[str, str]]:
    rows = [
        row
        for row in read_csv(C2_PREDICTIONS)
        if row.get("split") == "test" and row.get("model") == "heterorumor_c2"
    ]
    return {row["sample_id"]: row for row in rows}


def load_c3_rows() -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(C3_SIMULATIONS):
        grouped[row["sample_id"]][row["strategy"]] = row
    return dict(grouped)


def load_snapshots() -> dict[str, list[dict[str, str]]]:
    grouped = group_rows(read_csv(DATA_ROOT / "dynamic_snapshots" / "snapshots.csv"), "sample_id")
    for rows in grouped.values():
        rows.sort(key=lambda row: as_int(row.get("window_index"), 0))
    return grouped


def load_samples() -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_csv(DATA_ROOT / "samples.csv")}


def snapshot_pressure(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    cumulative = np.asarray([as_float(row.get("cumulative_nodes"), 0.0) for row in rows], dtype=float)
    new_communities = np.asarray([as_float(row.get("new_communities"), 0.0) for row in rows], dtype=float)
    branch_ratio = np.asarray([as_float(row.get("branch_community_ratio"), 0.0) for row in rows], dtype=float)
    cumulative_component = cumulative / max(float(cumulative.max()), 1.0)
    community_component = np.log1p(new_communities) / max(float(np.log1p(new_communities).max()), 1.0)
    pressure = 0.50 * cumulative_component + 0.30 * branch_ratio + 0.20 * community_component
    pressure = np.clip(pressure, 0.0, 1.0)
    return [float(value) for value in pressure]


def metric_bundle(sample_id: str, c2: dict[str, dict[str, str]], c3: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    c2_row = c2[sample_id]
    strategies = c3.get(sample_id, {})
    event = strategies.get("heterorumor_c3_event_pulse", {})
    random_budget = strategies.get("random_same_budget", {})
    fixed_budget = strategies.get("fixed_same_budget", {})
    return {
        "sample_id": sample_id,
        "label_id": as_int(c2_row.get("label_id")),
        "pred_label_id": as_int(c2_row.get("pred_label_id")),
        "c2_score": as_float(c2_row.get("score_label_1")),
        "first_warning_window": as_int(c2_row.get("first_warning_window")),
        "breakout_window": as_int(c2_row.get("breakout_window")),
        "lead_time": as_float(c2_row.get("lead_time_minutes"), 0.0),
        "num_eval_windows": as_int(c2_row.get("num_eval_windows"), 0),
        "c3_triggered": as_int(event.get("triggered"), 0),
        "c3_trigger_window": as_int(event.get("trigger_window")),
        "c3_effective_window": as_int(event.get("effective_window")),
        "baseline_size": as_float(event.get("baseline_size"), as_float(random_budget.get("baseline_size"), 0.0)),
        "controlled_size": as_float(event.get("controlled_size"), 0.0),
        "event_suppression": as_float(event.get("suppression_rate"), 0.0),
        "event_cost": as_float(event.get("cost"), 0.0),
        "random_suppression": as_float(random_budget.get("suppression_rate"), 0.0),
        "fixed_suppression": as_float(fixed_budget.get("suppression_rate"), 0.0),
    }


def select_cases(
    c2: dict[str, dict[str, str]],
    c3: dict[str, dict[str, dict[str, str]]],
    snapshots: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    bundles = [metric_bundle(sample_id, c2, c3) for sample_id in sorted(set(c2) & set(c3))]
    for row in bundles:
        row["num_snapshot_windows"] = len(snapshots.get(row["sample_id"], []))
    true_positives = [
        row
        for row in bundles
        if row["label_id"] == 1
        and row["pred_label_id"] == 1
        and row["lead_time"] > 0
        and 1 <= row["breakout_window"] <= 300
        and row["num_snapshot_windows"] >= 4
    ]
    high_control = [
        row
        for row in bundles
        if row["event_suppression"] > 0 and row["c3_triggered"] == 1 and row["num_snapshot_windows"] >= 4
    ]
    false_alarms = [
        row for row in bundles if row["label_id"] == 0 and row["pred_label_id"] == 1 and row["num_snapshot_windows"] >= 4
    ]
    missed = [
        row for row in bundles if row["label_id"] == 1 and row["pred_label_id"] == 0 and row["num_snapshot_windows"] >= 4
    ]

    selected: list[tuple[str, dict[str, Any]]] = []

    def add(kind: str, candidates: list[dict[str, Any]], key) -> None:
        used = {row["sample_id"] for _kind, row in selected}
        available = [row for row in candidates if row["sample_id"] not in used]
        if available:
            selected.append((kind, max(available, key=key)))

    add("early_warning_success", true_positives, lambda row: (row["lead_time"], row["c2_score"]))
    add(
        "high_control_gain",
        high_control,
        lambda row: (row["event_suppression"] - row["random_suppression"], row["event_suppression"]),
    )
    add("false_alarm_challenge", false_alarms, lambda row: row["c2_score"])
    add("missed_warning_challenge", missed, lambda row: row["baseline_size"])
    if len(selected) < 4:
        add("additional_control_success", high_control, lambda row: row["event_suppression"])

    case_rows = []
    for rank, (kind, row) in enumerate(selected[:4], start=1):
        case = dict(row)
        case["case_id"] = f"W{rank}"
        case["case_type"] = kind
        case_rows.append(case)
    return case_rows


def enrich_cases(
    cases: list[dict[str, Any]],
    samples: dict[str, dict[str, str]],
    snapshots: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    enriched = []
    for case in cases:
        sample = samples.get(case["sample_id"], {})
        rows = snapshots.get(case["sample_id"], [])
        final_nodes = as_float(sample.get("num_nodes"), as_float(case.get("baseline_size"), 0.0))
        peak_new_nodes = max([as_float(row.get("new_nodes"), 0.0) for row in rows] or [0.0])
        out = dict(case)
        out.update(
            {
                "num_nodes": as_int(sample.get("num_nodes"), as_int(case.get("baseline_size"), 0)),
                "num_edges": as_int(sample.get("num_edges"), 0),
                "has_source_text": as_int(sample.get("has_source_text"), 0),
                "num_snapshot_windows": len(rows),
                "peak_new_nodes": peak_new_nodes,
                "suppressed_nodes": max(0.0, as_float(case.get("baseline_size"), final_nodes) - as_float(case.get("controlled_size"), 0.0)),
            }
        )
        enriched.append(out)
    return enriched


def build_curve_rows(cases: list[dict[str, Any]], snapshots: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    curve_rows = []
    for case in cases:
        rows = snapshots.get(case["sample_id"], [])
        pressures = snapshot_pressure(rows)
        for row, pressure in zip(rows, pressures):
            curve_rows.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "sample_id": case["sample_id"],
                    "window_index": as_int(row.get("window_index")),
                    "window_start": as_float(row.get("window_start")),
                    "window_end": as_float(row.get("window_end")),
                    "new_nodes": as_float(row.get("new_nodes")),
                    "cumulative_nodes": as_float(row.get("cumulative_nodes")),
                    "branch_community_ratio": as_float(row.get("branch_community_ratio")),
                    "new_communities": as_float(row.get("new_communities")),
                    "snapshot_pressure": pressure,
                    "first_warning_window": case["first_warning_window"],
                    "breakout_window": case["breakout_window"],
                    "c3_trigger_window": case["c3_trigger_window"],
                    "c3_effective_window": case["c3_effective_window"],
                }
            )
    return curve_rows


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_marker(ax: plt.Axes, value: int, label: str, color: str) -> None:
    if value < 0:
        return
    ax.axvline(value, color=color, linewidth=1.2, linestyle="--")


def plot_cases(cases: list[dict[str, Any]], snapshots: dict[str, list[dict[str, str]]]) -> list[str]:
    publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    axes = axes.flatten()
    for ax, case in zip(axes, cases):
        rows = snapshots.get(case["sample_id"], [])
        if not rows:
            ax.set_axis_off()
            continue
        x = np.asarray([as_int(row.get("window_index")) for row in rows], dtype=float)
        cumulative = np.asarray([as_float(row.get("cumulative_nodes"), 0.0) for row in rows], dtype=float)
        pressure = np.asarray(snapshot_pressure(rows), dtype=float)
        ax.plot(x, cumulative, color=PALETTE["blue"], linewidth=1.8, marker="o", markersize=3.2, label="Cumulative nodes")
        ax.set_xlabel("Event-order window")
        ax.set_ylabel("Cumulative nodes")
        ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
        twin = ax.twinx()
        twin.plot(x, pressure, color=PALETTE["orange"], linewidth=1.5, label="Snapshot pressure")
        twin.set_ylim(0, 1.05)
        twin.set_ylabel("Pressure")
        add_marker(ax, as_int(case.get("first_warning_window")), "warning", PALETTE["red"])
        add_marker(ax, as_int(case.get("breakout_window")), "breakout", PALETTE["gray"])
        add_marker(ax, as_int(case.get("c3_trigger_window")), "C3", PALETTE["green"])
        if len(x) == 1:
            ax.set_xlim(float(x[0]) - 1.0, float(x[0]) + 1.0)
        title = (
            f"{case['case_id']} {case['case_type'].replace('_', ' ')}\n"
            f"score={case['c2_score']:.3f}, lead={case['lead_time']:.0f}, "
            f"supp={case['event_suppression']:.3f}"
        )
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
        lines, labels = ax.get_legend_handles_labels()
        twin_lines, twin_labels = twin.get_legend_handles_labels()
        ax.legend(lines + twin_lines, labels + twin_labels, loc="lower right", fontsize=7.2)
    fig.tight_layout(w_pad=2.1, h_pad=2.4)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf", "svg"):
        path = FIGURE_DIR / f"fig_weibo_raw_e10_case_studies.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def write_note(cases: list[dict[str, Any]], outputs: list[str]) -> None:
    lines = [
        "# Raw Weibo E10 Case Studies",
        "",
        "## Case Selection",
        "",
        "- Source results: preferred raw-Weibo C2/C3 seed42 with `order_window_size=50`.",
        "- Cases are selected from the stratified test split using C2 prediction correctness, lead time, and C3 suppression gain.",
        "- The plotted pressure curve is a reproducible snapshot diagnostic from cumulative nodes, new communities, and branch-community ratio; it is not an additional trained model.",
        "",
        "## Selected Cases",
        "",
        "| case | type | sample_id | label | pred | C2 score | lead | C3 suppression | random suppression |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['case_id']} | {case['case_type']} | {case['sample_id']} | "
            f"{case['label_id']} | {case['pred_label_id']} | {case['c2_score']:.4f} | "
            f"{case['lead_time']:.0f} | {case['event_suppression']:.4f} | {case['random_suppression']:.4f} |"
        )
    lines.extend(["", "## Outputs", ""])
    lines.extend(f"- {path}" for path in outputs)
    lines.extend(
        [
            "- results/case_studies/weibo_raw_e10_cases.csv",
            "- results/case_studies/weibo_raw_e10_case_curves.csv",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_e10_case_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c2 = load_c2_rows()
    c3 = load_c3_rows()
    snapshots = load_snapshots()
    samples = load_samples()
    cases = enrich_cases(select_cases(c2, c3, snapshots), samples, snapshots)
    curves = build_curve_rows(cases, snapshots)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        CASE_DIR / "weibo_raw_e10_cases.csv",
        cases,
        [
            "case_id",
            "case_type",
            "sample_id",
            "label_id",
            "pred_label_id",
            "c2_score",
            "first_warning_window",
            "breakout_window",
            "lead_time",
            "num_eval_windows",
            "c3_triggered",
            "c3_trigger_window",
            "c3_effective_window",
            "baseline_size",
            "controlled_size",
            "suppressed_nodes",
            "event_suppression",
            "event_cost",
            "random_suppression",
            "fixed_suppression",
            "num_nodes",
            "num_edges",
            "has_source_text",
            "num_snapshot_windows",
            "peak_new_nodes",
        ],
    )
    write_csv(
        CASE_DIR / "weibo_raw_e10_case_curves.csv",
        curves,
        [
            "case_id",
            "case_type",
            "sample_id",
            "window_index",
            "window_start",
            "window_end",
            "new_nodes",
            "cumulative_nodes",
            "branch_community_ratio",
            "new_communities",
            "snapshot_pressure",
            "first_warning_window",
            "breakout_window",
            "c3_trigger_window",
            "c3_effective_window",
        ],
    )
    outputs = plot_cases(cases, snapshots)
    write_note(cases, outputs)
    print(
        {
            "cases": str(CASE_DIR / "weibo_raw_e10_cases.csv"),
            "curves": str(CASE_DIR / "weibo_raw_e10_case_curves.csv"),
            "figures": outputs,
            "note": str(DRAFTS / "weibo_raw_e10_case_analysis.md"),
        }
    )


if __name__ == "__main__":
    main()
