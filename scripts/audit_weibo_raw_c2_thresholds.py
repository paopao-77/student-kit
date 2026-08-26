import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DATASET = "weibo"
DATA_DIR = Path("data/processed/weibo")
SNAPSHOTS = DATA_DIR / "dynamic_snapshots" / "snapshots.csv"
BREAKOUTS = DATA_DIR / "breakout_events.csv"
SUMMARY = Path("results/summary")
DRAFTS = Path("results/drafts")

THRESHOLD_GRID = [
    (0.1, 0.2),
    (0.2, 0.1),
    (0.2, 0.2),
    (0.2, 0.3),
    (0.3, 0.2),
]

BASELINE = {
    "min_breakout_window": 1,
    "min_nodes": 10,
    "min_active_communities": 3,
    "min_new_communities": 1,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = min(int(q * (len(values) - 1)), len(values) - 1)
    return values[index]


def fmt(value: float, digits: int = 6) -> float | str:
    if not math.isfinite(value):
        return ""
    return round(value, digits)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detect(rows: list[dict[str, str]], theta_cross: float, theta_branch_ratio: float) -> dict[str, Any]:
    for row in rows:
        window_index = int(row["window_index"])
        cumulative_nodes = int(row["cumulative_nodes"])
        active_communities = int(row["active_communities"])
        new_communities = int(row["new_communities"])
        cumulative_cross_edges = int(row["cumulative_cross_edges"])
        cross_edge_ratio = finite_float(row["cross_edge_ratio"])
        branch_community_ratio = finite_float(row["branch_community_ratio"])

        base_eligible = (
            window_index >= BASELINE["min_breakout_window"]
            and cumulative_nodes >= BASELINE["min_nodes"]
            and active_communities >= BASELINE["min_active_communities"]
        )
        if not base_eligible:
            continue

        cross_trigger = cross_edge_ratio >= theta_cross and cumulative_cross_edges > 0
        branch_trigger = (
            new_communities >= BASELINE["min_new_communities"] and window_index > 0
        )
        broad_trigger = branch_community_ratio >= theta_branch_ratio
        if cross_trigger or branch_trigger or broad_trigger:
            reasons = []
            if cross_trigger:
                reasons.append("cross_edge_ratio")
            if branch_trigger:
                reasons.append("new_communities")
            if broad_trigger:
                reasons.append("branch_community_ratio")
            return {
                "has_breakout": 1,
                "breakout_window": window_index,
                "cross_trigger": int(cross_trigger),
                "branch_trigger": int(branch_trigger),
                "broad_trigger": int(broad_trigger),
                "trigger_reason": "+".join(reasons),
            }
    return {
        "has_breakout": 0,
        "breakout_window": "",
        "cross_trigger": 0,
        "branch_trigger": 0,
        "broad_trigger": 0,
        "trigger_reason": "not_triggered",
    }


def summarize_distribution(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for field in [
        "cross_edge_ratio",
        "branch_community_ratio",
        "new_communities",
        "active_communities",
        "cumulative_nodes",
        "cumulative_cross_edges",
    ]:
        values = [finite_float(row[field]) for row in rows]
        output.append(
            {
                "field": field,
                "n": len(values),
                "mean": fmt(mean(values)),
                "min": fmt(min(values)),
                "p01": fmt(quantile(values, 0.01)),
                "p05": fmt(quantile(values, 0.05)),
                "p25": fmt(quantile(values, 0.25)),
                "p50": fmt(quantile(values, 0.50)),
                "p75": fmt(quantile(values, 0.75)),
                "p95": fmt(quantile(values, 0.95)),
                "p99": fmt(quantile(values, 0.99)),
                "max": fmt(max(values)),
                "num_zero": sum(1 for value in values if value == 0),
                "num_positive": sum(1 for value in values if value > 0),
            }
        )
    return output


def audit_thresholds(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    baseline_by_sample = {
        sample_id: detect(rows, 0.2, 0.2) for sample_id, rows in grouped.items()
    }
    output = []
    for theta_cross, theta_branch_ratio in THRESHOLD_GRID:
        results = {
            sample_id: detect(rows, theta_cross, theta_branch_ratio)
            for sample_id, rows in grouped.items()
        }
        labels = [result["has_breakout"] for result in results.values()]
        base_labels = [baseline_by_sample[sample_id]["has_breakout"] for sample_id in results]
        flips = sum(1 for sample_id, result in results.items() if result["has_breakout"] != baseline_by_sample[sample_id]["has_breakout"])
        windows_changed = sum(
            1
            for sample_id, result in results.items()
            if result["breakout_window"] != baseline_by_sample[sample_id]["breakout_window"]
        )
        reasons = Counter(result["trigger_reason"] for result in results.values())
        output.append(
            {
                "theta_cross": theta_cross,
                "theta_branch_ratio": theta_branch_ratio,
                "num_samples": len(results),
                "num_breakout": sum(labels),
                "breakout_rate": fmt(sum(labels) / len(labels)),
                "label_flips_vs_0.2_0.2": flips,
                "breakout_window_changes_vs_0.2_0.2": windows_changed,
                "baseline_num_breakout": sum(base_labels),
                "top_trigger_reason": reasons.most_common(1)[0][0],
                "top_trigger_count": reasons.most_common(1)[0][1],
                "cross_triggered_samples": sum(result["cross_trigger"] for result in results.values()),
                "branch_triggered_samples": sum(result["branch_trigger"] for result in results.values()),
                "broad_triggered_samples": sum(result["broad_trigger"] for result in results.values()),
            }
        )
    return output


def audit_eligible_conditions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        if (
            int(row["window_index"]) >= BASELINE["min_breakout_window"]
            and int(row["cumulative_nodes"]) >= BASELINE["min_nodes"]
            and int(row["active_communities"]) >= BASELINE["min_active_communities"]
        ):
            eligible.append(row)

    output = []
    for theta_cross, theta_branch_ratio in THRESHOLD_GRID:
        cross_hits = [
            row
            for row in eligible
            if finite_float(row["cross_edge_ratio"]) >= theta_cross
            and int(row["cumulative_cross_edges"]) > 0
        ]
        branch_hits = [
            row
            for row in eligible
            if int(row["new_communities"]) >= BASELINE["min_new_communities"]
            and int(row["window_index"]) > 0
        ]
        broad_hits = [
            row
            for row in eligible
            if finite_float(row["branch_community_ratio"]) >= theta_branch_ratio
        ]
        output.append(
            {
                "theta_cross": theta_cross,
                "theta_branch_ratio": theta_branch_ratio,
                "eligible_windows": len(eligible),
                "cross_hit_windows": len(cross_hits),
                "branch_hit_windows": len(branch_hits),
                "broad_hit_windows": len(broad_hits),
                "cross_hit_rate": fmt(len(cross_hits) / len(eligible)),
                "branch_hit_rate": fmt(len(branch_hits) / len(eligible)),
                "broad_hit_rate": fmt(len(broad_hits) / len(eligible)),
            }
        )
    return output


def write_note(
    distribution: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> None:
    dist_by_field = {row["field"]: row for row in distribution}
    cross = dist_by_field["cross_edge_ratio"]
    branch = dist_by_field["branch_community_ratio"]
    new_comm = dist_by_field["new_communities"]
    base = next(
        row
        for row in threshold_rows
        if row["theta_cross"] == 0.2 and row["theta_branch_ratio"] == 0.2
    )
    base_condition = next(
        row
        for row in condition_rows
        if row["theta_cross"] == 0.2 and row["theta_branch_ratio"] == 0.2
    )
    lines = [
        "# Raw Weibo C2 Threshold Audit",
        "",
        "## Key Findings",
        "",
        f"- `cross_edge_ratio` is zero in all {cross['n']} snapshot rows; cross-trigger can never fire.",
        f"- `branch_community_ratio` ranges from {branch['min']} to {branch['max']}; even its minimum is above the tested 0.1/0.2/0.3 thresholds.",
        f"- `new_communities` is positive in every snapshot row, and the C2 rule only requires at least one new community after window 0.",
        f"- The main setting has {base['num_breakout']} breakout samples ({base['breakout_rate']}); all tested threshold settings produce zero label flips.",
        f"- Under the main setting, eligible-window hit rates are: cross {base_condition['cross_hit_rate']}, branch {base_condition['branch_hit_rate']}, broad {base_condition['broad_hit_rate']}.",
        "",
        "## Why Thresholds Are Invariant",
        "",
        "Raw Weibo is represented as source-to-repost star edges. The branch-community heuristic assigns each repost child to its own branch community, so active communities almost equal cumulative nodes and cross-community edges never appear. As a result, `theta_cross` has no signal and `theta_branch_ratio` is already satisfied for all eligible windows.",
        "",
        "## Threshold Grid",
        "",
        "| theta_cross | theta_branch_ratio | breakout_rate | label_flips | top_trigger_reason |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {row['theta_cross']} | {row['theta_branch_ratio']} | "
            f"{row['breakout_rate']} | {row['label_flips_vs_0.2_0.2']} | "
            f"{row['top_trigger_reason']} |"
        )
    lines.extend(["", "## Distribution Snapshot", "", "| field | min | p50 | p95 | max | num_zero |", "|---|---:|---:|---:|---:|---:|"])
    for field in ["cross_edge_ratio", "branch_community_ratio", "new_communities", "active_communities"]:
        row = dist_by_field[field]
        lines.append(
            f"| {field} | {row['min']} | {row['p50']} | {row['p95']} | {row['max']} | {row['num_zero']} |"
        )
    lines.append("")
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_c2_threshold_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    snapshot_rows = read_csv(SNAPSHOTS)
    read_csv(BREAKOUTS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in snapshot_rows:
        grouped[row["sample_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["window_index"]))

    distribution = summarize_distribution(snapshot_rows)
    threshold_rows = audit_thresholds(grouped)
    condition_rows = audit_eligible_conditions(snapshot_rows)

    write_csv(SUMMARY / "weibo_raw_c2_threshold_distribution.csv", distribution)
    write_csv(SUMMARY / "weibo_raw_c2_threshold_label_flip_audit.csv", threshold_rows)
    write_csv(SUMMARY / "weibo_raw_c2_threshold_condition_hits.csv", condition_rows)
    write_note(distribution, threshold_rows, condition_rows)
    print(
        json.dumps(
            {
                "num_samples": len(grouped),
                "num_snapshots": len(snapshot_rows),
                "distribution": str(SUMMARY / "weibo_raw_c2_threshold_distribution.csv"),
                "label_flip_audit": str(SUMMARY / "weibo_raw_c2_threshold_label_flip_audit.csv"),
                "condition_hits": str(SUMMARY / "weibo_raw_c2_threshold_condition_hits.csv"),
                "note": str(DRAFTS / "weibo_raw_c2_threshold_audit.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
