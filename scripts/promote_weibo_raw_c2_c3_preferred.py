import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


RESULTS = Path("results")
SUMMARY = RESULTS / "summary"
DRAFTS = RESULTS / "drafts"

PREFERRED_ORDER_WINDOW_SIZE = 50
C2_DIR = RESULTS / "c2_breakout_weibo_raw_ow50"
C3_DIR = RESULTS / "c3_control_weibo_raw_ow50"
SENSITIVITY = SUMMARY / "weibo_raw_c2_c3_order_window_sensitivity.csv"
THRESHOLD_AUDIT = DRAFTS / "weibo_raw_c2_threshold_audit.md"

C2_METRICS = [
    "auc",
    "f1",
    "macro_f1",
    "precision_at_10pct",
    "recall_at_10pct",
    "mean_lead_time_minutes",
    "warning_rate",
]
C3_METRICS = [
    "trigger_rate",
    "mean_suppression_rate",
    "mean_cost",
    "mean_benefit_cost_ratio",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> float | str:
    if not math.isfinite(value):
        return ""
    return round(value, 6)


def aggregate(rows: list[dict[str, Any]], group_key: str, metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    output = []
    for group, items in sorted(grouped.items()):
        seeds = sorted({int(item["seed"]) for item in items})
        out: dict[str, Any] = {
            group_key: group,
            "preferred_order_window_size": PREFERRED_ORDER_WINDOW_SIZE,
            "n_seeds": len(seeds),
            "seeds": " ".join(map(str, seeds)),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) not in (None, "")]
            avg = mean(values) if values else float("nan")
            sd = stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_mean"] = fmt(avg)
            out[f"{metric}_std"] = fmt(sd)
        output.append(out)
    return output


def load_c2() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(C2_DIR.glob("*_metrics.json")):
        payload = read_json(path)
        for model, splits in payload.get("models", {}).items():
            test = splits.get("test", {})
            row = {
                "dataset": payload.get("dataset"),
                "split_strategy": payload.get("split_strategy"),
                "seed": payload.get("seed"),
                "model": model,
                "source_file": str(path),
            }
            row.update({metric: test.get(metric) for metric in C2_METRICS})
            rows.append(row)
    if len({row["source_file"] for row in rows}) != 5:
        raise FileNotFoundError(f"Expected five C2 seed files under {C2_DIR}")
    return rows


def load_c3() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(C3_DIR.glob("*_metrics.json")):
        payload = read_json(path)
        for strategy, metrics in payload.get("strategies", {}).items():
            row = {
                "dataset": payload.get("dataset"),
                "split_strategy": payload.get("split_strategy"),
                "seed": payload.get("seed"),
                "strategy": strategy,
                "source_file": str(path),
            }
            row.update({metric: metrics.get(metric) for metric in C3_METRICS})
            rows.append(row)
    if len({row["source_file"] for row in rows}) != 5:
        raise FileNotFoundError(f"Expected five C3 seed files under {C3_DIR}")
    return rows


def write_note(c2_summary: list[dict[str, Any]], c3_summary: list[dict[str, Any]]) -> None:
    c2_main = next(row for row in c2_summary if row["model"] == "heterorumor_c2")
    c3_main = next(row for row in c3_summary if row["strategy"] == "heterorumor_c3_event_pulse")
    c3_random = next(row for row in c3_summary if row["strategy"] == "random_same_budget")
    c3_fixed = next(row for row in c3_summary if row["strategy"] == "fixed_same_budget")
    lines = [
        "# Raw Weibo C2/C3 Preferred Setting",
        "",
        f"Preferred `order_window_size`: `{PREFERRED_ORDER_WINDOW_SIZE}` events.",
        "",
        "## Reason",
        "",
        "- Order-window sensitivity showed that 50 events outperformed 100 and 200 on C2 AUC, C2 F1, and C3 event-pulse suppression.",
        "- Threshold sensitivity showed no variation across the tested `theta_cross` and `theta_branch_ratio` settings.",
        "- The threshold audit explains the invariance: raw Weibo is currently a source-to-repost star-edge proxy, so cross-community edges never appear and branch-community ratios are already above the tested thresholds.",
        "",
        "## Preferred Five-Seed Results",
        "",
        f"- C2 `heterorumor_c2`: AUC {c2_main['auc_mean']} +/- {c2_main['auc_std']}; F1 {c2_main['f1_mean']} +/- {c2_main['f1_std']}.",
        f"- C3 `heterorumor_c3_event_pulse`: suppression {c3_main['mean_suppression_rate_mean']} +/- {c3_main['mean_suppression_rate_std']}; cost {c3_main['mean_cost_mean']} +/- {c3_main['mean_cost_std']}.",
        f"- Same-budget controls: random suppression {c3_random['mean_suppression_rate_mean']}; fixed suppression {c3_fixed['mean_suppression_rate_mean']}.",
        "",
        "## Traceability",
        "",
        f"- Sensitivity table: `{SENSITIVITY}`.",
        f"- Threshold audit: `{THRESHOLD_AUDIT}`.",
        f"- Preferred C2 result directory: `{C2_DIR}`.",
        f"- Preferred C3 result directory: `{C3_DIR}`.",
        "",
    ]
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_c2_c3_preferred_setting.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    c2_rows = load_c2()
    c3_rows = load_c3()
    c2_summary = aggregate(c2_rows, "model", C2_METRICS)
    c3_summary = aggregate(c3_rows, "strategy", C3_METRICS)
    write_csv(SUMMARY / "c2_breakout_weibo_raw_preferred_all_runs.csv", c2_rows)
    write_csv(SUMMARY / "c2_breakout_weibo_raw_preferred_summary.csv", c2_summary)
    write_csv(SUMMARY / "c3_control_weibo_raw_preferred_all_runs.csv", c3_rows)
    write_csv(SUMMARY / "c3_control_weibo_raw_preferred_summary.csv", c3_summary)
    write_note(c2_summary, c3_summary)
    print(
        json.dumps(
            {
                "preferred_order_window_size": PREFERRED_ORDER_WINDOW_SIZE,
                "c2_summary": str(SUMMARY / "c2_breakout_weibo_raw_preferred_summary.csv"),
                "c3_summary": str(SUMMARY / "c3_control_weibo_raw_preferred_summary.csv"),
                "note": str(DRAFTS / "weibo_raw_c2_c3_preferred_setting.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
