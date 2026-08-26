import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RESULTS_DIR = Path("results") / "c3_control"
SUMMARY_DIR = Path("results") / "summary"
DATASETS = ["pheme", "twitter15", "twitter16", "weibo"]

STRATEGIES = [
    ("fixed_same_budget", "Fixed same-budget"),
    ("random_same_budget", "Random same-budget"),
    ("ed_id_internal_only", "ED-ID internal-only"),
    ("ed_id_external_only", "ED-ID external-only"),
    ("ed_id_adapted", "ED-ID-adapted"),
    ("ed_id_adapted_same_budget", "ED-ID same-budget"),
    ("heterorumor_c3_event_pulse", "HeteroRumorDyn-C3"),
]

METRICS = [
    "trigger_rate",
    "mean_suppression_rate",
    "mean_cost",
    "mean_benefit_cost_ratio",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [
        "\\begin{tabular}{" + "l" * len(fields) + "}",
        "\\toprule",
        " & ".join(fields) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("_", "\\_") for field in fields]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def fmt(value: float, digits: int = 4) -> str:
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def fmt_pm(mean_value: float, std_value: float, digits: int = 4) -> str:
    if not math.isfinite(mean_value):
        return ""
    if std_value > 0 and math.isfinite(std_value):
        return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"
    return f"{mean_value:.{digits}f}"


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        payload = read_json(path)
        for strategy, metrics in payload.get("strategies", {}).items():
            if strategy not in dict(STRATEGIES):
                continue
            rows.append(
                {
                    "dataset": payload.get("dataset", ""),
                    "split_strategy": payload.get("split_strategy", ""),
                    "seed": int(payload.get("seed", 0)),
                    "strategy": strategy,
                    **{metric: metrics.get(metric) for metric in METRICS},
                    "source_file": str(path),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["split_strategy"], row["strategy"])].append(row)

    output = []
    for (dataset, split_strategy, strategy), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "dataset": dataset,
            "split_strategy": split_strategy,
            "strategy": strategy,
            "n_seeds": len({item["seed"] for item in items}),
        }
        for metric in METRICS:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) not in (None, "")
                and math.isfinite(float(item[metric]))
            ]
            mean_value, std_value = mean_std(values)
            result[f"{metric}_mean"] = mean_value
            result[f"{metric}_std"] = std_value
            result[f"{metric}_report"] = fmt_pm(mean_value, std_value)
        output.append(result)
    return output


def paper_rows(aggregated: list[dict[str, Any]], split_strategy: str) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        subset = [
            row
            for row in aggregated
            if row["dataset"] == dataset and row["split_strategy"] == split_strategy
        ]
        c3 = next(
            (
                row
                for row in subset
                if row["strategy"] == "heterorumor_c3_event_pulse"
            ),
            None,
        )
        c3_suppression = (
            float(c3["mean_suppression_rate_mean"]) if c3 is not None else math.nan
        )
        c3_cost = float(c3["mean_cost_mean"]) if c3 is not None else math.nan
        for strategy, display_name in STRATEGIES:
            row = next((item for item in subset if item["strategy"] == strategy), None)
            if row is None:
                continue
            suppression = float(row["mean_suppression_rate_mean"])
            cost = float(row["mean_cost_mean"])
            rows.append(
                {
                    "Dataset": dataset,
                    "Strategy": display_name,
                    "Seeds": row["n_seeds"],
                    "Suppression": row["mean_suppression_rate_report"],
                    "Cost": row["mean_cost_report"],
                    "Benefit/Cost": row["mean_benefit_cost_ratio_report"],
                    "Trigger rate": row["trigger_rate_report"],
                    "Suppression gap vs C3": fmt(suppression - c3_suppression),
                    "Cost gap vs C3": fmt(cost - c3_cost),
                }
            )
    return rows


def write_bundle(stem: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(SUMMARY_DIR / f"{stem}.csv", rows, fields)
    write_markdown(SUMMARY_DIR / f"{stem}.md", rows, fields)
    write_latex(SUMMARY_DIR / f"{stem}.tex", rows, fields)


def main() -> None:
    long_rows = load_rows()
    if not long_rows:
        raise FileNotFoundError(f"No C3 metrics found under {RESULTS_DIR}")
    required = {"ed_id_adapted", "ed_id_adapted_same_budget"}
    available = {row["strategy"] for row in long_rows}
    missing = required - available
    if missing:
        raise RuntimeError(
            "ED-ID results are missing. Rerun scripts/simulate_c3_control.py first: "
            + ", ".join(sorted(missing))
        )

    long_fields = [
        "dataset",
        "split_strategy",
        "seed",
        "strategy",
        *METRICS,
        "source_file",
    ]
    write_csv(SUMMARY_DIR / "paper_ed_id_adapted_all_runs.csv", long_rows, long_fields)

    fields = [
        "Dataset",
        "Strategy",
        "Seeds",
        "Suppression",
        "Cost",
        "Benefit/Cost",
        "Trigger rate",
        "Suppression gap vs C3",
        "Cost gap vs C3",
    ]
    aggregated = aggregate(long_rows)
    write_bundle(
        "paper_ed_id_adapted_table",
        paper_rows(aggregated, "stratified"),
        fields,
    )
    write_bundle(
        "paper_ed_id_adapted_temporal",
        paper_rows(aggregated, "temporal"),
        fields,
    )
    print("Generated ED-ID-adapted paper tables.")


if __name__ == "__main__":
    main()
