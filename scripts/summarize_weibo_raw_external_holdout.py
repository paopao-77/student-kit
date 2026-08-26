import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> float | str:
    if not math.isfinite(value):
        return ""
    return round(value, digits)


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def aggregate(rows: list[dict[str, Any]], group_key: str, metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    output = []
    for key, items in sorted(groups.items()):
        seeds = sorted({int(row["seed"]) for row in items})
        result: dict[str, Any] = {
            group_key: key,
            "n_seeds": len(seeds),
            "seeds": " ".join(map(str, seeds)),
        }
        for metric in metrics:
            values = finite_values(items, metric)
            avg = mean(values) if values else float("nan")
            sd = stdev(values) if len(values) > 1 else 0.0
            ci95 = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = fmt(avg)
            result[f"{metric}_std"] = fmt(sd)
            result[f"{metric}_ci95"] = fmt(ci95)
        output.append(result)
    return output


def load_c2(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*_metrics.json")):
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
    return rows


def load_c3(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*_metrics.json")):
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
    return rows


def first_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row where {key}={value}")


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def comparison_rows() -> list[dict[str, Any]]:
    v1_main = first_row(read_csv(SUMMARY / "v1_weibo_multiseed_summary.csv"), "model", "heterorumor_v1_hurdle")
    v1_holdout = first_row(read_csv(SUMMARY / "v1_weibo_external_holdout_summary.csv"), "model", "heterorumor_v1_hurdle")
    v2_main = first_row(
        read_csv(SUMMARY / "v2_c1_weibo_selected_multiseed_summary.csv"),
        "model",
        "heterorumor_v2_c1_vae_k4_weibo_selected",
    )
    v2_holdout = first_row(
        read_csv(SUMMARY / "v2_c1_weibo_external_holdout_summary.csv"),
        "model",
        "heterorumor_v2_c1_vae_k4_weibo_selected",
    )
    c2_main = first_row(read_csv(SUMMARY / "c2_breakout_weibo_raw_preferred_summary.csv"), "model", "heterorumor_c2")
    c2_holdout = first_row(read_csv(SUMMARY / "c2_breakout_weibo_external_holdout_summary.csv"), "model", "heterorumor_c2")
    c3_main = first_row(
        read_csv(SUMMARY / "c3_control_weibo_raw_preferred_summary.csv"),
        "strategy",
        "heterorumor_c3_event_pulse",
    )
    c3_holdout = first_row(
        read_csv(SUMMARY / "c3_control_weibo_external_holdout_summary.csv"),
        "strategy",
        "heterorumor_c3_event_pulse",
    )
    specs = [
        ("V1", "mape_mean", "lower", v1_main, v1_holdout),
        ("V2/C1", "mape_mean", "lower", v2_main, v2_holdout),
        ("C2", "auc_mean", "higher", c2_main, c2_holdout),
        ("C3", "mean_suppression_rate_mean", "higher", c3_main, c3_holdout),
    ]
    rows = []
    for module, name, better, main, holdout in specs:
        main_value = metric(main, name)
        holdout_value = metric(holdout, name)
        rows.append(
            {
                "module": module,
                "metric": name,
                "better": better,
                "stratified_preferred": fmt(main_value),
                "external_holdout": fmt(holdout_value),
                "absolute_delta_holdout_minus_stratified": fmt(holdout_value - main_value),
                "relative_delta": fmt((holdout_value - main_value) / abs(main_value), 6),
            }
        )
    return rows


def write_note(comparison: list[dict[str, Any]]) -> None:
    by_module = {row["module"]: row for row in comparison}
    lines = [
        "# Raw Weibo External Holdout Validation",
        "",
        "## Validation Design",
        "",
        "- Main raw-Weibo results use `split_strategy=stratified`.",
        "- This validation uses `split_strategy=temporal`; for Weibo this resolves to `weibo_rumor_binary_seed42_split.json`, a fixed-seed non-stratified holdout because reliable source timestamps are unavailable.",
        "- The validation reruns V1, V2/C1, C2, and C3 with seeds `7, 21, 42, 84, 2024` without replacing the preferred stratified summaries.",
        "",
        "## Main Comparison",
        "",
        "| module | metric | stratified preferred | external holdout | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['module']} | {row['metric']} | {row['stratified_preferred']} | "
            f"{row['external_holdout']} | {row['absolute_delta_holdout_minus_stratified']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- V1 holdout MAPE is {by_module['V1']['external_holdout']} versus stratified {by_module['V1']['stratified_preferred']}.",
            f"- V2/C1 holdout MAPE is {by_module['V2/C1']['external_holdout']} versus stratified {by_module['V2/C1']['stratified_preferred']}.",
            f"- C2 holdout AUC is {by_module['C2']['external_holdout']} versus preferred {by_module['C2']['stratified_preferred']}.",
            f"- C3 holdout suppression is {by_module['C3']['external_holdout']} versus preferred {by_module['C3']['stratified_preferred']}.",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_external_holdout_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    c2_rows = load_c2(ROOT / "results" / "c2_breakout_weibo_external_holdout")
    c3_rows = load_c3(ROOT / "results" / "c3_control_weibo_external_holdout")
    write_csv(SUMMARY / "c2_breakout_weibo_external_holdout_all_runs.csv", c2_rows)
    write_csv(
        SUMMARY / "c2_breakout_weibo_external_holdout_summary.csv",
        aggregate(c2_rows, "model", C2_METRICS),
    )
    write_csv(SUMMARY / "c3_control_weibo_external_holdout_all_runs.csv", c3_rows)
    write_csv(
        SUMMARY / "c3_control_weibo_external_holdout_summary.csv",
        aggregate(c3_rows, "strategy", C3_METRICS),
    )
    comparison = comparison_rows()
    write_csv(SUMMARY / "weibo_raw_external_holdout_comparison.csv", comparison)
    write_note(comparison)
    print(
        json.dumps(
            {
                "c2_runs": len({row["source_file"] for row in c2_rows}),
                "c3_runs": len({row["source_file"] for row in c3_rows}),
                "comparison": str(SUMMARY / "weibo_raw_external_holdout_comparison.csv"),
                "note": str(DRAFTS / "weibo_raw_external_holdout_validation.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
