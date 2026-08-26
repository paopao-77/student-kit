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
C2_DIR = RESULTS / "c2_breakout_weibo_raw"
C3_DIR = RESULTS / "c3_control_weibo_raw"
C2_STATS = Path("data/processed/weibo/c2_foundation_stats.json")

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
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def fmt(value: float, digits: int = 6) -> float | str:
    if not math.isfinite(value):
        return ""
    return round(value, digits)


def aggregate(
    rows: list[dict[str, Any]], group_key: str, metrics: list[str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)

    output = []
    for key, items in sorted(groups.items()):
        seeds = sorted({int(row["seed"]) for row in items})
        out: dict[str, Any] = {group_key: key, "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds))}
        for metric in metrics:
            values = finite_values(items, metric)
            avg = mean(values) if values else float("nan")
            sd = stdev(values) if len(values) > 1 else 0.0
            ci95 = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            out[f"{metric}_mean"] = fmt(avg)
            out[f"{metric}_std"] = fmt(sd)
            out[f"{metric}_ci95"] = fmt(ci95)
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
    return rows


def write_note(c2_summary: list[dict[str, Any]], c3_summary: list[dict[str, Any]]) -> None:
    stats = read_json(C2_STATS) if C2_STATS.exists() else {}
    c2_main = next(row for row in c2_summary if row["model"] == "heterorumor_c2")
    c3_main = next(row for row in c3_summary if row["strategy"] == "heterorumor_c3_event_pulse")
    c3_random = next(row for row in c3_summary if row["strategy"] == "random_same_budget")
    c3_fixed = next(row for row in c3_summary if row["strategy"] == "fixed_same_budget")
    lines = [
        "# Raw Weibo C2/C3 Experiment Note",
        "",
        "## Data Artifact",
        "",
        "- Source: `data/processed/weibo/`, regenerated from `数据集/微博/*`.",
        f"- Samples: {stats.get('num_samples', '')}; samples with edges: {stats.get('num_samples_with_edges', '')}.",
        f"- Breakout positives: {stats.get('num_breakout_samples', '')}; breakout rate: {stats.get('breakout_rate', '')}.",
        f"- Time mode: `{stats.get('time_mode', '')}`; order window size: {stats.get('order_window_size', '')}.",
        "",
        "## Five-Seed Results",
        "",
        f"- C2 `heterorumor_c2`: AUC {c2_main['auc_mean']} +/- {c2_main['auc_std']}, F1 {c2_main['f1_mean']} +/- {c2_main['f1_std']}.",
        f"- C2 mean lead field: {c2_main['mean_lead_time_minutes_mean']} +/- {c2_main['mean_lead_time_minutes_std']} event-order units.",
        f"- C3 `heterorumor_c3_event_pulse`: suppression {c3_main['mean_suppression_rate_mean']} +/- {c3_main['mean_suppression_rate_std']}, cost {c3_main['mean_cost_mean']} +/- {c3_main['mean_cost_std']}.",
        f"- C3 same-budget comparisons: random suppression {c3_random['mean_suppression_rate_mean']}, fixed suppression {c3_fixed['mean_suppression_rate_mean']}.",
        "",
        "## Interpretation Guardrail",
        "",
        "Raw Weibo provides source-to-repost star edges but not retweet-parent edges. Therefore the rebuilt C2/C3 run is a raw-data sanity/proxy experiment based on event-order dynamics and branch heuristics, not a full social-community cascade reconstruction.",
        "",
    ]
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_c2_c3_experiment_note.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c2_rows = load_c2()
    c3_rows = load_c3()
    if not c2_rows:
        raise FileNotFoundError(f"No C2 metrics found under {C2_DIR}")
    if not c3_rows:
        raise FileNotFoundError(f"No C3 metrics found under {C3_DIR}")

    c2_summary = aggregate(c2_rows, "model", C2_METRICS)
    c3_summary = aggregate(c3_rows, "strategy", C3_METRICS)
    write_csv(SUMMARY / "c2_breakout_weibo_raw_all_runs.csv", c2_rows)
    write_csv(SUMMARY / "c2_breakout_weibo_raw_summary.csv", c2_summary)
    write_csv(SUMMARY / "c3_control_weibo_raw_all_runs.csv", c3_rows)
    write_csv(SUMMARY / "c3_control_weibo_raw_summary.csv", c3_summary)
    write_note(c2_summary, c3_summary)
    print(
        json.dumps(
            {
                "c2_runs": len({row["source_file"] for row in c2_rows}),
                "c3_runs": len({row["source_file"] for row in c3_rows}),
                "c2_summary": str(SUMMARY / "c2_breakout_weibo_raw_summary.csv"),
                "c3_summary": str(SUMMARY / "c3_control_weibo_raw_summary.csv"),
                "note": str(DRAFTS / "weibo_raw_c2_c3_experiment_note.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
