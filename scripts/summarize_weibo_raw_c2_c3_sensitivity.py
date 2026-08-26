import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SUMMARY = Path("results/summary")
DRAFTS = Path("results/drafts")

CONFIGS = [
    {
        "order_window_size": 50,
        "c2_dir": Path("results/c2_breakout_weibo_raw_ow50"),
        "c3_dir": Path("results/c3_control_weibo_raw_ow50"),
    },
    {
        "order_window_size": 100,
        "c2_dir": Path("results/c2_breakout_weibo_raw"),
        "c3_dir": Path("results/c3_control_weibo_raw"),
    },
    {
        "order_window_size": 200,
        "c2_dir": Path("results/c2_breakout_weibo_raw_ow200"),
        "c3_dir": Path("results/c3_control_weibo_raw_ow200"),
    },
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stat(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def fmt(value: float) -> float | str:
    if not math.isfinite(value):
        return ""
    return round(value, 6)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_c2(c2_dir: Path) -> dict[str, float | int | str]:
    rows = []
    for path in sorted(c2_dir.glob("*_metrics.json")):
        payload = read_json(path)
        test = payload["models"]["heterorumor_c2"]["test"]
        rows.append(
            {
                "seed": int(payload["seed"]),
                "auc": float(test["auc"]),
                "f1": float(test["f1"]),
                "macro_f1": float(test["macro_f1"]),
                "mean_lead_time": float(test["mean_lead_time_minutes"]),
                "warning_rate": float(test["warning_rate"]),
            }
        )
    if len(rows) < 5:
        raise FileNotFoundError(f"Expected 5 C2 runs under {c2_dir}, found {len(rows)}")
    output: dict[str, float | int | str] = {
        "c2_n_seeds": len(rows),
        "c2_seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
    }
    for metric in ["auc", "f1", "macro_f1", "mean_lead_time", "warning_rate"]:
        avg, sd = stat([float(row[metric]) for row in rows])
        output[f"c2_{metric}_mean"] = fmt(avg)
        output[f"c2_{metric}_std"] = fmt(sd)
    return output


def collect_c3(c3_dir: Path) -> dict[str, float | int | str]:
    per_strategy: dict[str, list[dict[str, float | int]]] = {}
    for path in sorted(c3_dir.glob("*_metrics.json")):
        payload = read_json(path)
        for strategy, metrics in payload["strategies"].items():
            per_strategy.setdefault(strategy, []).append(
                {
                    "seed": int(payload["seed"]),
                    "trigger_rate": float(metrics["trigger_rate"]),
                    "suppression": float(metrics["mean_suppression_rate"]),
                    "cost": float(metrics["mean_cost"]),
                    "bcr": float(metrics["mean_benefit_cost_ratio"]),
                }
            )
    if len(per_strategy.get("heterorumor_c3_event_pulse", [])) < 5:
        found = len(per_strategy.get("heterorumor_c3_event_pulse", []))
        raise FileNotFoundError(f"Expected 5 C3 runs under {c3_dir}, found {found}")

    output: dict[str, float | int | str] = {}
    for strategy in [
        "heterorumor_c3_event_pulse",
        "random_same_budget",
        "fixed_same_budget",
        "ed_id_adapted",
    ]:
        prefix = {
            "heterorumor_c3_event_pulse": "c3_event",
            "random_same_budget": "c3_random",
            "fixed_same_budget": "c3_fixed",
            "ed_id_adapted": "c3_edid",
        }[strategy]
        rows = per_strategy[strategy]
        for metric in ["trigger_rate", "suppression", "cost", "bcr"]:
            avg, sd = stat([float(row[metric]) for row in rows])
            output[f"{prefix}_{metric}_mean"] = fmt(avg)
            output[f"{prefix}_{metric}_std"] = fmt(sd)
    return output


def write_note(rows: list[dict[str, Any]]) -> None:
    best_c2 = max(rows, key=lambda row: float(row["c2_auc_mean"]))
    best_c3 = max(rows, key=lambda row: float(row["c3_event_suppression_mean"]))
    lines = [
        "# Raw Weibo C2/C3 Order-Window Sensitivity",
        "",
        "Five random seeds were run for each event-order window size: 50, 100, and 200.",
        "",
        "| order_window_size | C2 AUC | C2 F1 | C3 event suppression | C3 event cost | random same-budget suppression |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {order_window_size} | {c2_auc_mean} | {c2_f1_mean} | "
            "{c3_event_suppression_mean} | {c3_event_cost_mean} | "
            "{c3_random_suppression_mean} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"- Highest C2 AUC in this grid: order window {best_c2['order_window_size']}.",
            f"- Highest C3 event-pulse suppression in this grid: order window {best_c3['order_window_size']}.",
            "- Interpretation remains proxy-only because raw Weibo lacks retweet-parent edges.",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_c2_c3_order_window_sensitivity.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    rows = []
    for config in CONFIGS:
        row: dict[str, Any] = {"order_window_size": config["order_window_size"]}
        row.update(collect_c2(config["c2_dir"]))
        row.update(collect_c3(config["c3_dir"]))
        rows.append(row)
    write_csv(SUMMARY / "weibo_raw_c2_c3_order_window_sensitivity.csv", rows)
    write_note(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary": str(SUMMARY / "weibo_raw_c2_c3_order_window_sensitivity.csv"),
                "note": str(DRAFTS / "weibo_raw_c2_c3_order_window_sensitivity.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
