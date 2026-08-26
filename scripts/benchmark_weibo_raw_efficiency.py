import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"
BENCH_ROOT = ROOT / "results" / "efficiency_benchmark"

V1_DIR = ROOT / "results/heterorumor_v1_weibo_multiseed"
V2_DIR = ROOT / "results/heterorumor_v2_c1_weibo_selected_multiseed"
C2_BENCH_DIR = BENCH_ROOT / "c2_breakout_weibo_raw_ow50_seed42"
C3_BENCH_DIR = BENCH_ROOT / "c3_control_weibo_raw_ow50_seed42"


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


def collect_training_seconds(directory: Path, pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob(pattern)):
        payload = read_json(path)
        rows.append(
            {
                "seed": payload.get("seed"),
                "training_seconds": float(payload.get("training_seconds", 0.0)),
                "epochs_ran": payload.get("epochs_ran", ""),
                "parameter_count": payload.get("parameter_count", ""),
                "source_file": str(path),
            }
        )
    return rows


def summarize_times(module: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["training_seconds"]) for row in rows]
    if not values:
        raise ValueError(f"No timing rows for {module}")
    sorted_values = sorted(values)
    med = median(sorted_values)
    outlier_threshold = med * 5 if med > 0 else float("inf")
    outliers = [value for value in sorted_values if value > outlier_threshold]
    return {
        "module": module,
        "measurement_type": "existing_training_seconds",
        "n_runs": len(values),
        "seconds_mean": fmt(mean(values)),
        "seconds_median": fmt(med),
        "seconds_std": fmt(stdev(values) if len(values) > 1 else 0.0),
        "seconds_min": fmt(min(values)),
        "seconds_max": fmt(max(values)),
        "outlier_rule": "value > 5x median",
        "num_outliers": len(outliers),
        "notes": "Uses training_seconds recorded in existing metrics JSON files.",
    }


def run_timed(command: list[str], cwd: Path) -> tuple[float, int]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return time.perf_counter() - started, int(completed.returncode)


def benchmark_c2_c3(python_exe: str) -> list[dict[str, Any]]:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    c2_cmd = [
        python_exe,
        "scripts/train_c2_breakout.py",
        "--dataset",
        "weibo",
        "--data-root",
        "data/processed",
        "--split-strategy",
        "stratified",
        "--split-seed",
        "42",
        "--seed",
        "42",
        "--output-dir",
        str(C2_BENCH_DIR),
    ]
    c2_seconds, c2_rc = run_timed(c2_cmd, ROOT)
    if c2_rc != 0:
        raise RuntimeError(f"C2 benchmark failed with return code {c2_rc}")

    c3_cmd = [
        python_exe,
        "scripts/simulate_c3_control.py",
        "--dataset",
        "weibo",
        "--data-root",
        "data/processed",
        "--c2-dir",
        str(C2_BENCH_DIR),
        "--c2-model",
        "heterorumor_c2",
        "--split-strategy",
        "stratified",
        "--seed",
        "42",
        "--output-dir",
        str(C3_BENCH_DIR),
    ]
    c3_seconds, c3_rc = run_timed(c3_cmd, ROOT)
    if c3_rc != 0:
        raise RuntimeError(f"C3 benchmark failed with return code {c3_rc}")

    rows = [
        {
            "module": "C2 preferred benchmark",
            "measurement_type": "subprocess_wall_seconds_seed42",
            "n_runs": 1,
            "seconds_mean": fmt(c2_seconds),
            "seconds_median": fmt(c2_seconds),
            "seconds_std": 0.0,
            "seconds_min": fmt(c2_seconds),
            "seconds_max": fmt(c2_seconds),
            "outlier_rule": "",
            "num_outliers": 0,
            "notes": f"Timed seed42 subprocess; output_dir={C2_BENCH_DIR}.",
        },
        {
            "module": "C3 preferred benchmark",
            "measurement_type": "subprocess_wall_seconds_seed42",
            "n_runs": 1,
            "seconds_mean": fmt(c3_seconds),
            "seconds_median": fmt(c3_seconds),
            "seconds_std": 0.0,
            "seconds_min": fmt(c3_seconds),
            "seconds_max": fmt(c3_seconds),
            "outlier_rule": "",
            "num_outliers": 0,
            "notes": f"Timed seed42 subprocess; output_dir={C3_BENCH_DIR}.",
        },
    ]
    (BENCH_ROOT / "weibo_raw_c2_c3_benchmark_commands.json").write_text(
        json.dumps(
            {
                "c2_command": c2_cmd,
                "c3_command": c3_cmd,
                "c2_seconds": c2_seconds,
                "c3_seconds": c3_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def write_note(summary_rows: list[dict[str, Any]], v1_rows: list[dict[str, Any]], v2_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Raw Weibo Efficiency Benchmark",
        "",
        "## Summary",
        "",
        "| module | measurement | n | mean seconds | median seconds | notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['module']} | {row['measurement_type']} | {row['n_runs']} | "
            f"{row['seconds_mean']} | {row['seconds_median']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- V1/V2-C1 timings are read from existing metrics JSON files, so they reflect the original run environment and any interruptions.",
            "- V1 has a recorded long-running outlier; use the median as the robust headline number.",
            "- C2/C3 timings are fresh seed42 subprocess wall-clock measurements under the current preferred `order_window_size=50` artifact.",
            "",
            "## V1 Raw Timings",
            "",
            "| seed | training_seconds | epochs_ran |",
            "|---:|---:|---:|",
        ]
    )
    for row in sorted(v1_rows, key=lambda item: int(item["seed"])):
        lines.append(f"| {row['seed']} | {fmt(float(row['training_seconds']))} | {row['epochs_ran']} |")
    lines.extend(["", "## V2/C1 Raw Timings", "", "| seed | training_seconds | epochs_ran |", "|---:|---:|---:|"])
    for row in sorted(v2_rows, key=lambda item: int(item["seed"])):
        lines.append(f"| {row['seed']} | {fmt(float(row['training_seconds']))} | {row['epochs_ran']} |")
    lines.append("")
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_efficiency_benchmark.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-c2-c3-benchmark", action="store_true")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    v1_rows = collect_training_seconds(V1_DIR, "*_metrics.json")
    v2_rows = collect_training_seconds(V2_DIR, "*_metrics.json")
    summary_rows = [
        summarize_times("V1 raw Weibo", v1_rows),
        summarize_times("V2/C1 raw Weibo", v2_rows),
    ]
    if not args.skip_c2_c3_benchmark:
        summary_rows.extend(benchmark_c2_c3(args.python_exe))

    write_csv(SUMMARY / "weibo_raw_efficiency_summary.csv", summary_rows)
    write_csv(SUMMARY / "weibo_raw_v1_efficiency_runs.csv", v1_rows)
    write_csv(SUMMARY / "weibo_raw_v2_c1_efficiency_runs.csv", v2_rows)
    write_note(summary_rows, v1_rows, v2_rows)
    print(
        json.dumps(
            {
                "summary": str(SUMMARY / "weibo_raw_efficiency_summary.csv"),
                "v1_runs": str(SUMMARY / "weibo_raw_v1_efficiency_runs.csv"),
                "v2_runs": str(SUMMARY / "weibo_raw_v2_c1_efficiency_runs.csv"),
                "note": str(DRAFTS / "weibo_raw_efficiency_benchmark.md"),
                "c2_c3_benchmarked": not args.skip_c2_c3_benchmark,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
