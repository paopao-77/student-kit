import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = ["mae", "rmse", "mape", "smape", "r2", "median_ae"]
HASH_MODEL = "heterorumor_v1_hurdle"
MINILM_MODEL = "heterorumor_v1_hurdle_multilingual_minilm"
NO_TEXT_MODEL = "heterorumor_v1_hurdle_multilingual_minilm_wo_text"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = read_json(path)
        model = str(payload.get("model_type", ""))
        test = payload.get("models", {}).get(model, {}).get("test")
        if not test:
            continue
        row = {
            "dataset": payload.get("dataset"),
            "split_strategy": payload.get("split_strategy"),
            "observation_window_minutes": payload.get("observation_window_minutes"),
            "model": model,
            "seed": int(payload.get("seed")),
            "split_seed": int(payload.get("split_seed", 42)),
            "source_file": str(path),
        }
        for metric in METRICS:
            row[metric] = float(test[metric])
        rows.append(row)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def paired_values(
    rows: list[dict[str, Any]],
    dataset: str,
    base_model: str,
    candidate_model: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    by_key = {
        (row["dataset"], row["model"], row["seed"]): row
        for row in rows
        if row["dataset"] == dataset
    }
    seeds = sorted(
        seed
        for row_dataset, model, seed in by_key
        if row_dataset == dataset
        and model == base_model
        and (dataset, candidate_model, seed) in by_key
    )
    deltas = [
        by_key[(dataset, candidate_model, seed)][metric]
        - by_key[(dataset, base_model, seed)][metric]
        for seed in seeds
    ]
    return seeds, deltas


def paired_ttest_p(deltas: list[float]) -> float | str:
    if len(deltas) < 2:
        return ""
    try:
        from scipy import stats

        return float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
    except ImportError:
        return ""


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({row["dataset"] for row in rows})
    output = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_model = {
            model: [row for row in dataset_rows if row["model"] == model]
            for model in [HASH_MODEL, MINILM_MODEL, NO_TEXT_MODEL]
        }
        if any(not by_model[model] for model in by_model):
            continue

        hash_seeds = sorted(row["seed"] for row in by_model[HASH_MODEL])
        minilm_seeds = sorted(row["seed"] for row in by_model[MINILM_MODEL])
        no_text_seeds = sorted(row["seed"] for row in by_model[NO_TEXT_MODEL])
        row = {
            "dataset": dataset,
            "split_strategy": by_model[MINILM_MODEL][0]["split_strategy"],
            "observation_window_minutes": by_model[MINILM_MODEL][0][
                "observation_window_minutes"
            ],
            "split_seed": by_model[MINILM_MODEL][0]["split_seed"],
            "hash_seeds": ",".join(map(str, hash_seeds)),
            "minilm_seeds": ",".join(map(str, minilm_seeds)),
            "no_text_seeds": ",".join(map(str, no_text_seeds)),
            "num_hash": len(hash_seeds),
            "num_minilm": len(minilm_seeds),
            "num_no_text": len(no_text_seeds),
        }

        for model_key, model_name in [
            ("hash", HASH_MODEL),
            ("minilm", MINILM_MODEL),
            ("no_text", NO_TEXT_MODEL),
        ]:
            for metric in METRICS:
                mean, std = mean_std([item[metric] for item in by_model[model_name]])
                row[f"{model_key}_{metric}_mean"] = mean
                row[f"{model_key}_{metric}_std"] = std

        hash_mean = row["hash_mape_mean"]
        minilm_mean = row["minilm_mape_mean"]
        no_text_mean = row["no_text_mape_mean"]
        row["minilm_minus_hash_mape"] = minilm_mean - hash_mean
        row["minilm_relative_mape_reduction_vs_hash_pct"] = (
            100.0 * (hash_mean - minilm_mean) / hash_mean if hash_mean else ""
        )
        row["no_text_minus_minilm_mape"] = no_text_mean - minilm_mean
        row["text_relative_mape_gain_pct"] = (
            100.0 * (no_text_mean - minilm_mean) / no_text_mean if no_text_mean else ""
        )

        seeds, deltas = paired_values(rows, dataset, HASH_MODEL, MINILM_MODEL, "mape")
        row["hash_vs_minilm_paired_seeds"] = ",".join(map(str, seeds))
        row["minilm_better_than_hash_pairs"] = sum(delta < 0.0 for delta in deltas)
        row["minilm_minus_hash_paired_mape_mean"] = (
            statistics.fmean(deltas) if deltas else ""
        )
        row["minilm_vs_hash_paired_ttest_p"] = paired_ttest_p(deltas)

        seeds, deltas = paired_values(rows, dataset, MINILM_MODEL, NO_TEXT_MODEL, "mape")
        row["minilm_vs_no_text_paired_seeds"] = ",".join(map(str, seeds))
        row["minilm_better_than_no_text_pairs"] = sum(delta > 0.0 for delta in deltas)
        row["no_text_minus_minilm_paired_mape_mean"] = (
            statistics.fmean(deltas) if deltas else ""
        )
        row["text_ablation_paired_ttest_p"] = paired_ttest_p(deltas)
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No complete text fairness rows were found")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "hash_mape_mean",
        "hash_mape_std",
        "minilm_mape_mean",
        "minilm_mape_std",
        "no_text_mape_mean",
        "no_text_mape_std",
        "minilm_minus_hash_mape",
        "no_text_minus_minilm_mape",
        "minilm_better_than_hash_pairs",
        "minilm_better_than_no_text_pairs",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hash-dir",
        default="results/heterorumor_v1_rumdetect2017_hash_multiseed",
    )
    parser.add_argument(
        "--plm-dir",
        default="results/heterorumor_v1_rumdetect2017_plm_multiseed",
    )
    parser.add_argument(
        "--output",
        default="results/summary/v1_rumdetect2017_text_fairness.csv",
    )
    parser.add_argument(
        "--markdown-output",
        default="results/summary/v1_rumdetect2017_text_fairness.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.hash_dir).glob("*_metrics.json")) + sorted(
        Path(args.plm_dir).glob("*_metrics.json")
    )
    rows = summarize(load_rows(paths))
    write_csv(Path(args.output), rows)
    write_markdown(Path(args.markdown_output), rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output": args.output,
                "markdown_output": args.markdown_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
