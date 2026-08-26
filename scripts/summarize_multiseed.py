import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = ["mae", "rmse", "mape", "smape", "r2", "median_ae"]
OPTIONAL_METRICS = [
    "best_val_mape",
    "active_latent_factors",
    "active_content_factors",
    "content_dynamics_cross_covariance_mse",
    "text_noise_0.3_mape",
    "matched_text_swap_mape",
    "matched_text_swap_delta_mape",
    "mean_matched_target_gap",
]


def load_rows(results_dir: Path, model_prefix: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        model_type = str(payload.get("model_type", ""))
        if not model_type.startswith(model_prefix):
            continue
        test = payload.get("models", {}).get(model_type, {}).get("test")
        if not test:
            continue
        row = {
            "dataset": payload.get("dataset"),
            "split_strategy": payload.get("split_strategy"),
            "observation_window_minutes": payload.get("observation_window_minutes"),
            "model": model_type,
            "seed": int(payload.get("seed")),
            "split_seed": int(payload.get("split_seed", 42)),
            "num_samples": int(test.get("num_samples")),
            "source_file": str(path),
        }
        for metric in METRICS:
            row[metric] = float(test[metric])
        if payload.get("best_val_mape") is not None:
            row["best_val_mape"] = float(payload["best_val_mape"])
        for metric in [
            "active_latent_factors",
            "active_content_factors",
            "content_dynamics_cross_covariance_mse",
        ]:
            if test.get(metric) is not None:
                row[metric] = float(test[metric])
        robustness = payload.get("robustness", {})
        text_noise = robustness.get("text_noise_0.3", {})
        matched_swap = robustness.get("matched_text_swap", {})
        if text_noise.get("mape") is not None:
            row["text_noise_0.3_mape"] = float(text_noise["mape"])
        if matched_swap.get("mape") is not None:
            row["matched_text_swap_mape"] = float(matched_swap["mape"])
            row["matched_text_swap_delta_mape"] = (
                row["matched_text_swap_mape"] - row["mape"]
            )
        if matched_swap.get("mean_matched_target_gap") is not None:
            row["mean_matched_target_gap"] = float(
                matched_swap["mean_matched_target_gap"]
            )
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["split_strategy"],
            row["observation_window_minutes"],
            row["model"],
            row["split_seed"],
        )
        groups.setdefault(key, []).append(row)

    output = []
    for key, group in sorted(groups.items()):
        result = {
            "dataset": key[0],
            "split_strategy": key[1],
            "observation_window_minutes": key[2],
            "model": key[3],
            "split_seed": key[4],
            "num_seeds": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
        }
        for metric in METRICS + OPTIONAL_METRICS:
            if not all(metric in row for row in group):
                continue
            values = [row[metric] for row in group]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_ci95"] = ci95
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No matching multi-seed metrics were found")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_text_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_name = "heterorumor_v1_hurdle_multilingual_minilm"
    no_text_name = full_name + "_wo_text"
    groups: dict[tuple[Any, ...], dict[tuple[str, int], dict[str, Any]]] = {}
    for row in rows:
        group_key = (
            row["dataset"],
            row["split_strategy"],
            row["observation_window_minutes"],
            row["split_seed"],
        )
        groups.setdefault(group_key, {})[(row["model"], row["seed"])] = row
    output = []
    for group_key, by_model_seed in sorted(groups.items()):
        seeds = sorted(
            seed
            for model, seed in by_model_seed
            if model == full_name and (no_text_name, seed) in by_model_seed
        )
        if not seeds:
            continue
        for metric in METRICS:
            differences = [
                by_model_seed[(no_text_name, seed)][metric]
                - by_model_seed[(full_name, seed)][metric]
                for seed in seeds
            ]
            mean = statistics.fmean(differences)
            std = statistics.stdev(differences) if len(differences) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
            p_value = ""
            try:
                from scipy import stats

                p_value = float(stats.ttest_1samp(differences, popmean=0.0).pvalue)
            except ImportError:
                pass
            output.append(
                {
                    "dataset": group_key[0],
                    "split_strategy": group_key[1],
                    "observation_window_minutes": group_key[2],
                    "split_seed": group_key[3],
                    "metric": metric,
                    "num_pairs": len(seeds),
                    "seeds": ",".join(map(str, seeds)),
                    "difference_definition": "no_text_minus_full",
                    "mean_difference": mean,
                    "std_difference": std,
                    "ci95_half_width": ci95,
                    "paired_ttest_p": p_value,
                    "full_better_pairs": sum(value > 0 for value in differences),
                    "differences": ",".join(f"{value:.8f}" for value in differences),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/heterorumor_v1_plm_multiseed")
    parser.add_argument("--model-prefix", default="heterorumor_v1_hurdle_multilingual_minilm")
    parser.add_argument("--output", default="results/summary/v1_plm_multiseed_summary.csv")
    parser.add_argument(
        "--paired-output", default="results/summary/v1_plm_multiseed_paired_text.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.results_dir), args.model_prefix)
    summary = aggregate(rows)
    write_csv(Path(args.output), summary)
    paired = paired_text_comparison(rows)
    if paired:
        write_csv(Path(args.paired_output), paired)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "groups": len(summary),
                "output": args.output,
                "paired_output": args.paired_output if paired else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
