import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.train_heterorumor_v1 import build_model, evaluate
from v1_dataset import V1InputDataset


def experiment_args(payload: dict, checkpoint: dict, batch_size: int) -> Namespace:
    config = checkpoint["model_config"]
    return Namespace(
        dataset=payload["dataset"],
        observation=int(payload["observation_window_minutes"]),
        split_strategy=payload["split_strategy"],
        split_seed=int(payload.get("split_seed", 42)),
        seed=int(payload["seed"]),
        task=payload.get("label_task", "rumor_binary"),
        input_root="data/processed/v1_inputs",
        text_feature_name=config["text_feature_name"],
        text_feature_path=config["text_feature_path"],
        model_version=config["model_version"],
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        graph_layers=int(config["graph_layers"]),
        dropout=float(config["dropout"]),
        growth_threshold=float(config["growth_threshold"]),
        run_tag=config.get("run_tag", ""),
        batch_size=batch_size,
        active_factor_std=0.05,
        disable_text=False,
        disable_topology=False,
        disable_temporal=False,
        disable_user=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/heterorumor_v2_c1_cf_sensitivity")
    parser.add_argument("--output", default="results/summary/v2_c1_counterfactual_robustness.csv")
    parser.add_argument("--noise-scales", default="0,0.1,0.2,0.3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_scales = [float(value.strip()) for value in args.noise_scales.split(",")]
    output_rows = []
    results_dir = Path(args.results_dir)
    for checkpoint_path in sorted(results_dir.glob("*_checkpoint.pt")):
        metrics_path = checkpoint_path.with_name(
            checkpoint_path.name.replace("_checkpoint.pt", "_metrics.json")
        )
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        run_args = experiment_args(payload, checkpoint, args.batch_size)
        datasets = {
            split: V1InputDataset(
                dataset=run_args.dataset,
                observation=run_args.observation,
                split=split,
                split_strategy=run_args.split_strategy,
                seed=run_args.split_seed,
                task=run_args.task,
                input_root=run_args.input_root,
                text_feature_path=run_args.text_feature_path,
            )
            for split in ("val", "test")
        }
        model = build_model(datasets["val"], run_args).to(torch.device(args.device))
        model.load_state_dict(checkpoint["model_state_dict"])
        for split in ("val", "test"):
            indices = list(range(len(datasets[split])))
            for noise_scale in noise_scales:
                result = evaluate(
                    model,
                    datasets[split],
                    indices,
                    run_args,
                    torch.device(args.device),
                    split=split,
                    growth_threshold=run_args.growth_threshold,
                    text_noise_scale=noise_scale,
                )
                metrics = result["metrics"]
                output_rows.append(
                    {
                        "model": payload["model_type"],
                        "counterfactual_weight": payload.get("counterfactual_weight", 0.0),
                        "split": split,
                        "text_corruption_rate": noise_scale,
                        "mape": metrics["mape"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "active_latent_factors": metrics.get("active_latent_factors", ""),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    validation = [row for row in output_rows if row["split"] == "val"]
    scores = {}
    for row in validation:
        scores.setdefault(row["model"], []).append(float(row["mape"]))
    best_model = min(scores, key=lambda model: sum(scores[model]) / len(scores[model]))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_rows": len(output_rows),
                "best_validation_robust_model": best_model,
                "mean_validation_mape": sum(scores[best_model]) / len(scores[best_model]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
