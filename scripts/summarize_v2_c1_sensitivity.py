import csv
import json
from pathlib import Path


def load_metric(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    model = payload["model_type"]
    return {
        "model": model,
        "latent_dim": payload.get("latent_dim"),
        "kl_weight": payload.get("kl_weight"),
        "counterfactual_weight": payload.get("counterfactual_weight", 0.0),
        "best_val_mape": payload["best_val_mape"],
        "test_mape": payload["models"][model]["test"]["mape"],
        "test_mae": payload["models"][model]["test"]["mae"],
        "test_r2": payload["models"][model]["test"]["r2"],
        "active_latent_factors": payload["models"][model]["test"].get(
            "active_latent_factors"
        ),
        "latent_std_mean": payload["models"][model]["test"].get("latent_std_mean"),
        "source_file": str(path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    k_paths = list(Path("results/heterorumor_v2_c1_sensitivity").glob("*_metrics.json"))
    k_paths.append(
        Path(
            "results/heterorumor_v2_c1/"
            "pheme_cascade_size_stratified_heterorumor_v2_c1_vae_k16_"
            "multilingual_minilm_obs180_seed42_metrics.json"
        )
    )
    k_rows = sorted((load_metric(path) for path in k_paths), key=lambda row: row["latent_dim"])

    kl_rows = [row for row in k_rows if row["latent_dim"] == 4]
    kl_rows.extend(
        load_metric(path)
        for path in Path("results/heterorumor_v2_c1_kl_sensitivity").glob("*_metrics.json")
    )
    kl_rows = sorted(kl_rows, key=lambda row: row["kl_weight"])

    robustness_path = Path("results/summary/v2_c1_counterfactual_robustness.csv")
    with robustness_path.open("r", encoding="utf-8-sig", newline="") as f:
        robustness_rows = list(csv.DictReader(f))
    grouped = {}
    for row in robustness_rows:
        key = (row["model"], float(row["counterfactual_weight"]), row["split"])
        grouped.setdefault(key, {})[float(row["text_corruption_rate"])] = float(row["mape"])
    cf_rows = []
    for (model, weight, split), values in sorted(grouped.items()):
        cf_rows.append(
            {
                "model": model,
                "counterfactual_weight": weight,
                "split": split,
                "mape_clean": values[0.0],
                "mape_noise_0.1": values[0.1],
                "mape_noise_0.2": values[0.2],
                "mape_noise_0.3": values[0.3],
                "mean_mape_all_conditions": sum(values.values()) / len(values),
                "degradation_at_0.3": values[0.3] - values[0.0],
            }
        )

    output_dir = Path("results/summary")
    write_csv(output_dir / "v2_c1_k_sensitivity.csv", k_rows)
    write_csv(output_dir / "v2_c1_kl_sensitivity.csv", kl_rows)
    write_csv(output_dir / "v2_c1_counterfactual_summary.csv", cf_rows)
    print(
        json.dumps(
            {
                "k_rows": len(k_rows),
                "kl_rows": len(kl_rows),
                "counterfactual_rows": len(cf_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
