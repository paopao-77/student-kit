import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RESULTS = Path("results")
SUMMARY = RESULTS / "summary"
SEEDS = [7, 21, 42, 84, 2024]
DATASETS = ["pheme", "twitter15", "twitter16"]
FAMILY_ORDER = [
    "HeteroRumorDyn-V1",
    "MIDPMS-adapted",
    "DSHCL-adapted",
    "Inf-VAE-adapted",
    "SEIZ/CD-SEIZ",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def emit(stem: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(SUMMARY / f"{stem}.csv", rows, fields)
    write_markdown(SUMMARY / f"{stem}.md", rows, fields)
    write_latex(SUMMARY / f"{stem}.tex", rows, fields)


def fmt(value: float, digits: int = 4) -> str:
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def fmt_pm(values: list[float]) -> str:
    mean_value, std_value = mean_std(values)
    return fmt(mean_value) if std_value == 0 else f"{mean_value:.4f} +/- {std_value:.4f}"


def v1_path(dataset: str, split_strategy: str, seed: int) -> Path:
    if dataset == "pheme":
        directory = (
            RESULTS / "heterorumor_v1_plm_multiseed"
            if split_strategy == "stratified"
            else RESULTS / "heterorumor_v1_plm"
        )
        name = (
            f"pheme_cascade_size_{split_strategy}_"
            f"heterorumor_v1_hurdle_multilingual_minilm_obs180_seed{seed}_metrics.json"
        )
    else:
        directory = RESULTS / "v1_fair180"
        name = (
            f"{dataset}_cascade_size_{split_strategy}_"
            f"heterorumor_v1_hurdle_obs180_seed{seed}_metrics.json"
        )
    return directory / name


def baseline_path(family: str, dataset: str, split_strategy: str, seed: int) -> Path:
    if family == "MIDPMS-adapted":
        directory = RESULTS / "paper_baselines" / "fair180" / "midpms"
        stem = "midpms_adapted"
    elif family == "DSHCL-adapted":
        directory = RESULTS / "paper_baselines" / "fair180" / "dshcl"
        stem = "dshcl_adapted"
    elif family == "Inf-VAE-adapted":
        directory = RESULTS / "paper_baselines" / "fair180" / "inf_vae"
        stem = "inf_vae_adapted"
    else:
        directory = RESULTS / "paper_baselines" / "fair180" / "dynamics"
        stem = "dynamics"
    return directory / (
        f"{dataset}_cascade_size_{split_strategy}_{stem}_seed{seed}_metrics.json"
    )


def select_on_validation(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = []
    for model, split_payload in payload["models"].items():
        if int(round(float(split_payload["test"]["observation_window_minutes"]))) != 180:
            continue
        candidates.append((float(split_payload["val"]["mape"]), model, split_payload))
    if not candidates:
        raise RuntimeError("No 180-minute model found")
    _, model, split_payload = min(candidates, key=lambda item: (item[0], item[1]))
    return model, split_payload


def load_run(
    family: str,
    dataset: str,
    split_strategy: str,
    seed: int,
) -> dict[str, Any]:
    path = (
        v1_path(dataset, split_strategy, seed)
        if family == "HeteroRumorDyn-V1"
        else baseline_path(family, dataset, split_strategy, seed)
    )
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if family == "HeteroRumorDyn-V1":
        model, split_payload = next(iter(payload["models"].items()))
    else:
        model, split_payload = select_on_validation(payload)
    metrics = split_payload["test"]
    return {
        "dataset": dataset,
        "split_strategy": split_strategy,
        "family": family,
        "seed": seed,
        "selected_method": model,
        "selection_rule": "validation_mape" if family != "HeteroRumorDyn-V1" else "single_model",
        "observation_minutes": int(round(float(metrics["observation_window_minutes"]))),
        "num_samples": int(metrics["num_samples"]),
        "val_mape": float(split_payload["val"]["mape"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mape": float(metrics["mape"]),
        "smape": float(metrics["smape"]),
        "r2": float(metrics["r2"]),
        "source_file": str(path),
        "predictions_file": str(path).replace("_metrics.json", "_predictions.csv"),
    }


def load_all_runs() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for family in FAMILY_ORDER:
            family_seeds = [42] if family == "SEIZ/CD-SEIZ" else SEEDS
            for seed in family_seeds:
                rows.append(load_run(family, dataset, "stratified", seed))
            rows.append(load_run(family, dataset, "temporal", 42))
    return rows


def aggregate_main(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and row["split_strategy"] == "stratified"
        ]
        family_means = {
            family: np.mean([row["mape"] for row in subset if row["family"] == family])
            for family in FAMILY_ORDER
        }
        best = min(family_means, key=family_means.get)
        for family in FAMILY_ORDER:
            selected = [row for row in subset if row["family"] == family]
            method_counts = Counter(row["selected_method"] for row in selected)
            output.append(
                {
                    "Dataset": dataset,
                    "Method family": family,
                    "Selected branch": "; ".join(
                        f"{name} ({count}/{len(selected)})"
                        for name, count in sorted(method_counts.items())
                    ),
                    "Seeds": len(selected),
                    "MAE": fmt_pm([row["mae"] for row in selected]),
                    "RMSE": fmt_pm([row["rmse"] for row in selected]),
                    "MAPE": fmt_pm([row["mape"] for row in selected]),
                    "SMAPE": fmt_pm([row["smape"] for row in selected]),
                    "R2": fmt_pm([row["r2"] for row in selected]),
                    "Best MAPE": "yes" if family == best else "",
                }
            )
    return output


def temporal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and row["split_strategy"] == "temporal"
        ]
        best = min(subset, key=lambda row: row["mape"])["family"]
        for family in FAMILY_ORDER:
            row = next(item for item in subset if item["family"] == family)
            output.append(
                {
                    "Dataset": dataset,
                    "Method family": family,
                    "Selected branch": row["selected_method"],
                    "Seed": row["seed"],
                    "MAE": fmt(row["mae"]),
                    "RMSE": fmt(row["rmse"]),
                    "MAPE": fmt(row["mape"]),
                    "SMAPE": fmt(row["smape"]),
                    "R2": fmt(row["r2"]),
                    "Best MAPE": "yes" if family == best else "",
                }
            )
    return output


def prediction_ape(run: dict[str, Any]) -> dict[str, float]:
    rows = read_csv(Path(run["predictions_file"]))
    selected = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        if row.get("model") != run["selected_method"]:
            continue
        selected[row["sample_id"]] = float(row["absolute_percentage_error"])
    return selected


def bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float, float, float]:
    delta = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    return (
        float(delta.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        float(np.mean(boot < 0)),
    )


def fairness_and_bootstrap(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits = []
    bootstrap_rows = []
    for dataset in DATASETS:
        for split_strategy in ("stratified", "temporal"):
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["split_strategy"] == split_strategy
                and row["seed"] == 42
            ]
            predictions = {row["family"]: prediction_ape(row) for row in subset}
            reference_ids = set(predictions["HeteroRumorDyn-V1"])
            for row in subset:
                sample_ids = set(predictions[row["family"]])
                audits.append(
                    {
                        "Dataset": dataset,
                        "Split": split_strategy,
                        "Method family": row["family"],
                        "Observation": row["observation_minutes"],
                        "Split seed": 42,
                        "Test samples": row["num_samples"],
                        "Prediction IDs": len(sample_ids),
                        "Exact ID match": "yes" if sample_ids == reference_ids else "no",
                        "Selection": row["selection_rule"],
                    }
                )
            reference = predictions["HeteroRumorDyn-V1"]
            ordered_ids = sorted(reference_ids)
            for index, family in enumerate(FAMILY_ORDER[1:]):
                other = predictions[family]
                common = sorted(reference_ids & set(other))
                left = np.asarray([reference[sample_id] for sample_id in common])
                right = np.asarray([other[sample_id] for sample_id in common])
                delta, lower, upper, probability = bootstrap_delta(
                    left,
                    right,
                    seed=42 + index + 100 * DATASETS.index(dataset),
                )
                bootstrap_rows.append(
                    {
                        "Dataset": dataset,
                        "Split": split_strategy,
                        "Comparison": f"HeteroRumorDyn-V1 - {family}",
                        "Paired samples": len(common),
                        "MAPE delta": fmt(delta),
                        "95% CI lower": fmt(lower),
                        "95% CI upper": fmt(upper),
                        "P(delta < 0)": fmt(probability),
                        "Interpretation": (
                            "HeteroRumorDyn lower error"
                            if upper < 0
                            else "Baseline lower error"
                            if lower > 0
                            else "CI overlaps zero"
                        ),
                    }
                )
    return audits, bootstrap_rows


def main() -> None:
    rows = load_all_runs()
    long_fields = list(rows[0].keys())
    write_csv(SUMMARY / "v1_fair180_all_runs.csv", rows, long_fields)

    main_fields = [
        "Dataset",
        "Method family",
        "Selected branch",
        "Seeds",
        "MAE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "R2",
        "Best MAPE",
    ]
    temporal_fields = [
        "Dataset",
        "Method family",
        "Selected branch",
        "Seed",
        "MAE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "R2",
        "Best MAPE",
    ]
    emit("paper_v1_fair180_main_table", aggregate_main(rows), main_fields)
    emit("paper_v1_fair180_temporal_table", temporal_rows(rows), temporal_fields)

    audits, bootstrap_rows = fairness_and_bootstrap(rows)
    audit_fields = [
        "Dataset",
        "Split",
        "Method family",
        "Observation",
        "Split seed",
        "Test samples",
        "Prediction IDs",
        "Exact ID match",
        "Selection",
    ]
    bootstrap_fields = [
        "Dataset",
        "Split",
        "Comparison",
        "Paired samples",
        "MAPE delta",
        "95% CI lower",
        "95% CI upper",
        "P(delta < 0)",
        "Interpretation",
    ]
    write_csv(SUMMARY / "v1_fair180_fairness_audit.csv", audits, audit_fields)
    emit("paper_v1_fair180_paired_bootstrap", bootstrap_rows, bootstrap_fields)
    print("Generated fixed-180-minute V1 fair comparison tables.")


if __name__ == "__main__":
    main()
