import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset_loader import RumorDataset


DEFAULT_RESULTS_DIR = Path("results/baseline")
SUPPORTED_DATASETS = [
    "weibo",
    "twitter15",
    "twitter16",
    "twitter15_rumdetect2017",
    "twitter16_rumdetect2017",
    "pheme",
]
FEATURE_NAMES = [
    "num_nodes",
    "num_edges",
    "max_delay_minutes",
    "has_source_text",
    "cascade_density",
    "avg_branching_factor",
    "log_num_nodes",
    "log_num_edges",
    "log_max_delay_minutes",
]


def finite_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def extract_features(sample: dict[str, Any]) -> list[float]:
    num_nodes = max(finite_float(sample.get("num_nodes")), 0.0)
    num_edges = max(finite_float(sample.get("num_edges")), 0.0)
    max_delay = max(finite_float(sample.get("max_delay_minutes")), 0.0)
    has_source_text = 1.0 if sample.get("has_source_text") else 0.0

    possible_directed_edges = num_nodes * max(num_nodes - 1.0, 1.0)
    cascade_density = num_edges / possible_directed_edges if num_nodes > 1 else 0.0
    avg_branching_factor = num_edges / max(num_nodes - 1.0, 1.0) if num_nodes > 1 else 0.0

    return [
        num_nodes,
        num_edges,
        max_delay,
        has_source_text,
        cascade_density,
        avg_branching_factor,
        math.log1p(num_nodes),
        math.log1p(num_edges),
        math.log1p(max_delay),
    ]


def dataset_to_xy(dataset: RumorDataset) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    x_rows = []
    y_rows = []
    meta_rows = []
    for sample in dataset:
        label_id = int(sample["label_id"])
        x_rows.append(extract_features(sample))
        y_rows.append(label_id)
        meta_rows.append(
            {
                "dataset": sample["dataset"],
                "sample_id": sample["sample_id"],
                "raw_label": sample["raw_label"],
                "label_id": label_id,
                "num_nodes": sample["num_nodes"],
                "num_edges": sample["num_edges"],
                "max_delay_minutes": sample["max_delay_minutes"],
                "has_source_text": sample["has_source_text"],
            }
        )
    return x_rows, y_rows, meta_rows


def make_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def positive_scores(model: Any, x_rows: list[list[float]]) -> list[float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(x_rows)
    classes = list(model.classes_)
    if 1 not in classes:
        return None
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in probabilities]


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float] | None) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(set(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = None
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split_strategy",
        "model",
        "sample_id",
        "raw_label",
        "label_id",
        "pred_label_id",
        "score_label_1",
        *FEATURE_NAMES,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split_dataset(args: argparse.Namespace, split: str) -> RumorDataset:
    return RumorDataset(
        dataset=args.dataset,
        data_root=args.data_root,
        label_map_path=args.label_map,
        task=args.task,
        split=split,
        split_strategy=args.split_strategy,
        split_seed=args.seed,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_dataset = load_split_dataset(args, "train")
    val_dataset = load_split_dataset(args, "val")
    test_dataset = load_split_dataset(args, "test")

    x_train, y_train, train_meta = dataset_to_xy(train_dataset)
    x_val, y_val, val_meta = dataset_to_xy(val_dataset)
    x_test, y_test, test_meta = dataset_to_xy(test_dataset)

    models = make_models(args.seed)
    metrics_by_model = {}
    prediction_rows = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        split_payloads = {
            "train": (x_train, y_train, train_meta),
            "val": (x_val, y_val, val_meta),
            "test": (x_test, y_test, test_meta),
        }
        metrics_by_model[model_name] = {}
        for split_name, (x_rows, y_rows, meta_rows) in split_payloads.items():
            y_pred = [int(value) for value in model.predict(x_rows)]
            y_score = positive_scores(model, x_rows)
            metrics_by_model[model_name][split_name] = compute_metrics(y_rows, y_pred, y_score)

            if split_name == "test":
                for meta, features, pred, score in zip(
                    meta_rows,
                    x_rows,
                    y_pred,
                    y_score if y_score is not None else [None] * len(y_pred),
                ):
                    row = {
                        "dataset": args.dataset,
                        "split_strategy": args.split_strategy,
                        "model": model_name,
                        "sample_id": meta["sample_id"],
                        "raw_label": meta["raw_label"],
                        "label_id": meta["label_id"],
                        "pred_label_id": pred,
                        "score_label_1": score,
                    }
                    row.update(dict(zip(FEATURE_NAMES, features)))
                    prediction_rows.append(row)

    split_summaries = {
        "train": train_dataset.split_summary(),
        "val": val_dataset.split_summary(),
        "test": test_dataset.split_summary(),
    }
    result = {
        "dataset": args.dataset,
        "task": args.task,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "split_summaries": split_summaries,
        "models": metrics_by_model,
    }

    prefix = f"{args.dataset}_{args.task}_{args.split_strategy}_seed{args.seed}"
    metrics_path = Path(args.output_dir) / f"{prefix}_metrics.json"
    predictions_path = Path(args.output_dir) / f"{prefix}_predictions.csv"
    write_json(metrics_path, result)
    write_predictions(predictions_path, prediction_rows)
    result["outputs"] = {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--task", default="rumor_binary", choices=["rumor_binary"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--label-map", default="label_map.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    result = run(parse_args())
    compact = {
        "dataset": result["dataset"],
        "task": result["task"],
        "split_strategy": result["split_strategy"],
        "outputs": result["outputs"],
        "test_metrics": {name: metrics["test"] for name, metrics in result["models"].items()},
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
