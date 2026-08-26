import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import default_split_path, load_split_ids


DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("results/c2_breakout")

BASE_FEATURES = [
    "window_index",
    "window_start",
    "window_end",
    "new_nodes",
    "cumulative_nodes",
    "new_edges",
    "cumulative_edges",
    "new_communities",
    "active_communities",
    "new_cross_edges",
    "cumulative_cross_edges",
    "cross_edge_ratio",
    "branch_community_ratio",
    "node_growth_rate",
    "edge_growth_rate",
    "community_growth_rate",
    "cross_growth_rate",
]

TREND_FEATURES = [
    "prefix_len",
    "new_nodes_mean",
    "new_nodes_std",
    "new_nodes_max",
    "new_nodes_slope",
    "cum_nodes_slope",
    "new_edges_mean",
    "new_edges_std",
    "new_edges_slope",
    "active_comm_slope",
    "branch_ratio_slope",
]

SPECTRAL_FEATURES = [
    "low_freq_energy_nodes",
    "low_freq_energy_edges",
    "low_freq_energy_communities",
    "spectral_smoothness_nodes",
]

CROSS_FEATURES = [
    "active_communities",
    "new_communities",
    "cumulative_cross_edges",
    "new_cross_edges",
    "cross_edge_ratio",
    "branch_community_ratio",
    "cross_growth_rate",
    "community_growth_rate",
    "active_comm_slope",
    "branch_ratio_slope",
]

DYNAMIC_ONLY_FEATURES = [
    "window_index",
    "window_start",
    "window_end",
    "prefix_len",
    "node_growth_rate",
    "edge_growth_rate",
    "community_growth_rate",
    "cross_growth_rate",
] + TREND_FEATURES + SPECTRAL_FEATURES

COMMUNITY_ONLY_FEATURES = [
    "active_communities",
    "new_communities",
    "community_growth_rate",
    "active_comm_slope",
    "cumulative_cross_edges",
    "new_cross_edges",
    "cross_edge_ratio",
    "branch_community_ratio",
    "cross_growth_rate",
    "branch_ratio_slope",
    "low_freq_energy_communities",
]


def finite_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite_int(value: Any, default: int = 0) -> int:
    return int(round(finite_float(value, default)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def group_by_sample(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("sample_id", "")].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda item: finite_int(item.get("window_index")))
    return dict(grouped)


def split_ids(dataset: str, split_strategy: str, split: str, args: argparse.Namespace) -> set[str]:
    split_file = (
        Path(args.split_file)
        if args.split_file
        else default_split_path(
            dataset=dataset,
            task=args.task,
            split_strategy=split_strategy,
            split_root=Path(args.split_root),
            seed=args.split_seed,
        )
    )
    ids, _metadata = load_split_ids(split_file, split, dataset, args.task)
    return set(ids)


def vector_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0:
        return 0.0
    return float(np.dot(x_centered, y - y.mean()) / denom)


def low_freq_energy(values: list[float], keep_ratio: float = 0.35) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - arr.mean()
    spectrum = np.fft.rfft(arr)
    power = np.abs(spectrum) ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    keep = max(1, int(math.ceil(len(power) * keep_ratio)))
    return float(power[:keep].sum() / total)


def spectral_smoothness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    denom = float(np.mean(np.abs(values))) + 1e-6
    return float(np.std(diffs) / denom)


def row_float(row: dict[str, str], name: str) -> float:
    return finite_float(row.get(name), 0.0)


def build_features(prefix_rows: list[dict[str, str]], feature_set: str) -> dict[str, float]:
    row = prefix_rows[-1]
    cumulative_nodes = max(row_float(row, "cumulative_nodes"), 1.0)
    cumulative_edges = max(row_float(row, "cumulative_edges"), 1.0)
    active_communities = max(row_float(row, "active_communities"), 1.0)
    cumulative_cross_edges = max(row_float(row, "cumulative_cross_edges"), 1.0)

    features = {
        "window_index": row_float(row, "window_index"),
        "window_start": row_float(row, "window_start"),
        "window_end": row_float(row, "window_end"),
        "new_nodes": math.log1p(row_float(row, "new_nodes")),
        "cumulative_nodes": math.log1p(cumulative_nodes),
        "new_edges": math.log1p(row_float(row, "new_edges")),
        "cumulative_edges": math.log1p(cumulative_edges),
        "new_communities": math.log1p(row_float(row, "new_communities")),
        "active_communities": math.log1p(active_communities),
        "new_cross_edges": math.log1p(row_float(row, "new_cross_edges")),
        "cumulative_cross_edges": math.log1p(cumulative_cross_edges),
        "cross_edge_ratio": row_float(row, "cross_edge_ratio"),
        "branch_community_ratio": row_float(row, "branch_community_ratio"),
        "node_growth_rate": row_float(row, "new_nodes") / cumulative_nodes,
        "edge_growth_rate": row_float(row, "new_edges") / cumulative_edges,
        "community_growth_rate": row_float(row, "new_communities") / active_communities,
        "cross_growth_rate": row_float(row, "new_cross_edges") / cumulative_cross_edges,
    }

    new_nodes = [row_float(item, "new_nodes") for item in prefix_rows]
    cumulative_nodes_series = [row_float(item, "cumulative_nodes") for item in prefix_rows]
    new_edges = [row_float(item, "new_edges") for item in prefix_rows]
    active_communities_series = [row_float(item, "active_communities") for item in prefix_rows]
    branch_ratio_series = [row_float(item, "branch_community_ratio") for item in prefix_rows]

    features.update(
        {
            "prefix_len": float(len(prefix_rows)),
            "new_nodes_mean": float(np.mean(new_nodes)) if new_nodes else 0.0,
            "new_nodes_std": float(np.std(new_nodes)) if new_nodes else 0.0,
            "new_nodes_max": float(np.max(new_nodes)) if new_nodes else 0.0,
            "new_nodes_slope": vector_slope(new_nodes),
            "cum_nodes_slope": vector_slope(cumulative_nodes_series),
            "new_edges_mean": float(np.mean(new_edges)) if new_edges else 0.0,
            "new_edges_std": float(np.std(new_edges)) if new_edges else 0.0,
            "new_edges_slope": vector_slope(new_edges),
            "active_comm_slope": vector_slope(active_communities_series),
            "branch_ratio_slope": vector_slope(branch_ratio_series),
        }
    )

    features.update(
        {
            "low_freq_energy_nodes": low_freq_energy(cumulative_nodes_series),
            "low_freq_energy_edges": low_freq_energy([row_float(item, "cumulative_edges") for item in prefix_rows]),
            "low_freq_energy_communities": low_freq_energy(active_communities_series),
            "spectral_smoothness_nodes": spectral_smoothness(cumulative_nodes_series),
        }
    )

    if feature_set == "static_topology":
        keep = set(BASE_FEATURES[:13])
    elif feature_set == "dynamic_temporal":
        keep = set(BASE_FEATURES + TREND_FEATURES)
    elif feature_set == "heterorumor_c2_no_lowfreq":
        keep = set(BASE_FEATURES + TREND_FEATURES + CROSS_FEATURES)
    elif feature_set == "heterorumor_c2_no_cross":
        keep = set(BASE_FEATURES + TREND_FEATURES + SPECTRAL_FEATURES) - set(CROSS_FEATURES)
    elif feature_set == "heterorumor_c2_no_temporal_trend":
        keep = (set(BASE_FEATURES + CROSS_FEATURES) - set(TREND_FEATURES)) - set(SPECTRAL_FEATURES)
    elif feature_set == "heterorumor_c2_dynamic_only":
        keep = set(DYNAMIC_ONLY_FEATURES)
    elif feature_set == "heterorumor_c2_community_only":
        keep = set(COMMUNITY_ONLY_FEATURES)
    else:
        keep = set(BASE_FEATURES + TREND_FEATURES + SPECTRAL_FEATURES + CROSS_FEATURES)

    return {name: features.get(name, 0.0) for name in sorted(keep)}


def eligible_rows(sample_rows: list[dict[str, str]], breakout: dict[str, str]) -> list[dict[str, str]]:
    if not sample_rows:
        return []
    has_breakout = finite_int(breakout.get("has_breakout"))
    if has_breakout:
        breakout_window = finite_int(breakout.get("breakout_window"))
        rows = [row for row in sample_rows if finite_int(row.get("window_index")) < breakout_window]
        return rows if rows else [sample_rows[0]]
    return sample_rows


def build_examples(
    sample_ids: set[str],
    snapshots: dict[str, list[dict[str, str]]],
    breakouts: dict[str, dict[str, str]],
    feature_set: str,
    max_windows_per_sample: int | None,
) -> tuple[list[list[float]], list[int], list[dict[str, Any]], list[str]]:
    feature_names: list[str] | None = None
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for sample_id in sorted(sample_ids):
        sample_rows = eligible_rows(snapshots.get(sample_id, []), breakouts.get(sample_id, {}))
        if max_windows_per_sample is not None:
            sample_rows = sample_rows[:max_windows_per_sample]
        label = finite_int(breakouts.get(sample_id, {}).get("has_breakout"))
        breakout_time = finite_float(breakouts.get(sample_id, {}).get("breakout_time"), 0.0)
        breakout_window = finite_int(breakouts.get(sample_id, {}).get("breakout_window"), -1)

        prefix: list[dict[str, str]] = []
        for row in sample_rows:
            prefix.append(row)
            features = build_features(prefix, feature_set)
            if feature_names is None:
                feature_names = list(features.keys())
            x_rows.append([features.get(name, 0.0) for name in feature_names])
            y_rows.append(label)
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "label_id": label,
                    "window_index": finite_int(row.get("window_index")),
                    "window_start": finite_float(row.get("window_start")),
                    "window_end": finite_float(row.get("window_end")),
                    "breakout_window": breakout_window if label else "",
                    "breakout_time": breakout_time if label else "",
                }
            )

    return x_rows, y_rows, meta_rows, feature_names or []


def make_models(seed: int, feature_set: str) -> dict[str, Any]:
    if feature_set == "static_topology":
        return {
            "static_logistic": Pipeline(
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
            "static_random_forest": RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            ),
        }
    if feature_set == "dynamic_temporal":
        return {
            "dynamic_random_forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
        }
    return {
        feature_set: GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        )
    }


def positive_scores(model: Any, x_rows: list[list[float]]) -> list[float]:
    probabilities = model.predict_proba(x_rows)
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0 for _ in x_rows]
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in probabilities]


def aggregate_sample_scores(
    meta_rows: list[dict[str, Any]],
    y_rows: list[int],
    scores: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], int, float]]] = defaultdict(list)
    for meta, label, score in zip(meta_rows, y_rows, scores):
        grouped[meta["sample_id"]].append((meta, label, score))

    rows = []
    for sample_id, items in grouped.items():
        items.sort(key=lambda item: int(item[0]["window_index"]))
        label = int(items[0][1])
        max_score = max(float(score) for _meta, _label, score in items)
        warning = next((item for item in items if float(item[2]) >= threshold), None)
        warning_meta = warning[0] if warning is not None else None
        pred = 1 if max_score >= threshold else 0
        lead_time = ""
        if label and pred and warning_meta is not None:
            lead_time = max(0.0, finite_float(warning_meta.get("breakout_time")) - finite_float(warning_meta.get("window_start")))
        rows.append(
            {
                "sample_id": sample_id,
                "label_id": label,
                "pred_label_id": pred,
                "score_label_1": max_score,
                "first_warning_window": warning_meta["window_index"] if warning_meta is not None else "",
                "first_warning_time": warning_meta["window_start"] if warning_meta is not None else "",
                "breakout_window": items[0][0].get("breakout_window", ""),
                "breakout_time": items[0][0].get("breakout_time", ""),
                "lead_time_minutes": lead_time,
                "num_eval_windows": len(items),
            }
        )
    return rows


def threshold_grid(scores: list[float]) -> list[float]:
    if not scores:
        return [0.5]
    quantiles = np.linspace(0.05, 0.95, 37)
    values = sorted({float(np.quantile(scores, q)) for q in quantiles} | {0.5})
    return values


def select_threshold(y_true: list[int], scores: list[float]) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in threshold_grid(scores):
        preds = [1 if score >= threshold else 0 for score in scores]
        value = f1_score(y_true, preds, zero_division=0)
        if value > best_f1:
            best_threshold = threshold
            best_f1 = value
    return float(best_threshold)


def precision_recall_at_k(y_true: list[int], scores: list[float], k: int) -> tuple[float, float]:
    if not y_true:
        return 0.0, 0.0
    k = max(1, min(k, len(y_true)))
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    positives = sum(1 for value in y_true if int(value) == 1)
    hits = sum(1 for idx in order if int(y_true[idx]) == 1)
    precision = hits / k
    recall = hits / positives if positives else 0.0
    return float(precision), float(recall)


def sample_metrics(sample_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y_true = [int(row["label_id"]) for row in sample_rows]
    scores = [float(row["score_label_1"]) for row in sample_rows]
    preds = [1 if score >= threshold else 0 for score in scores]
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )
    positives = sum(y_true)
    p_at_10, r_at_10 = precision_recall_at_k(y_true, scores, max(1, math.ceil(0.10 * len(y_true))))
    p_at_pos, r_at_pos = precision_recall_at_k(y_true, scores, max(1, positives))
    lead_times = [
        float(row["lead_time_minutes"])
        for row in sample_rows
        if int(row["label_id"]) == 1 and int(row["pred_label_id"]) == 1 and row["lead_time_minutes"] != ""
    ]
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "precision_at_10pct": p_at_10,
        "recall_at_10pct": r_at_10,
        "precision_at_pos_count": p_at_pos,
        "recall_at_pos_count": r_at_pos,
        "mean_lead_time_minutes": float(np.mean(lead_times)) if lead_times else 0.0,
        "median_lead_time_minutes": float(np.median(lead_times)) if lead_times else 0.0,
        "num_samples": len(sample_rows),
        "num_positive": int(positives),
        "warning_rate": float(sum(preds) / max(len(preds), 1)),
    }
    return metrics


def run_feature_set(
    feature_set: str,
    split_payloads: dict[str, tuple[list[list[float]], list[int], list[dict[str, Any]]]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    x_train, y_train, _train_meta = split_payloads["train"]
    x_val, y_val, val_meta = split_payloads["val"]
    feature_names = args._feature_names[feature_set]

    results: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for model_name, model in make_models(args.seed, feature_set).items():
        model.fit(x_train, y_train)
        val_scores = positive_scores(model, x_val)
        val_sample_rows_raw = aggregate_sample_scores(val_meta, y_val, val_scores, threshold=0.5)
        threshold = select_threshold(
            [int(row["label_id"]) for row in val_sample_rows_raw],
            [float(row["score_label_1"]) for row in val_sample_rows_raw],
        )

        results[model_name] = {}
        predictions[model_name] = []
        for split_name, (x_rows, y_rows, meta_rows) in split_payloads.items():
            scores = positive_scores(model, x_rows)
            sample_rows = aggregate_sample_scores(meta_rows, y_rows, scores, threshold=threshold)
            results[model_name][split_name] = sample_metrics(sample_rows, threshold)
            for row in sample_rows:
                predictions[model_name].append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "feature_set": feature_set,
                        **row,
                    }
                )
    return results, predictions, feature_names


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.data_root) / args.dataset
    snapshot_path = dataset_dir / "dynamic_snapshots" / "snapshots.csv"
    breakout_path = dataset_dir / "breakout_events.csv"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing snapshots: {snapshot_path}")
    if not breakout_path.exists():
        raise FileNotFoundError(f"Missing breakout events: {breakout_path}")

    snapshots = group_by_sample(read_csv(snapshot_path))
    breakouts = {row["sample_id"]: row for row in read_csv(breakout_path)}
    split_sample_ids = {
        split: split_ids(args.dataset, args.split_strategy, split, args)
        for split in ("train", "val", "test")
    }

    feature_sets = [
        "static_topology",
        "dynamic_temporal",
        "heterorumor_c2",
        "heterorumor_c2_no_lowfreq",
        "heterorumor_c2_no_cross",
        "heterorumor_c2_no_temporal_trend",
        "heterorumor_c2_dynamic_only",
        "heterorumor_c2_community_only",
    ]
    all_results: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    feature_names_by_set: dict[str, list[str]] = {}

    args._feature_names = {}
    for feature_set in feature_sets:
        split_payloads = {}
        for split, ids in split_sample_ids.items():
            x_rows, y_rows, meta_rows, feature_names = build_examples(
                ids,
                snapshots,
                breakouts,
                feature_set,
                args.max_windows_per_sample,
            )
            split_payloads[split] = (x_rows, y_rows, meta_rows)
            if feature_set not in args._feature_names:
                args._feature_names[feature_set] = feature_names
        results, predictions, names = run_feature_set(feature_set, split_payloads, args)
        all_results.update(results)
        feature_names_by_set[feature_set] = names
        for model_rows in predictions.values():
            all_predictions.extend(model_rows)

    payload = {
        "dataset": args.dataset,
        "task": "breakout_forecasting",
        "label_source": "data/processed/{dataset}/breakout_events.csv",
        "split_strategy": args.split_strategy,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "model_family": "heterorumor_c2",
        "time_mode": read_json(dataset_dir / "c2_foundation_stats.json").get("time_mode", ""),
        "window_minutes": read_json(dataset_dir / "c2_foundation_stats.json").get("window_minutes", ""),
        "max_windows_per_sample": args.max_windows_per_sample,
        "feature_names_by_set": feature_names_by_set,
        "split_sizes": {split: len(ids) for split, ids in split_sample_ids.items()},
        "models": all_results,
    }

    prefix = f"{args.dataset}_breakout_{args.split_strategy}_seed{args.seed}"
    output_dir = Path(args.output_dir)
    write_json(output_dir / f"{prefix}_metrics.json", payload)
    prediction_fields = [
        "split",
        "model",
        "feature_set",
        "sample_id",
        "label_id",
        "pred_label_id",
        "score_label_1",
        "first_warning_window",
        "first_warning_time",
        "breakout_window",
        "breakout_time",
        "lead_time_minutes",
        "num_eval_windows",
    ]
    write_csv(output_dir / f"{prefix}_predictions.csv", all_predictions, prediction_fields)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pheme", "twitter15", "twitter16", "weibo"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split-root", default=str(DEFAULT_DATA_ROOT / "splits"))
    parser.add_argument("--split-file")
    parser.add_argument("--task", default="rumor_binary")
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows-per-sample", type=int, default=12)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    payload = run(parse_args())
    print(json.dumps({"dataset": payload["dataset"], "models": list(payload["models"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
