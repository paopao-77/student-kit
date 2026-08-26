import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import RumorDataset


DEFAULT_OUTPUT_DIR = Path("results/paper_baselines/midpms_adapted")
DEFAULT_OBSERVATION_WINDOWS = "60,180,360"


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


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"Expected at least one numeric value, got {raw!r}")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer value, got {raw!r}")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    return dict(grouped)


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


def load_split_dataset(args: argparse.Namespace, split: str) -> RumorDataset:
    return RumorDataset(
        dataset=args.dataset,
        data_root=args.data_root,
        label_map_path=args.label_map,
        task=args.label_task,
        split=split,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )


def load_event_delay_lookup(dataset_dir: Path) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = defaultdict(dict)
    event_path = dataset_dir / "events.csv"
    if not event_path.exists():
        return {}
    for row in read_csv(event_path):
        sample_id = row.get("sample_id", "")
        tweet_id = row.get("tweet_id", "")
        if not sample_id or not tweet_id or tweet_id == "ROOT":
            continue
        delay = row.get("delay_minutes", "")
        if delay == "":
            delay = row.get("event_order", "")
        value = finite_float(delay, -1.0)
        if value >= 0:
            lookup[sample_id][tweet_id] = value
    return dict(lookup)


def early_snapshot_rows(rows: list[dict[str, str]], observation_window: float) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda row: finite_int(row.get("window_index")))
    selected = [row for row in rows if finite_float(row.get("window_end")) <= observation_window]
    if selected:
        return selected
    return rows[:1]


def macro_features(rows: list[dict[str, str]], observation_window: float) -> dict[str, float]:
    prefix = early_snapshot_rows(rows, observation_window)
    if not prefix:
        return {name: 0.0 for name in MACRO_FEATURES}
    last = prefix[-1]
    series = {
        "new_nodes": [finite_float(row.get("new_nodes")) for row in prefix],
        "new_edges": [finite_float(row.get("new_edges")) for row in prefix],
        "cumulative_nodes": [finite_float(row.get("cumulative_nodes")) for row in prefix],
        "cumulative_edges": [finite_float(row.get("cumulative_edges")) for row in prefix],
        "active_communities": [finite_float(row.get("active_communities")) for row in prefix],
        "branch_community_ratio": [finite_float(row.get("branch_community_ratio")) for row in prefix],
        "cross_edge_ratio": [finite_float(row.get("cross_edge_ratio")) for row in prefix],
    }
    cumulative_nodes = max(finite_float(last.get("cumulative_nodes")), 1.0)
    cumulative_edges = max(finite_float(last.get("cumulative_edges")), 1.0)
    active_communities = max(finite_float(last.get("active_communities")), 1.0)
    features = {
        "obs_window": observation_window,
        "prefix_len": float(len(prefix)),
        "macro_new_nodes_last": math.log1p(finite_float(last.get("new_nodes"))),
        "macro_nodes": math.log1p(cumulative_nodes),
        "macro_new_edges_last": math.log1p(finite_float(last.get("new_edges"))),
        "macro_edges": math.log1p(cumulative_edges),
        "macro_active_communities": math.log1p(active_communities),
        "macro_branch_community_ratio": finite_float(last.get("branch_community_ratio")),
        "macro_cross_edge_ratio": finite_float(last.get("cross_edge_ratio")),
        "macro_node_growth_rate": finite_float(last.get("new_nodes")) / cumulative_nodes,
        "macro_edge_growth_rate": finite_float(last.get("new_edges")) / cumulative_edges,
        "macro_community_growth_rate": finite_float(last.get("new_communities")) / active_communities,
    }
    for name, values in series.items():
        features[f"{name}_mean"] = float(np.mean(values)) if values else 0.0
        features[f"{name}_std"] = float(np.std(values)) if values else 0.0
        features[f"{name}_slope"] = vector_slope(values)
        features[f"{name}_max"] = float(np.max(values)) if values else 0.0
    return {name: features.get(name, 0.0) for name in MACRO_FEATURES}


def edge_delay(row: dict[str, str], sample_lookup: dict[str, float]) -> float:
    raw = row.get("child_delay_minutes", "")
    if raw == "":
        raw = sample_lookup.get(row.get("child_tweet_id", ""), "")
    return finite_float(raw, -1.0)


def node_id(row: dict[str, str], prefix: str) -> str:
    user = row.get(f"{prefix}_user_id", "")
    tweet = row.get(f"{prefix}_tweet_id", "")
    return user or tweet


def micro_features(
    edges: list[dict[str, str]],
    event_lookup: dict[str, float],
    observation_window: float,
) -> dict[str, float]:
    observed_edges = []
    nodes: set[str] = set()
    adjacency: dict[str, list[str]] = defaultdict(list)
    parent_set: set[str] = set()
    child_set: set[str] = set()

    for row in edges:
        delay = edge_delay(row, event_lookup)
        if delay < 0 or delay > observation_window:
            continue
        parent = node_id(row, "parent")
        child = node_id(row, "child")
        if not child or child == "ROOT":
            continue
        if parent and parent != "ROOT":
            adjacency[parent].append(child)
            parent_set.add(parent)
        child_set.add(child)
        nodes.add(child)
        if parent and parent != "ROOT":
            nodes.add(parent)
        observed_edges.append((parent, child))

    num_nodes = max(len(nodes), 1)
    num_edges = len(observed_edges)
    outdegrees = [len(children) for children in adjacency.values()]
    roots = [child for parent, child in observed_edges if parent == "ROOT"]
    if not roots:
        roots = list(parent_set - child_set) or list(child_set - parent_set)[:1]

    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((root, 0) for root in roots if root)
    while queue:
        node, depth = queue.popleft()
        if node in depths and depths[node] <= depth:
            continue
        depths[node] = depth
        for child in adjacency.get(node, []):
            queue.append((child, depth + 1))

    leaf_nodes = [node for node in nodes if len(adjacency.get(node, [])) == 0]
    depth_values = list(depths.values())
    root_outdegree = max((len(adjacency.get(root, [])) for root in roots), default=0)
    features = {
        "micro_nodes": math.log1p(num_nodes),
        "micro_edges": math.log1p(num_edges),
        "micro_edge_node_ratio": num_edges / num_nodes,
        "micro_num_roots": math.log1p(len(roots)),
        "micro_max_depth": float(max(depth_values, default=0)),
        "micro_mean_depth": float(np.mean(depth_values)) if depth_values else 0.0,
        "micro_depth_std": float(np.std(depth_values)) if depth_values else 0.0,
        "micro_leaf_ratio": len(leaf_nodes) / num_nodes,
        "micro_branching_mean": float(np.mean(outdegrees)) if outdegrees else 0.0,
        "micro_branching_max": float(max(outdegrees, default=0)),
        "micro_root_outdegree": float(root_outdegree),
        "micro_reach_depth_ratio": len(depths) / num_nodes,
    }
    return {name: features.get(name, 0.0) for name in MICRO_FEATURES}


MACRO_FEATURES = [
    "obs_window",
    "prefix_len",
    "macro_new_nodes_last",
    "macro_nodes",
    "macro_new_edges_last",
    "macro_edges",
    "macro_active_communities",
    "macro_branch_community_ratio",
    "macro_cross_edge_ratio",
    "macro_node_growth_rate",
    "macro_edge_growth_rate",
    "macro_community_growth_rate",
    "new_nodes_mean",
    "new_nodes_std",
    "new_nodes_slope",
    "new_nodes_max",
    "new_edges_mean",
    "new_edges_std",
    "new_edges_slope",
    "new_edges_max",
    "cumulative_nodes_slope",
    "cumulative_edges_slope",
    "active_communities_slope",
    "branch_community_ratio_slope",
    "cross_edge_ratio_slope",
]

MICRO_FEATURES = [
    "micro_nodes",
    "micro_edges",
    "micro_edge_node_ratio",
    "micro_num_roots",
    "micro_max_depth",
    "micro_mean_depth",
    "micro_depth_std",
    "micro_leaf_ratio",
    "micro_branching_mean",
    "micro_branching_max",
    "micro_root_outdegree",
    "micro_reach_depth_ratio",
]


def load_feature_context(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.data_root) / args.dataset
    return {
        "snapshots": group_by_sample(
            read_csv(dataset_dir / "dynamic_snapshots" / "snapshots.csv")
        ),
        "edges": group_by_sample(read_csv(dataset_dir / "edges.csv")),
        "event_delays": load_event_delay_lookup(dataset_dir),
    }


def build_rows(
    args: argparse.Namespace,
    split: str,
    observation_window: float,
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    dataset = load_split_dataset(args, split)
    context = context or load_feature_context(args)
    snapshots = context["snapshots"]
    edges = context["edges"]
    event_delays = context["event_delays"]

    rows: list[dict[str, Any]] = []
    for sample in dataset:
        sample_id = sample["sample_id"]
        macro = macro_features(snapshots.get(sample_id, []), observation_window)
        micro = micro_features(edges.get(sample_id, []), event_delays.get(sample_id, {}), observation_window)
        rows.append(
            {
                "dataset": args.dataset,
                "split_strategy": args.split_strategy,
                "split": split,
                "observation_window_minutes": observation_window,
                "sample_id": sample_id,
                "raw_label": sample["raw_label"],
                "label_id": sample["label_id"],
                "final_size": max(int(sample.get("num_nodes") or 1), 1),
                **macro,
                **micro,
            }
        )
    return rows, MACRO_FEATURES, MICRO_FEATURES


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    denom = np.maximum(np.abs(y_true), 1.0)
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
    return {
        "num_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(abs_errors / denom)),
        "smape": float(np.mean(2.0 * abs_errors / smape_denom)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "median_ae": float(np.median(abs_errors)),
    }


def rows_to_matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[finite_float(row.get(name)) for name in features] for row in rows], dtype=np.float64)
    y = np.asarray([finite_float(row["final_size"], 1.0) for row in rows], dtype=np.float64)
    return x, y


def make_models(seed: int, n_jobs: int = -1) -> dict[str, tuple[Any, str]]:
    return {
        "midpms_macro": (
            Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0))]),
            "macro",
        ),
        "midpms_micro": (
            RandomForestRegressor(
                n_estimators=260,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            "micro",
        ),
        "midpms_adapted": (
            GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=3,
                subsample=0.85,
                random_state=seed,
            ),
            "fusion",
        ),
    }


def fit_predict(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_rows: np.ndarray) -> np.ndarray:
    model.fit(x_train, np.log1p(y_train))
    pred = np.expm1(model.predict(x_rows))
    return np.clip(pred, 1.0, None)


def prepare_feature_rows(args: argparse.Namespace) -> dict[float, dict[str, list[dict[str, Any]]]]:
    windows = parse_float_list(args.observation_windows)
    context = load_feature_context(args)
    prepared = {}
    for window in windows:
        prepared[window] = {
            split: build_rows(args, split, window, context=context)[0]
            for split in ("train", "val", "test")
        }
    return prepared


def run(
    args: argparse.Namespace,
    prepared_rows: dict[float, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    windows = parse_float_list(args.observation_windows)
    prepared_rows = prepared_rows or prepare_feature_rows(args)
    models_payload: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []

    for window in windows:
        split_rows = prepared_rows[window]
        macro_names = MACRO_FEATURES
        micro_names = MICRO_FEATURES

        feature_sets = {
            "macro": macro_names,
            "micro": micro_names,
            "fusion": macro_names + micro_names,
        }
        matrices = {
            split: {name: rows_to_matrix(rows, features) for name, features in feature_sets.items()}
            for split, rows in split_rows.items()
        }

        for model_base, (model, feature_role) in make_models(args.seed, args.n_jobs).items():
            model_name = f"{model_base}_w{int(window)}m"
            x_train, y_train = matrices["train"][feature_role]
            models_payload[model_name] = {}
            for split in ("train", "val", "test"):
                x_rows, y_true = matrices[split][feature_role]
                y_pred = fit_predict(model, x_train, y_train, x_rows) if split == "train" else np.clip(np.expm1(model.predict(x_rows)), 1.0, None)
                metrics = regression_metrics(y_true, y_pred)
                metrics["observation_window_minutes"] = window
                metrics["feature_role"] = feature_role
                models_payload[model_name][split] = metrics
                for row, truth, pred in zip(split_rows[split], y_true, y_pred):
                    all_predictions.append(
                        {
                            "dataset": args.dataset,
                            "split_strategy": args.split_strategy,
                            "split": split,
                            "model": model_name,
                            "feature_role": feature_role,
                            "observation_window_minutes": window,
                            "sample_id": row["sample_id"],
                            "raw_label": row["raw_label"],
                            "label_id": row["label_id"],
                            "final_size": float(truth),
                            "pred_final_size": float(pred),
                            "abs_error": float(abs(pred - truth)),
                            "absolute_percentage_error": float(abs(pred - truth) / max(truth, 1.0)),
                        }
                    )

    payload = {
        "dataset": args.dataset,
        "task": "cascade_size_prediction",
        "label_task": args.label_task,
        "split_strategy": args.split_strategy,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "model_family": "midpms_adapted",
        "target": "final_cascade_size",
        "observation_windows_minutes": windows,
        "feature_sets": {
            "macro": MACRO_FEATURES,
            "micro": MICRO_FEATURES,
            "fusion": MACRO_FEATURES + MICRO_FEATURES,
        },
        "notes": [
            "This is an adapted MIDPMS-style baseline using macro diffusion popularity features and micro propagation-path features.",
            "It is not an original-code reproduction; it uses the same local splits and observation windows as HeteroRumorDyn.",
        ],
        "models": models_payload,
    }

    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset}_cascade_size_{args.split_strategy}_midpms_adapted_seed{args.seed}"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    write_json(metrics_path, payload)
    write_csv(
        predictions_path,
        all_predictions,
        [
            "dataset",
            "split_strategy",
            "split",
            "model",
            "feature_role",
            "observation_window_minutes",
            "sample_id",
            "raw_label",
            "label_id",
            "final_size",
            "pred_final_size",
            "abs_error",
            "absolute_percentage_error",
        ],
    )
    payload["outputs"] = {"metrics": str(metrics_path), "predictions": str(predictions_path)}
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pheme", "twitter15", "twitter16", "weibo"])
    parser.add_argument("--label-task", default="rumor_binary", choices=["rumor_binary", "veracity", "raw"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--label-map", default="label_map.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--observation-windows", default=DEFAULT_OBSERVATION_WINDOWS)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    seeds = parse_int_list(args.seeds) if args.seeds else [args.seed]
    prepared_rows = prepare_feature_rows(args)
    outputs = []
    for seed in seeds:
        args.seed = seed
        payload = run(args, prepared_rows=prepared_rows)
        outputs.append(payload["outputs"])
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split_strategy": args.split_strategy,
                "seeds": seeds,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
