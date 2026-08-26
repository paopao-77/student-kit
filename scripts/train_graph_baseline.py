import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import RumorDataset


DEFAULT_RESULTS_DIR = Path("results/graph_baseline")
SUPPORTED_DATASETS = [
    "weibo",
    "twitter15",
    "twitter16",
    "twitter15_rumdetect2017",
    "twitter16_rumdetect2017",
    "pheme",
]
BASE_NODE_FEATURES = [
    "node_in_degree",
    "node_out_degree",
    "node_total_degree",
    "node_is_root",
    "node_depth",
    "node_delay",
]
POOL_NAMES = ["mean", "max", "std"]
GLOBAL_FEATURES = [
    "num_nodes",
    "num_edges",
    "cascade_density",
    "avg_branching_factor",
    "max_depth",
    "max_delay_minutes",
]
FEATURE_NAMES = [
    *[f"base_{pool}_{name}" for pool in POOL_NAMES for name in BASE_NODE_FEATURES],
    *[f"topdown_{pool}_{name}" for pool in POOL_NAMES for name in BASE_NODE_FEATURES],
    *[f"bottomup_{pool}_{name}" for pool in POOL_NAMES for name in BASE_NODE_FEATURES],
    *GLOBAL_FEATURES,
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


def read_edges_for_samples(
    edge_path: Path,
    sample_ids: set[str],
    max_edges_per_graph: int | None = None,
) -> dict[str, list[tuple[str, str, float]]]:
    edges_by_sample: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    with edge_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            if max_edges_per_graph is not None and len(edges_by_sample[sample_id]) >= max_edges_per_graph:
                continue
            parent = row.get("parent_tweet_id", "")
            child = row.get("child_tweet_id", "")
            if not child or child == "ROOT":
                continue
            if parent == "ROOT":
                parent = ""
            delay = finite_float(row.get("child_delay_minutes"), 0.0)
            edges_by_sample[sample_id].append((parent, child, delay))
    return dict(edges_by_sample)


def node_index(edges: list[tuple[str, str, float]]) -> dict[str, int]:
    nodes = {}
    for parent, child, _delay in edges:
        if parent and parent not in nodes:
            nodes[parent] = len(nodes)
        if child and child not in nodes:
            nodes[child] = len(nodes)
    return nodes


def graph_arrays(
    sample: dict[str, Any],
    edges: list[tuple[str, str, float]],
) -> tuple[np.ndarray, list[list[int]], list[list[int]], list[int], float, int]:
    nodes = node_index(edges)
    if not nodes:
        source_id = sample.get("sample_id", "source")
        nodes[source_id] = 0

    n_nodes = len(nodes)
    parents_of = [[] for _ in range(n_nodes)]
    children_of = [[] for _ in range(n_nodes)]
    in_degree = np.zeros(n_nodes, dtype=np.float32)
    out_degree = np.zeros(n_nodes, dtype=np.float32)
    delay = np.zeros(n_nodes, dtype=np.float32)

    for parent, child, child_delay in edges:
        child_idx = nodes.get(child)
        if child_idx is None:
            continue
        delay[child_idx] = max(delay[child_idx], float(child_delay))
        if not parent:
            continue
        parent_idx = nodes.get(parent)
        if parent_idx is None:
            continue
        parents_of[child_idx].append(parent_idx)
        children_of[parent_idx].append(child_idx)
        in_degree[child_idx] += 1.0
        out_degree[parent_idx] += 1.0

    roots = [idx for idx, degree in enumerate(in_degree) if degree == 0]
    depth = compute_depth(children_of, roots)
    max_depth = max(depth) if depth else 0
    max_delay = float(delay.max()) if len(delay) else finite_float(sample.get("max_delay_minutes"), 0.0)
    if max_delay == 0:
        max_delay = finite_float(sample.get("max_delay_minutes"), 0.0)

    features = np.stack(
        [
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(in_degree + out_degree),
            np.array([1.0 if idx in roots else 0.0 for idx in range(n_nodes)], dtype=np.float32),
            np.array(depth, dtype=np.float32) / max(max_depth, 1),
            np.log1p(delay) / max(math.log1p(max_delay), 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    return features, parents_of, children_of, roots, max_delay, max_depth


def compute_depth(children_of: list[list[int]], roots: list[int]) -> list[int]:
    depth = [-1 for _ in children_of]
    queue: deque[int] = deque()
    for root in roots:
        depth[root] = 0
        queue.append(root)
    while queue:
        parent = queue.popleft()
        for child in children_of[parent]:
            next_depth = depth[parent] + 1
            if depth[child] < 0 or next_depth < depth[child]:
                depth[child] = next_depth
                queue.append(child)
    return [value if value >= 0 else 0 for value in depth]


def propagate(features: np.ndarray, neighbors: list[list[int]], hops: int) -> np.ndarray:
    hidden = features
    for _ in range(hops):
        next_hidden = np.empty_like(hidden)
        for idx, neighbor_ids in enumerate(neighbors):
            if neighbor_ids:
                messages = hidden[neighbor_ids].mean(axis=0)
                next_hidden[idx] = 0.5 * hidden[idx] + 0.5 * messages
            else:
                next_hidden[idx] = hidden[idx]
        hidden = next_hidden
    return hidden


def pool(features: np.ndarray) -> list[float]:
    if features.size == 0:
        return [0.0] * (len(BASE_NODE_FEATURES) * len(POOL_NAMES))
    values = []
    for reducer in (np.mean, np.max, np.std):
        values.extend(float(value) for value in reducer(features, axis=0))
    return values


def graph_embedding(sample: dict[str, Any], edges: list[tuple[str, str, float]], hops: int) -> list[float]:
    node_features, parents_of, children_of, _roots, max_delay, max_depth = graph_arrays(sample, edges)
    topdown = propagate(node_features, parents_of, hops)
    bottomup = propagate(node_features, children_of, hops)

    n_nodes = max(float(node_features.shape[0]), finite_float(sample.get("num_nodes"), 0.0), 1.0)
    n_edges = max(float(len(edges)), finite_float(sample.get("num_edges"), 0.0), 0.0)
    possible_edges = n_nodes * max(n_nodes - 1.0, 1.0)
    global_features = [
        math.log1p(n_nodes),
        math.log1p(n_edges),
        n_edges / possible_edges if n_nodes > 1 else 0.0,
        n_edges / max(n_nodes - 1.0, 1.0) if n_nodes > 1 else 0.0,
        float(max_depth),
        math.log1p(max_delay),
    ]
    return [*pool(node_features), *pool(topdown), *pool(bottomup), *global_features]


def dataset_to_xy(
    dataset: RumorDataset,
    edges_by_sample: dict[str, list[tuple[str, str, float]]],
    hops: int,
) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    x_rows = []
    y_rows = []
    meta_rows = []
    for sample in dataset:
        edges = edges_by_sample.get(sample["sample_id"], [])
        x_rows.append(graph_embedding(sample, edges, hops))
        y_rows.append(int(sample["label_id"]))
        meta_rows.append(
            {
                "dataset": sample["dataset"],
                "sample_id": sample["sample_id"],
                "raw_label": sample["raw_label"],
                "label_id": int(sample["label_id"]),
            }
        )
    return x_rows, y_rows, meta_rows


def make_models(seed: int) -> dict[str, Any]:
    return {
        "sgc_logistic_regression": Pipeline(
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
        "bigcn_style_random_forest": RandomForestClassifier(
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
    metrics["auc"] = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(set(y_true)) == 2 else None
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

    sample_ids = {
        sample["sample_id"]
        for split_dataset in (train_dataset, val_dataset, test_dataset)
        for sample in split_dataset
    }
    edge_path = Path(args.data_root) / args.dataset / "edges.csv"
    edges_by_sample = read_edges_for_samples(
        edge_path,
        sample_ids,
        max_edges_per_graph=args.max_edges_per_graph,
    )

    x_train, y_train, train_meta = dataset_to_xy(train_dataset, edges_by_sample, args.hops)
    x_val, y_val, val_meta = dataset_to_xy(val_dataset, edges_by_sample, args.hops)
    x_test, y_test, test_meta = dataset_to_xy(test_dataset, edges_by_sample, args.hops)

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
                for meta, pred, score in zip(
                    meta_rows,
                    y_pred,
                    y_score if y_score is not None else [None] * len(y_pred),
                ):
                    prediction_rows.append(
                        {
                            "dataset": args.dataset,
                            "split_strategy": args.split_strategy,
                            "model": model_name,
                            "sample_id": meta["sample_id"],
                            "raw_label": meta["raw_label"],
                            "label_id": meta["label_id"],
                            "pred_label_id": pred,
                            "score_label_1": score,
                        }
                    )

    result = {
        "dataset": args.dataset,
        "task": args.task,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "hops": args.hops,
        "max_edges_per_graph": args.max_edges_per_graph,
        "feature_names": FEATURE_NAMES,
        "num_graphs": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "num_graphs_with_edges": sum(1 for sample_id in sample_ids if edges_by_sample.get(sample_id)),
        "split_summaries": {
            "train": train_dataset.split_summary(),
            "val": val_dataset.split_summary(),
            "test": test_dataset.split_summary(),
        },
        "models": metrics_by_model,
    }
    edge_suffix = f"_maxe{args.max_edges_per_graph}" if args.max_edges_per_graph is not None else ""
    prefix = f"{args.dataset}_{args.task}_{args.split_strategy}_sgc_h{args.hops}{edge_suffix}_seed{args.seed}"
    metrics_path = Path(args.output_dir) / f"{prefix}_metrics.json"
    predictions_path = Path(args.output_dir) / f"{prefix}_predictions.csv"
    write_json(metrics_path, result)
    write_predictions(predictions_path, prediction_rows)
    result["outputs"] = {"metrics": str(metrics_path), "predictions": str(predictions_path)}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--task", default="rumor_binary", choices=["rumor_binary"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-edges-per-graph", type=int)
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
        "hops": result["hops"],
        "max_edges_per_graph": result["max_edges_per_graph"],
        "num_graphs": result["num_graphs"],
        "num_graphs_with_edges": result["num_graphs_with_edges"],
        "outputs": result["outputs"],
        "test_metrics": {name: metrics["test"] for name, metrics in result["models"].items()},
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
