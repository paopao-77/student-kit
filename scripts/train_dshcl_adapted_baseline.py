import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import RumorDataset


DEFAULT_OUTPUT_DIR = Path("results/paper_baselines/dshcl_adapted")
DEFAULT_OBSERVATION_WINDOWS = "60,180,360"

DIFFUSION_FEATURES = [
    "obs_window",
    "prefix_len",
    "diff_nodes",
    "diff_edges",
    "diff_active_communities",
    "diff_new_nodes_last",
    "diff_new_edges_last",
    "diff_node_growth_rate",
    "diff_edge_growth_rate",
    "diff_branch_community_ratio",
    "diff_cross_edge_ratio",
    "diff_nodes_slope",
    "diff_edges_slope",
    "diff_communities_slope",
    "diff_community_entropy",
    "diff_top_community_share",
    "diff_depth_mean",
    "diff_depth_std",
    "diff_depth_max",
    "diff_root_child_ratio",
]

INTERACTION_FEATURES = [
    "inter_nodes",
    "inter_edges",
    "inter_edge_node_ratio",
    "inter_roots",
    "inter_max_depth",
    "inter_mean_depth",
    "inter_depth_std",
    "inter_leaf_ratio",
    "inter_branching_mean",
    "inter_branching_max",
    "inter_root_outdegree",
    "inter_two_hop_paths",
    "inter_same_community_edge_ratio",
    "inter_cross_community_edges",
    "inter_cross_community_ratio",
    "inter_reach_depth_ratio",
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


def entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts: dict[Any, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())


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


def load_community_lookup(dataset_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = dataset_dir / "community_ids.csv"
    if not path.exists():
        return {}
    lookup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_csv(path):
        sample_id = row.get("sample_id", "")
        node_id = row.get("node_id", "")
        if not sample_id or not node_id:
            continue
        lookup[sample_id][node_id] = {
            "community_id": row.get("community_id", ""),
            "depth": finite_int(row.get("depth")),
            "is_root": row.get("is_root", "").lower() == "true",
            "is_root_child": row.get("is_root_child", "").lower() == "true",
        }
    return dict(lookup)


def early_snapshot_rows(rows: list[dict[str, str]], observation_window: float) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda row: finite_int(row.get("window_index")))
    selected = [row for row in rows if finite_float(row.get("window_end")) <= observation_window]
    if selected:
        return selected
    return rows[:1]


def observed_community_rows(
    sample_communities: dict[str, dict[str, Any]],
    sample_delays: dict[str, float],
    observation_window: float,
) -> list[dict[str, Any]]:
    observed = []
    for node_id, meta in sample_communities.items():
        delay = sample_delays.get(node_id)
        if delay is None:
            # If a node has no event delay, keep shallow nodes as proxy early exposure.
            if int(meta.get("depth", 0)) > 1:
                continue
        elif delay > observation_window:
            continue
        observed.append(meta)
    return observed


def diffusion_features(
    snapshots: list[dict[str, str]],
    sample_communities: dict[str, dict[str, Any]],
    sample_delays: dict[str, float],
    observation_window: float,
) -> dict[str, float]:
    prefix = early_snapshot_rows(snapshots, observation_window)
    last = prefix[-1] if prefix else {}
    cumulative_nodes = max(finite_float(last.get("cumulative_nodes")), 1.0)
    cumulative_edges = max(finite_float(last.get("cumulative_edges")), 1.0)
    active_communities = max(finite_float(last.get("active_communities")), 1.0)
    node_series = [finite_float(row.get("cumulative_nodes")) for row in prefix]
    edge_series = [finite_float(row.get("cumulative_edges")) for row in prefix]
    community_series = [finite_float(row.get("active_communities")) for row in prefix]
    observed_comms = observed_community_rows(sample_communities, sample_delays, observation_window)
    communities = [row.get("community_id", "") for row in observed_comms]
    depths = [float(row.get("depth", 0)) for row in observed_comms]
    top_share = 0.0
    if communities:
        counts: dict[str, int] = defaultdict(int)
        for community in communities:
            counts[str(community)] += 1
        top_share = max(counts.values()) / max(sum(counts.values()), 1)
    root_child_ratio = (
        sum(1 for row in observed_comms if row.get("is_root_child")) / max(len(observed_comms), 1)
        if observed_comms
        else 0.0
    )
    values = {
        "obs_window": observation_window,
        "prefix_len": float(len(prefix)),
        "diff_nodes": math.log1p(cumulative_nodes),
        "diff_edges": math.log1p(cumulative_edges),
        "diff_active_communities": math.log1p(active_communities),
        "diff_new_nodes_last": math.log1p(finite_float(last.get("new_nodes"))),
        "diff_new_edges_last": math.log1p(finite_float(last.get("new_edges"))),
        "diff_node_growth_rate": finite_float(last.get("new_nodes")) / cumulative_nodes,
        "diff_edge_growth_rate": finite_float(last.get("new_edges")) / cumulative_edges,
        "diff_branch_community_ratio": finite_float(last.get("branch_community_ratio")),
        "diff_cross_edge_ratio": finite_float(last.get("cross_edge_ratio")),
        "diff_nodes_slope": vector_slope(node_series),
        "diff_edges_slope": vector_slope(edge_series),
        "diff_communities_slope": vector_slope(community_series),
        "diff_community_entropy": entropy(communities),
        "diff_top_community_share": top_share,
        "diff_depth_mean": float(np.mean(depths)) if depths else 0.0,
        "diff_depth_std": float(np.std(depths)) if depths else 0.0,
        "diff_depth_max": float(max(depths, default=0.0)),
        "diff_root_child_ratio": root_child_ratio,
    }
    return {name: values.get(name, 0.0) for name in DIFFUSION_FEATURES}


def edge_delay(row: dict[str, str], sample_lookup: dict[str, float]) -> float:
    raw = row.get("child_delay_minutes", "")
    if raw == "":
        raw = sample_lookup.get(row.get("child_tweet_id", ""), "")
    return finite_float(raw, -1.0)


def node_id(row: dict[str, str], prefix: str) -> str:
    user = row.get(f"{prefix}_user_id", "")
    tweet = row.get(f"{prefix}_tweet_id", "")
    return user or tweet


def interaction_features(
    edges: list[dict[str, str]],
    sample_delays: dict[str, float],
    sample_communities: dict[str, dict[str, Any]],
    observation_window: float,
) -> dict[str, float]:
    observed_edges = []
    nodes: set[str] = set()
    adjacency: dict[str, list[str]] = defaultdict(list)
    parent_set: set[str] = set()
    child_set: set[str] = set()
    same_comm = 0
    cross_comm = 0
    for row in edges:
        delay = edge_delay(row, sample_delays)
        if delay < 0 or delay > observation_window:
            continue
        parent = node_id(row, "parent")
        child = node_id(row, "child")
        if not child or child == "ROOT":
            continue
        if parent and parent != "ROOT":
            adjacency[parent].append(child)
            parent_set.add(parent)
            nodes.add(parent)
            parent_comm = sample_communities.get(parent, {}).get("community_id")
            child_comm = sample_communities.get(child, {}).get("community_id")
            if parent_comm is not None and child_comm is not None:
                if str(parent_comm) == str(child_comm):
                    same_comm += 1
                else:
                    cross_comm += 1
        child_set.add(child)
        nodes.add(child)
        observed_edges.append((parent, child))

    num_nodes = max(len(nodes), 1)
    num_edges = len(observed_edges)
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
    outdegrees = [len(children) for children in adjacency.values()]
    leaf_nodes = [node for node in nodes if len(adjacency.get(node, [])) == 0]
    two_hop_paths = sum(len(adjacency.get(child, [])) for children in adjacency.values() for child in children)
    community_known_edges = same_comm + cross_comm
    values = {
        "inter_nodes": math.log1p(num_nodes),
        "inter_edges": math.log1p(num_edges),
        "inter_edge_node_ratio": num_edges / num_nodes,
        "inter_roots": math.log1p(len(roots)),
        "inter_max_depth": float(max(depths.values(), default=0)),
        "inter_mean_depth": float(np.mean(list(depths.values()))) if depths else 0.0,
        "inter_depth_std": float(np.std(list(depths.values()))) if depths else 0.0,
        "inter_leaf_ratio": len(leaf_nodes) / num_nodes,
        "inter_branching_mean": float(np.mean(outdegrees)) if outdegrees else 0.0,
        "inter_branching_max": float(max(outdegrees, default=0)),
        "inter_root_outdegree": float(max((len(adjacency.get(root, [])) for root in roots), default=0)),
        "inter_two_hop_paths": math.log1p(two_hop_paths),
        "inter_same_community_edge_ratio": same_comm / max(community_known_edges, 1),
        "inter_cross_community_edges": math.log1p(cross_comm),
        "inter_cross_community_ratio": cross_comm / max(community_known_edges, 1),
        "inter_reach_depth_ratio": len(depths) / num_nodes,
    }
    return {name: values.get(name, 0.0) for name in INTERACTION_FEATURES}


def load_feature_context(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.data_root) / args.dataset
    return {
        "snapshots": group_by_sample(
            read_csv(dataset_dir / "dynamic_snapshots" / "snapshots.csv")
        ),
        "edges": group_by_sample(read_csv(dataset_dir / "edges.csv")),
        "delays": load_event_delay_lookup(dataset_dir),
        "communities": load_community_lookup(dataset_dir),
    }


def build_rows(
    args: argparse.Namespace,
    split: str,
    observation_window: float,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dataset = load_split_dataset(args, split)
    context = context or load_feature_context(args)
    snapshots = context["snapshots"]
    edges = context["edges"]
    delays = context["delays"]
    communities = context["communities"]
    rows = []
    for sample in dataset:
        sample_id = sample["sample_id"]
        diffusion = diffusion_features(
            snapshots.get(sample_id, []),
            communities.get(sample_id, {}),
            delays.get(sample_id, {}),
            observation_window,
        )
        interaction = interaction_features(
            edges.get(sample_id, []),
            delays.get(sample_id, {}),
            communities.get(sample_id, {}),
            observation_window,
        )
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
                **diffusion,
                **interaction,
            }
        )
    return rows


def rows_to_matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[finite_float(row.get(name)) for name in features] for row in rows], dtype=np.float64)
    y = np.asarray([finite_float(row["final_size"], 1.0) for row in rows], dtype=np.float64)
    return x, y


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


def fit_contrastive_transform(
    x_diffusion_train: np.ndarray,
    x_interaction_train: np.ndarray,
    max_components: int,
    seed: int,
) -> dict[str, Any]:
    scaler_a = StandardScaler().fit(x_diffusion_train)
    scaler_b = StandardScaler().fit(x_interaction_train)
    za = scaler_a.transform(x_diffusion_train)
    zb = scaler_b.transform(x_interaction_train)
    n_components = max(1, min(max_components, za.shape[1], zb.shape[1], za.shape[0] - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cca = CCA(n_components=n_components, max_iter=800)
            cca.fit(za, zb)
        return {
            "mode": "cca",
            "scaler_a": scaler_a,
            "scaler_b": scaler_b,
            "model": cca,
            "n_components": n_components,
        }
    except Exception:
        concat = np.hstack([za, zb])
        n_components = max(1, min(max_components, concat.shape[1], concat.shape[0] - 1))
        pca = PCA(n_components=n_components, random_state=seed).fit(concat)
        return {
            "mode": "pca_fallback",
            "scaler_a": scaler_a,
            "scaler_b": scaler_b,
            "model": pca,
            "n_components": n_components,
        }


def transform_contrastive(transformer: dict[str, Any], x_diffusion: np.ndarray, x_interaction: np.ndarray) -> np.ndarray:
    za = transformer["scaler_a"].transform(x_diffusion)
    zb = transformer["scaler_b"].transform(x_interaction)
    if transformer["mode"] == "cca":
        ta, tb = transformer["model"].transform(za, zb)
        return np.hstack([za, zb, ta, tb, 0.5 * (ta + tb), np.abs(ta - tb)])
    concat = np.hstack([za, zb])
    return np.hstack([concat, transformer["model"].transform(concat)])


def fit_log_regressor(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    model.fit(x_train, np.log1p(y_train))
    return model


def predict_log_regressor(model: Any, x_rows: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(model.predict(x_rows)), 1.0, None)


def make_models(seed: int, n_jobs: int = -1) -> dict[str, Any]:
    return {
        "dshcl_diffusion": Pipeline(
            [("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=260, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs))]
        ),
        "dshcl_interaction": Pipeline(
            [("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=260, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs))]
        ),
        "dshcl_adapted": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        ),
    }


def prepare_feature_rows(args: argparse.Namespace) -> dict[float, dict[str, list[dict[str, Any]]]]:
    windows = parse_float_list(args.observation_windows)
    context = load_feature_context(args)
    return {
        window: {
            split: build_rows(args, split, window, context=context)
            for split in ("train", "val", "test")
        }
        for window in windows
    }


def run(
    args: argparse.Namespace,
    prepared_rows: dict[float, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    windows = parse_float_list(args.observation_windows)
    prepared_rows = prepared_rows or prepare_feature_rows(args)
    payload_models: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []

    for window in windows:
        split_rows = prepared_rows[window]
        matrices = {
            split: {
                "diffusion": rows_to_matrix(rows, DIFFUSION_FEATURES),
                "interaction": rows_to_matrix(rows, INTERACTION_FEATURES),
            }
            for split, rows in split_rows.items()
        }
        x_diff_train, y_train = matrices["train"]["diffusion"]
        x_inter_train, _ = matrices["train"]["interaction"]
        transformer = fit_contrastive_transform(x_diff_train, x_inter_train, args.contrastive_components, args.seed)
        contrastive = {
            split: (
                transform_contrastive(
                    transformer,
                    matrices[split]["diffusion"][0],
                    matrices[split]["interaction"][0],
                ),
                matrices[split]["diffusion"][1],
            )
            for split in ("train", "val", "test")
        }
        models = make_models(args.seed, args.n_jobs)
        fit_log_regressor(models["dshcl_diffusion"], x_diff_train, y_train)
        fit_log_regressor(models["dshcl_interaction"], x_inter_train, y_train)
        fit_log_regressor(models["dshcl_adapted"], contrastive["train"][0], y_train)
        model_specs = {
            "dshcl_diffusion": ("diffusion", models["dshcl_diffusion"], "diffusion"),
            "dshcl_interaction": ("interaction", models["dshcl_interaction"], "interaction"),
            "dshcl_adapted": ("contrastive", models["dshcl_adapted"], "contrastive"),
        }

        for model_base, (feature_role, model, matrix_key) in model_specs.items():
            model_name = f"{model_base}_w{int(window)}m"
            payload_models[model_name] = {}
            for split in ("train", "val", "test"):
                if matrix_key == "diffusion":
                    x_rows, y_true = matrices[split]["diffusion"]
                elif matrix_key == "interaction":
                    x_rows, y_true = matrices[split]["interaction"]
                else:
                    x_rows, y_true = contrastive[split]
                y_pred = predict_log_regressor(model, x_rows)
                metrics = regression_metrics(y_true, y_pred)
                metrics["observation_window_minutes"] = window
                metrics["feature_role"] = feature_role
                metrics["contrastive_mode"] = transformer["mode"] if matrix_key == "contrastive" else ""
                metrics["contrastive_components"] = transformer["n_components"] if matrix_key == "contrastive" else ""
                payload_models[model_name][split] = metrics
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
        "model_family": "dshcl_adapted",
        "target": "final_cascade_size",
        "observation_windows_minutes": windows,
        "feature_sets": {
            "diffusion": DIFFUSION_FEATURES,
            "interaction": INTERACTION_FEATURES,
            "contrastive": "CCA-aligned diffusion and interaction views",
        },
        "notes": [
            "This is a DSHCL-adapted baseline: diffusion-state and interaction-state hypergraph proxies are aligned through CCA as a lightweight contrastive representation.",
            "It is not an original-code reproduction and uses the same local splits and observation windows as HeteroRumorDyn.",
        ],
        "models": payload_models,
    }
    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset}_cascade_size_{args.split_strategy}_dshcl_adapted_seed{args.seed}"
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
    parser.add_argument("--contrastive-components", type=int, default=16)
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
