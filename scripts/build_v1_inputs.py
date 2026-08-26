import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import RumorDataset


DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "v1_inputs"
SUPPORTED_DATASETS = {
    "pheme",
    "twitter15",
    "twitter16",
    "twitter15_rumdetect2017",
    "twitter16_rumdetect2017",
    "weibo",
}
TEXT_FEATURE_NAMES = ["stable_hash_char_word_features"]
NODE_FEATURE_NAMES = [
    "log_in_degree",
    "log_out_degree",
    "log_total_degree",
    "is_root",
    "normalized_depth",
    "normalized_delay",
    "has_branch_community",
    "is_root_child",
]
GLOBAL_FEATURE_NAMES = [
    "log_observed_nodes",
    "log_observed_edges",
    "cascade_density",
    "avg_branching_factor",
    "max_depth",
    "log_max_observed_delay",
    "log_active_communities",
    "cross_edge_ratio",
]
TEMPORAL_FEATURE_NAMES = [
    "log_new_nodes",
    "log_cumulative_nodes",
    "log_new_edges",
    "log_cumulative_edges",
    "log_new_users",
    "log_cumulative_users",
    "log_new_communities",
    "log_active_communities",
    "cross_edge_ratio",
    "branch_community_ratio",
]
USER_FEATURE_NAMES = [
    "log_mean_followers",
    "log_max_followers",
    "log_mean_friends",
    "log_mean_statuses",
    "verified_ratio",
    "unique_user_ratio",
    "profile_coverage",
    "log_source_followers",
]
MODALITY_NAMES = ["text", "topology", "temporal", "user_profile"]


def finite_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("Observation windows must contain positive integers")
    return sorted(set(values))


def stable_hash_features(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    normalized = " ".join(text.lower().split())
    if not normalized:
        return vector

    tokens = normalized.split()
    features = list(tokens)
    compact = normalized.replace(" ", "_")
    for ngram_size in (2, 3, 4):
        features.extend(
            compact[index : index + ngram_size]
            for index in range(max(len(compact) - ngram_size + 1, 0))
        )

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dim
        sign = 1.0 if (value >> 8) & 1 else -1.0
        vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def read_events(
    dataset: str,
    path: Path,
    sample_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    times: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            node_id = row.get("tweet_id", "")
            if not node_id or node_id == "ROOT":
                continue
            if dataset == "weibo":
                time_value = finite_float(row.get("event_order"), finite_float(node_id, 0.0))
            else:
                time_value = finite_float(row.get("delay_minutes"), 0.0)
            time_value = max(time_value, 0.0)
            normalized = dict(row)
            normalized["node_id"] = node_id
            normalized["time_value"] = time_value
            grouped[sample_id].append(normalized)
            times[sample_id][node_id] = time_value
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["time_value"], row["node_id"]))
    return dict(grouped), dict(times)


def read_edges(
    dataset: str,
    path: Path,
    sample_ids: set[str],
    event_times: dict[str, dict[str, float]],
    max_edges_per_sample: int | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            parent = row.get("parent_tweet_id", "")
            child = row.get("child_tweet_id", "")
            if not child or child == "ROOT":
                continue
            if parent == "ROOT":
                parent = ""
            edge_key = (parent, child)
            if edge_key in seen[sample_id]:
                continue
            if max_edges_per_sample is not None and len(grouped[sample_id]) >= max_edges_per_sample:
                continue
            seen[sample_id].add(edge_key)
            if dataset == "weibo":
                time_value = event_times.get(sample_id, {}).get(child, finite_float(child, 0.0))
            else:
                time_value = finite_float(
                    row.get("child_delay_minutes"),
                    event_times.get(sample_id, {}).get(child, 0.0),
                )
            time_value = max(time_value, 0.0)
            grouped[sample_id].append(
                {"parent": parent, "child": child, "time_value": time_value}
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["time_value"], row["parent"], row["child"]))
    return dict(grouped)


def read_communities(path: Path, sample_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            node_id = row.get("node_id", "")
            grouped[sample_id][node_id] = {
                "community_id": int(finite_float(row.get("community_id"), 0.0)),
                "depth": int(finite_float(row.get("depth"), 0.0)),
                "is_root": str(row.get("is_root", "")).lower() == "true",
                "is_root_child": str(row.get("is_root_child", "")).lower() == "true",
            }
    return {sample_id: dict(rows) for sample_id, rows in grouped.items()}


def compute_depth(children_of: list[list[int]], roots: list[int]) -> list[int]:
    depth = [-1] * len(children_of)
    queue: deque[int] = deque()
    for root in roots:
        depth[root] = 0
        queue.append(root)
    while queue:
        parent = queue.popleft()
        for child in children_of[parent]:
            candidate = depth[parent] + 1
            if depth[child] < 0 or candidate < depth[child]:
                depth[child] = candidate
                queue.append(child)
    return [value if value >= 0 else 0 for value in depth]


def is_observed(dataset: str, time_value: float, observation: int) -> bool:
    return time_value <= float(observation)


def build_graph_features(
    dataset: str,
    sample: dict[str, Any],
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    communities: dict[str, dict[str, Any]],
    observation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    observed_events = [row for row in events if is_observed(dataset, row["time_value"], observation)]
    observed_edges = [row for row in edges if is_observed(dataset, row["time_value"], observation)]

    node_ids: dict[str, int] = {}
    source_id = sample["sample_id"]
    node_ids[source_id] = 0
    for row in observed_events:
        node_ids.setdefault(row["node_id"], len(node_ids))
    for edge in observed_edges:
        if edge["parent"]:
            node_ids.setdefault(edge["parent"], len(node_ids))
        node_ids.setdefault(edge["child"], len(node_ids))

    n_nodes = len(node_ids)
    in_degree = np.zeros(n_nodes, dtype=np.float32)
    out_degree = np.zeros(n_nodes, dtype=np.float32)
    delays = np.zeros(n_nodes, dtype=np.float32)
    children_of: list[list[int]] = [[] for _ in range(n_nodes)]
    edge_src: list[int] = []
    edge_dst: list[int] = []
    event_time_map = {row["node_id"]: row["time_value"] for row in observed_events}

    for node_id, node_index in node_ids.items():
        delays[node_index] = finite_float(event_time_map.get(node_id), 0.0)
    for edge in observed_edges:
        child_index = node_ids[edge["child"]]
        delays[child_index] = max(delays[child_index], finite_float(edge["time_value"], 0.0))
        if not edge["parent"]:
            continue
        parent_index = node_ids[edge["parent"]]
        out_degree[parent_index] += 1.0
        in_degree[child_index] += 1.0
        children_of[parent_index].append(child_index)
        edge_src.append(parent_index)
        edge_dst.append(child_index)

    roots = [index for index, degree in enumerate(in_degree) if degree == 0.0]
    source_index = node_ids.get(source_id, 0)
    if source_index not in roots:
        roots.insert(0, source_index)
    depth = compute_depth(children_of, roots)
    max_depth = max(depth) if depth else 0
    max_delay = float(delays.max()) if len(delays) else 0.0
    delay_norm = max(math.log1p(max_delay), 1.0)
    root_set = set(roots)
    node_features = np.zeros((n_nodes, len(NODE_FEATURE_NAMES)), dtype=np.float32)

    for node_id, index in node_ids.items():
        community = communities.get(node_id, {})
        node_features[index] = np.asarray(
            [
                math.log1p(float(in_degree[index])),
                math.log1p(float(out_degree[index])),
                math.log1p(float(in_degree[index] + out_degree[index])),
                1.0 if index in root_set else 0.0,
                depth[index] / max(max_depth, 1),
                math.log1p(float(delays[index])) / delay_norm,
                1.0 if int(community.get("community_id", 0)) > 0 else 0.0,
                1.0 if community.get("is_root_child", False) else 0.0,
            ],
            dtype=np.float32,
        )

    edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
    n_edges = len(edge_src)
    active_communities = {
        int(communities.get(node_id, {}).get("community_id", 0))
        for node_id in node_ids
        if int(communities.get(node_id, {}).get("community_id", 0)) > 0
    }
    cross_edges = 0
    for edge in observed_edges:
        if not edge["parent"]:
            continue
        parent_info = communities.get(edge["parent"], {})
        child_info = communities.get(edge["child"], {})
        parent_community = int(parent_info.get("community_id", 0))
        child_community = int(child_info.get("community_id", 0))
        if not parent_info.get("is_root", False) and parent_community != child_community:
            cross_edges += 1

    possible_edges = n_nodes * max(n_nodes - 1, 1)
    global_features = np.asarray(
        [
            math.log1p(n_nodes),
            math.log1p(n_edges),
            n_edges / possible_edges if n_nodes > 1 else 0.0,
            n_edges / max(n_nodes - 1, 1) if n_nodes > 1 else 0.0,
            float(max_depth),
            math.log1p(max_delay),
            math.log1p(len(active_communities)),
            cross_edges / n_edges if n_edges else 0.0,
        ],
        dtype=np.float32,
    )
    return node_features, edge_index, global_features, observed_events, observed_edges


def bin_index(dataset: str, time_value: float, window_size: int, num_steps: int) -> int:
    if dataset == "weibo":
        index = int(max(time_value - 1.0, 0.0) // window_size)
    else:
        index = int(max(time_value, 0.0) // window_size)
    return min(max(index, 0), num_steps - 1)


def build_temporal_features(
    dataset: str,
    observed_events: list[dict[str, Any]],
    observed_edges: list[dict[str, Any]],
    communities: dict[str, dict[str, Any]],
    observation: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = max(int(math.ceil(observation / window_size)), 1)
    new_nodes = np.zeros(num_steps, dtype=np.float32)
    new_edges = np.zeros(num_steps, dtype=np.float32)
    new_users: list[set[str]] = [set() for _ in range(num_steps)]
    new_communities: list[set[int]] = [set() for _ in range(num_steps)]
    cross_edges = np.zeros(num_steps, dtype=np.float32)

    for event in observed_events:
        index = bin_index(dataset, event["time_value"], window_size, num_steps)
        new_nodes[index] += 1.0
        user_id = event.get("user_id", "")
        if user_id and user_id != "ROOT":
            new_users[index].add(user_id)
        community_id = int(communities.get(event["node_id"], {}).get("community_id", 0))
        if community_id > 0:
            new_communities[index].add(community_id)

    for edge in observed_edges:
        index = bin_index(dataset, edge["time_value"], window_size, num_steps)
        new_edges[index] += 1.0
        if edge["parent"]:
            parent_info = communities.get(edge["parent"], {})
            child_info = communities.get(edge["child"], {})
            if (
                not parent_info.get("is_root", False)
                and int(parent_info.get("community_id", 0))
                != int(child_info.get("community_id", 0))
            ):
                cross_edges[index] += 1.0

    features = np.zeros((num_steps, len(TEMPORAL_FEATURE_NAMES)), dtype=np.float32)
    seen_users: set[str] = set()
    seen_communities: set[int] = set()
    cumulative_nodes = 0.0
    cumulative_edges = 0.0
    cumulative_cross = 0.0

    for index in range(num_steps):
        cumulative_nodes += float(new_nodes[index])
        cumulative_edges += float(new_edges[index])
        cumulative_cross += float(cross_edges[index])
        unseen_communities = new_communities[index] - seen_communities
        seen_users.update(new_users[index])
        seen_communities.update(new_communities[index])
        features[index] = np.asarray(
            [
                math.log1p(float(new_nodes[index])),
                math.log1p(cumulative_nodes),
                math.log1p(float(new_edges[index])),
                math.log1p(cumulative_edges),
                math.log1p(len(new_users[index])),
                math.log1p(len(seen_users)),
                math.log1p(len(unseen_communities)),
                math.log1p(len(seen_communities)),
                cumulative_cross / cumulative_edges if cumulative_edges else 0.0,
                len(seen_communities) / max(cumulative_nodes, 1.0),
            ],
            dtype=np.float32,
        )
    return features, np.ones(num_steps, dtype=np.float32)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_user_features(
    sample_id: str,
    observed_events: list[dict[str, Any]],
) -> tuple[np.ndarray, bool]:
    user_ids = {
        row.get("user_id", "")
        for row in observed_events
        if row.get("user_id", "") and row.get("user_id", "") != "ROOT"
    }
    profile_rows = [
        row
        for row in observed_events
        if any(row.get(field, "") != "" for field in ("followers_count", "friends_count", "statuses_count"))
    ]
    followers = [max(finite_float(row.get("followers_count"), 0.0), 0.0) for row in profile_rows]
    friends = [max(finite_float(row.get("friends_count"), 0.0), 0.0) for row in profile_rows]
    statuses = [max(finite_float(row.get("statuses_count"), 0.0), 0.0) for row in profile_rows]
    verified = [1.0 if truthy(row.get("verified")) else 0.0 for row in profile_rows]
    source_rows = [row for row in profile_rows if row.get("node_id") == sample_id]
    source_followers = max(finite_float(source_rows[0].get("followers_count"), 0.0), 0.0) if source_rows else 0.0
    observed_nodes = max(len(observed_events), 1)
    features = np.asarray(
        [
            math.log1p(float(np.mean(followers))) if followers else 0.0,
            math.log1p(max(followers)) if followers else 0.0,
            math.log1p(float(np.mean(friends))) if friends else 0.0,
            math.log1p(float(np.mean(statuses))) if statuses else 0.0,
            float(np.mean(verified)) if verified else 0.0,
            len(user_ids) / observed_nodes,
            len(profile_rows) / observed_nodes,
            math.log1p(source_followers),
        ],
        dtype=np.float32,
    )
    return features, bool(profile_rows)


def artifact_stem(dataset: str, observation: int) -> str:
    unit = "events" if dataset == "weibo" else "m"
    return f"obs_{observation}{unit}"


def full_cascade_size(
    sample: dict[str, Any],
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> int:
    node_ids = {sample["sample_id"]}
    node_ids.update(row["node_id"] for row in events if row.get("node_id"))
    for edge in edges:
        if edge.get("parent"):
            node_ids.add(edge["parent"])
        if edge.get("child"):
            node_ids.add(edge["child"])
    return max(int(sample.get("num_nodes") or 1), len(node_ids), 1)


def build_artifact(
    dataset: str,
    samples: list[dict[str, Any]],
    events_by_sample: dict[str, list[dict[str, Any]]],
    edges_by_sample: dict[str, list[dict[str, Any]]],
    communities_by_sample: dict[str, dict[str, dict[str, Any]]],
    observation: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_ids: list[str] = []
    raw_labels: list[str] = []
    source_texts: list[str] = []
    label_ids: list[int] = []
    final_sizes: list[float] = []
    observed_sizes: list[float] = []
    text_features: list[np.ndarray] = []
    global_features: list[np.ndarray] = []
    temporal_features: list[np.ndarray] = []
    temporal_masks: list[np.ndarray] = []
    user_features: list[np.ndarray] = []
    modality_masks: list[np.ndarray] = []
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    node_ptr = [0]
    edge_ptr = [0]

    for sample in samples:
        sample_id = sample["sample_id"]
        source_text = sample.get("source_text", "")
        communities = communities_by_sample.get(sample_id, {})
        node_features, edge_index, graph_global, observed_events, observed_edges = build_graph_features(
            dataset=dataset,
            sample=sample,
            events=events_by_sample.get(sample_id, []),
            edges=edges_by_sample.get(sample_id, []),
            communities=communities,
            observation=observation,
        )
        temporal, temporal_mask = build_temporal_features(
            dataset=dataset,
            observed_events=observed_events,
            observed_edges=observed_edges,
            communities=communities,
            observation=observation,
            window_size=args.order_window_size if dataset == "weibo" else args.window_minutes,
        )
        user_vector, has_user_profile = build_user_features(sample_id, observed_events)
        has_text = bool(source_text.strip())
        has_topology = edge_index.shape[1] > 0
        has_temporal = bool(observed_events or observed_edges)

        sample_ids.append(sample_id)
        raw_labels.append(sample.get("raw_label", ""))
        source_texts.append(source_text)
        label_ids.append(int(sample["label_id"]))
        final_sizes.append(
            float(
                full_cascade_size(
                    sample,
                    events_by_sample.get(sample_id, []),
                    edges_by_sample.get(sample_id, []),
                )
            )
        )
        observed_sizes.append(float(node_features.shape[0]))
        text_features.append(stable_hash_features(source_text, args.text_dim))
        global_features.append(graph_global)
        temporal_features.append(temporal)
        temporal_masks.append(temporal_mask)
        user_features.append(user_vector)
        modality_masks.append(
            np.asarray(
                [float(has_text), float(has_topology), float(has_temporal), float(has_user_profile)],
                dtype=np.float32,
            )
        )
        node_parts.append(node_features)
        edge_parts.append(edge_index)
        node_ptr.append(node_ptr[-1] + node_features.shape[0])
        edge_ptr.append(edge_ptr[-1] + edge_index.shape[1])

    arrays = {
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "raw_labels": np.asarray(raw_labels, dtype=np.str_),
        "source_texts": np.asarray(source_texts, dtype=np.str_),
        "label_ids": np.asarray(label_ids, dtype=np.int64),
        "final_sizes": np.asarray(final_sizes, dtype=np.float32),
        "log_final_sizes": np.log1p(np.asarray(final_sizes, dtype=np.float32)),
        "observed_sizes": np.asarray(observed_sizes, dtype=np.float32),
        "text_features": np.stack(text_features).astype(np.float32),
        "node_features": np.concatenate(node_parts, axis=0).astype(np.float32),
        "edge_index": np.concatenate(edge_parts, axis=1).astype(np.int64),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "global_features": np.stack(global_features).astype(np.float32),
        "temporal_features": np.stack(temporal_features).astype(np.float32),
        "temporal_masks": np.stack(temporal_masks).astype(np.float32),
        "user_features": np.stack(user_features).astype(np.float32),
        "modality_masks": np.stack(modality_masks).astype(np.float32),
    }

    output_dir = Path(args.output_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(dataset, observation)
    npz_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}_metadata.json"
    np.savez_compressed(npz_path, **arrays)

    modality_coverage = {
        name: float(arrays["modality_masks"][:, index].mean())
        for index, name in enumerate(MODALITY_NAMES)
    }
    metadata = {
        "version": 1,
        "dataset": dataset,
        "task": "cascade_size_prediction",
        "label_task": args.task,
        "observation": observation,
        "time_mode": "event_order" if dataset == "weibo" else "delay_minutes",
        "window_size": args.order_window_size if dataset == "weibo" else args.window_minutes,
        "num_time_steps": int(arrays["temporal_features"].shape[1]),
        "num_samples": len(samples),
        "num_nodes": int(arrays["node_features"].shape[0]),
        "num_edges": int(arrays["edge_index"].shape[1]),
        "text_dim": args.text_dim,
        "node_feature_names": NODE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "temporal_feature_names": TEMPORAL_FEATURE_NAMES,
        "user_feature_names": USER_FEATURE_NAMES,
        "modality_names": MODALITY_NAMES,
        "modality_coverage": modality_coverage,
        "target_names": ["final_sizes", "log_final_sizes"],
        "leakage_policy": {
            "text": "source post only; available at cascade start",
            "topology": "nodes and edges with time <= observation only",
            "temporal": "events and edges with time <= observation only",
            "user_profile": "profiles attached to observed events only",
            "target": "maximum unique full-cascade node count across samples.csv, events.csv, and edges.csv",
            "time_sanitization": "negative relative delays are clipped to zero",
        },
        "outputs": {"npz": str(npz_path), "metadata": str(metadata_path)},
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def process_dataset(dataset: str, observations: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_dir = Path(args.data_root) / dataset
    dataset_object = RumorDataset(
        dataset=dataset,
        data_root=args.data_root,
        label_map_path=args.label_map,
        task=args.task,
        filter_unlabeled=True,
        limit=args.limit or None,
    )
    samples = list(dataset_object)
    sample_ids = {sample["sample_id"] for sample in samples}
    events_by_sample, event_times = read_events(dataset, dataset_dir / "events.csv", sample_ids)
    edges_by_sample = read_edges(
        dataset=dataset,
        path=dataset_dir / "edges.csv",
        sample_ids=sample_ids,
        event_times=event_times,
        max_edges_per_sample=args.max_edges_per_sample if args.max_edges_per_sample > 0 else None,
    )
    communities_by_sample = read_communities(dataset_dir / "community_ids.csv", sample_ids)
    return [
        build_artifact(
            dataset=dataset,
            samples=samples,
            events_by_sample=events_by_sample,
            edges_by_sample=edges_by_sample,
            communities_by_sample=communities_by_sample,
            observation=observation,
            args=args,
        )
        for observation in observations
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="pheme,twitter15,twitter16")
    parser.add_argument("--observations", default="60,180,360")
    parser.add_argument("--task", default="rumor_binary", choices=["rumor_binary", "veracity", "raw"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--label-map", default="label_map.json")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument("--order-window-size", type=int, default=100)
    parser.add_argument("--max-edges-per-sample", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - SUPPORTED_DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    observations = parse_int_list(args.observations)
    outputs = []
    for dataset in datasets:
        outputs.extend(process_dataset(dataset, observations, args))
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
