import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("data/processed")


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


def read_samples(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "samples.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_event_times(dataset: str, dataset_dir: Path) -> dict[str, dict[str, float]]:
    if dataset == "weibo":
        return {}

    event_times: dict[str, dict[str, float]] = defaultdict(dict)
    with (dataset_dir / "events.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            node_id = row.get("tweet_id", "")
            if not sample_id or not node_id:
                continue
            event_times[sample_id][node_id] = finite_float(row.get("delay_minutes"), 0.0)
    return dict(event_times)


def edge_time(dataset: str, child: str, child_delay: str, event_times: dict[str, float]) -> float:
    if child_delay:
        return finite_float(child_delay, 0.0)
    if child in event_times:
        return finite_float(event_times[child], 0.0)
    if dataset == "weibo":
        return finite_float(child, 0.0)
    return 0.0


def read_edges_grouped(
    dataset: str,
    dataset_dir: Path,
    event_times_by_sample: dict[str, dict[str, float]],
    max_edges_per_sample: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (dataset_dir / "edges.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if not sample_id:
                continue
            if max_edges_per_sample is not None and len(grouped[sample_id]) >= max_edges_per_sample:
                continue
            parent = row.get("parent_tweet_id", "")
            child = row.get("child_tweet_id", "")
            if not child or child == "ROOT":
                continue
            if parent == "ROOT":
                parent = ""
            grouped[sample_id].append(
                {
                    "parent": parent,
                    "child": child,
                    "time": edge_time(
                        dataset,
                        child,
                        row.get("child_delay_minutes", ""),
                        event_times_by_sample.get(sample_id, {}),
                    ),
                }
            )
    return dict(grouped)


def infer_nodes(sample_id: str, edges: list[dict[str, Any]]) -> set[str]:
    nodes = {sample_id}
    for edge in edges:
        if edge["parent"]:
            nodes.add(edge["parent"])
        if edge["child"]:
            nodes.add(edge["child"])
    return nodes


def build_branch_communities(sample_id: str, edges: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    nodes = infer_nodes(sample_id, edges)
    children_of: dict[str, list[str]] = defaultdict(list)
    parents_of: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        parent = edge["parent"]
        child = edge["child"]
        if parent:
            children_of[parent].append(child)
            parents_of[child].append(parent)

    roots = {node for node in nodes if not parents_of.get(node)}
    if sample_id in nodes:
        roots.add(sample_id)
    if not roots and nodes:
        roots.add(sorted(nodes)[0])

    community: dict[str, int] = {}
    depth: dict[str, int] = {}
    parent_choice: dict[str, str] = {}
    is_root_child: dict[str, bool] = defaultdict(bool)

    queue: deque[tuple[str, int, int, str]] = deque()
    next_community_id = 1
    for root in sorted(roots):
        community[root] = 0
        depth[root] = 0
        parent_choice[root] = ""
        for child in sorted(children_of.get(root, [])):
            if child in community:
                continue
            community_id = next_community_id
            next_community_id += 1
            community[child] = community_id
            depth[child] = 1
            parent_choice[child] = root
            is_root_child[child] = True
            queue.append((child, community_id, 1, root))

    while queue:
        node, community_id, node_depth, _root = queue.popleft()
        for child in sorted(children_of.get(node, [])):
            if child in community:
                continue
            community[child] = community_id
            depth[child] = node_depth + 1
            parent_choice[child] = node
            queue.append((child, community_id, node_depth + 1, node))

    for node in nodes:
        if node not in community:
            community[node] = 0
            depth[node] = 0
            parent_choice[node] = ""

    rows = {
        node: {
            "node_id": node,
            "community_id": community[node],
            "depth": depth[node],
            "parent_node_id": parent_choice.get(node, ""),
            "is_root": node in roots,
            "is_root_child": bool(is_root_child[node]),
        }
        for node in nodes
    }
    return rows, roots


def window_index(dataset: str, time_value: float, window_minutes: float, order_window_size: int) -> int:
    if dataset == "weibo":
        return max(int((time_value - 1) // max(order_window_size, 1)), 0)
    return max(int(time_value // max(window_minutes, 1.0)), 0)


def build_snapshots(
    dataset: str,
    sample_id: str,
    edges: list[dict[str, Any]],
    community_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    node_first_time: dict[str, float] = {}
    for node_id in community_rows:
        node_first_time[node_id] = 0.0 if community_rows[node_id]["is_root"] else math.inf
    for edge in edges:
        child = edge["child"]
        node_first_time[child] = min(node_first_time.get(child, math.inf), finite_float(edge["time"], 0.0))
        if edge["parent"] and edge["parent"] not in node_first_time:
            node_first_time[edge["parent"]] = 0.0

    edge_records = []
    for edge in edges:
        parent = edge["parent"]
        child = edge["child"]
        parent_comm = community_rows.get(parent, {}).get("community_id", 0)
        child_comm = community_rows.get(child, {}).get("community_id", 0)
        parent_is_root = bool(community_rows.get(parent, {}).get("is_root", False))
        cross = bool(parent and parent_comm != child_comm and not parent_is_root)
        edge_records.append(
            {
                "time": finite_float(edge["time"], 0.0),
                "parent": parent,
                "child": child,
                "cross": cross,
            }
        )

    nodes_by_window: dict[int, set[str]] = defaultdict(set)
    for node, time_value in node_first_time.items():
        idx = window_index(
            dataset,
            time_value if math.isfinite(time_value) else 0.0,
            args.window_minutes,
            args.order_window_size,
        )
        nodes_by_window[idx].add(node)

    edges_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edge_records:
        idx = window_index(dataset, edge["time"], args.window_minutes, args.order_window_size)
        edges_by_window[idx].append(edge)

    active_windows = sorted({0, *nodes_by_window.keys(), *edges_by_window.keys()})

    rows = []
    seen_nodes: set[str] = set()
    seen_edges = 0
    seen_cross_edges = 0
    seen_communities: set[int] = set()
    time_mode = "event_order" if dataset == "weibo" else "delay_minutes"

    for idx in active_windows:
        current_nodes = nodes_by_window.get(idx, set())
        current_edges = edges_by_window.get(idx, [])
        new_communities = {
            int(community_rows[node]["community_id"])
            for node in current_nodes
            if int(community_rows[node]["community_id"]) > 0
        } - seen_communities

        seen_nodes.update(current_nodes)
        seen_edges += len(current_edges)
        seen_cross_edges += sum(1 for edge in current_edges if edge["cross"])
        seen_communities.update(
            int(community_rows[node]["community_id"])
            for node in seen_nodes
            if int(community_rows[node]["community_id"]) > 0
        )

        rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "time_mode": time_mode,
                "window_index": idx,
                "window_start": idx * (args.order_window_size if dataset == "weibo" else args.window_minutes),
                "window_end": (idx + 1) * (args.order_window_size if dataset == "weibo" else args.window_minutes),
                "new_nodes": len(current_nodes),
                "cumulative_nodes": len(seen_nodes),
                "new_edges": len(current_edges),
                "cumulative_edges": seen_edges,
                "new_communities": len(new_communities),
                "active_communities": len(seen_communities),
                "new_cross_edges": sum(1 for edge in current_edges if edge["cross"]),
                "cumulative_cross_edges": seen_cross_edges,
                "cross_edge_ratio": seen_cross_edges / seen_edges if seen_edges else 0.0,
                "branch_community_ratio": len(seen_communities) / max(len(seen_nodes), 1),
            }
        )
    return rows


def detect_breakout(snapshot_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    for row in snapshot_rows:
        if int(row["window_index"]) < args.min_breakout_window:
            continue
        if int(row["cumulative_nodes"]) < args.min_nodes:
            continue
        if int(row["active_communities"]) < args.min_active_communities:
            continue
        cross_trigger = float(row["cross_edge_ratio"]) >= args.theta_cross and int(row["cumulative_cross_edges"]) > 0
        branch_trigger = int(row["new_communities"]) >= args.min_new_communities and int(row["window_index"]) > 0
        broad_trigger = float(row["branch_community_ratio"]) >= args.theta_branch_ratio
        if cross_trigger or branch_trigger or broad_trigger:
            reason = []
            if cross_trigger:
                reason.append("cross_edge_ratio")
            if branch_trigger:
                reason.append("new_communities")
            if broad_trigger:
                reason.append("branch_community_ratio")
            return {
                "has_breakout": 1,
                "breakout_window": row["window_index"],
                "breakout_time": row["window_start"],
                "trigger_reason": "+".join(reason),
                "cumulative_nodes": row["cumulative_nodes"],
                "active_communities": row["active_communities"],
                "cross_edge_ratio": row["cross_edge_ratio"],
                "new_communities": row["new_communities"],
            }

    last = snapshot_rows[-1] if snapshot_rows else {}
    return {
        "has_breakout": 0,
        "breakout_window": "",
        "breakout_time": "",
        "trigger_reason": "not_triggered",
        "cumulative_nodes": last.get("cumulative_nodes", 0),
        "active_communities": last.get("active_communities", 0),
        "cross_edge_ratio": last.get("cross_edge_ratio", 0.0),
        "new_communities": 0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def open_csv_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def write_row(writer: csv.DictWriter, row: dict[str, Any], fieldnames: list[str]) -> None:
    writer.writerow({field: row.get(field, "") for field in fieldnames})


def process_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.data_root) / dataset
    samples = read_samples(dataset_dir)
    sample_ids = {row["sample_id"] for row in samples}
    event_times = read_event_times(dataset, dataset_dir)
    edges_by_sample = read_edges_grouped(dataset, dataset_dir, event_times, args.max_edges_per_sample)
    snapshot_dir = dataset_dir / "dynamic_snapshots"

    community_fields = [
        "dataset",
        "sample_id",
        "node_id",
        "community_id",
        "depth",
        "parent_node_id",
        "is_root",
        "is_root_child",
    ]
    snapshot_fields = [
        "dataset",
        "sample_id",
        "time_mode",
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
    ]
    breakout_fields = [
        "dataset",
        "sample_id",
        "label",
        "has_breakout",
        "breakout_window",
        "breakout_time",
        "trigger_reason",
        "cumulative_nodes",
        "active_communities",
        "cross_edge_ratio",
        "new_communities",
    ]

    num_community_rows = 0
    num_snapshot_rows = 0
    num_breakout_samples = 0

    community_file, community_writer = open_csv_writer(dataset_dir / "community_ids.csv", community_fields)
    snapshot_file, snapshot_writer = open_csv_writer(snapshot_dir / "snapshots.csv", snapshot_fields)
    breakout_file, breakout_writer = open_csv_writer(dataset_dir / "breakout_events.csv", breakout_fields)
    try:
        for sample in samples:
            sample_id = sample["sample_id"]
            edges = edges_by_sample.get(sample_id, [])
            communities, _roots = build_branch_communities(sample_id, edges)
            for row in communities.values():
                write_row(community_writer, {"dataset": dataset, "sample_id": sample_id, **row}, community_fields)
                num_community_rows += 1

            snapshots = build_snapshots(dataset, sample_id, edges, communities, args)
            for row in snapshots:
                write_row(snapshot_writer, row, snapshot_fields)
                num_snapshot_rows += 1

            breakout = detect_breakout(snapshots, args)
            num_breakout_samples += int(breakout["has_breakout"])
            write_row(
                breakout_writer,
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "label": sample.get("label", ""),
                    **breakout,
                },
                breakout_fields,
            )
    finally:
        community_file.close()
        snapshot_file.close()
        breakout_file.close()

    stats = {
        "dataset": dataset,
        "num_samples": len(sample_ids),
        "num_samples_with_edges": sum(1 for sample_id in sample_ids if edges_by_sample.get(sample_id)),
        "num_community_rows": num_community_rows,
        "num_snapshot_rows": num_snapshot_rows,
        "num_breakout_samples": num_breakout_samples,
        "breakout_rate": num_breakout_samples / max(len(samples), 1),
        "community_method": "propagation_branch_heuristic",
        "time_mode": "event_order" if dataset == "weibo" else "delay_minutes",
        "window_minutes": args.window_minutes,
        "order_window_size": args.order_window_size,
        "max_edges_per_sample": args.max_edges_per_sample,
        "min_breakout_window": args.min_breakout_window,
        "min_nodes": args.min_nodes,
        "min_active_communities": args.min_active_communities,
        "theta_cross": args.theta_cross,
        "theta_branch_ratio": args.theta_branch_ratio,
    }
    (dataset_dir / "c2_foundation_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="pheme,twitter15,twitter16,weibo")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--window-minutes", type=float, default=60.0)
    parser.add_argument("--order-window-size", type=int, default=100)
    parser.add_argument("--max-edges-per-sample", type=int)
    parser.add_argument("--min-breakout-window", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=10)
    parser.add_argument("--min-active-communities", type=int, default=3)
    parser.add_argument("--min-new-communities", type=int, default=1)
    parser.add_argument("--theta-cross", type=float, default=0.2)
    parser.add_argument("--theta-branch-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    results = []
    for dataset in [part.strip().lower() for part in args.datasets.split(",") if part.strip()]:
        results.append(process_dataset(dataset, args))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
