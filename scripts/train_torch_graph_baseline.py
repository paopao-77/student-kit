import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from dataset_loader import RumorDataset


DEFAULT_RESULTS_DIR = Path("results/graph_baseline")
NODE_FEATURE_DIM = 6
GLOBAL_FEATURE_DIM = 6


@dataclass
class GraphExample:
    dataset: str
    sample_id: str
    raw_label: str
    label: int
    x: torch.Tensor
    td_src: torch.Tensor
    td_dst: torch.Tensor
    bu_src: torch.Tensor
    bu_dst: torch.Tensor
    global_features: torch.Tensor


@dataclass
class GraphBatch:
    x: torch.Tensor
    td_src: torch.Tensor
    td_dst: torch.Tensor
    bu_src: torch.Tensor
    bu_dst: torch.Tensor
    graph_id: torch.Tensor
    global_features: torch.Tensor
    labels: torch.Tensor


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
            edges_by_sample[sample_id].append((parent, child, finite_float(row.get("child_delay_minutes"), 0.0)))
    return dict(edges_by_sample)


def build_graph_example(sample: dict[str, Any], edges: list[tuple[str, str, float]]) -> GraphExample:
    node_ids: dict[str, int] = {}
    for parent, child, _delay in edges:
        if parent and parent not in node_ids:
            node_ids[parent] = len(node_ids)
        if child and child not in node_ids:
            node_ids[child] = len(node_ids)
    if not node_ids:
        node_ids[sample["sample_id"]] = 0

    n_nodes = len(node_ids)
    in_degree = [0.0] * n_nodes
    out_degree = [0.0] * n_nodes
    delay = [0.0] * n_nodes
    children_of: list[list[int]] = [[] for _ in range(n_nodes)]
    td_src: list[int] = []
    td_dst: list[int] = []

    for parent, child, child_delay in edges:
        child_idx = node_ids.get(child)
        if child_idx is None:
            continue
        delay[child_idx] = max(delay[child_idx], child_delay)
        if not parent:
            continue
        parent_idx = node_ids.get(parent)
        if parent_idx is None:
            continue
        out_degree[parent_idx] += 1.0
        in_degree[child_idx] += 1.0
        children_of[parent_idx].append(child_idx)
        td_src.append(parent_idx)
        td_dst.append(child_idx)

    roots = [idx for idx, value in enumerate(in_degree) if value == 0.0]
    depth = compute_depth(children_of, roots)
    max_depth = max(depth) if depth else 0
    max_delay = max(delay) if delay else finite_float(sample.get("max_delay_minutes"), 0.0)
    if max_delay <= 0.0:
        max_delay = finite_float(sample.get("max_delay_minutes"), 0.0)

    node_features = []
    root_set = set(roots)
    delay_norm = max(math.log1p(max_delay), 1.0)
    for idx in range(n_nodes):
        node_features.append(
            [
                math.log1p(in_degree[idx]),
                math.log1p(out_degree[idx]),
                math.log1p(in_degree[idx] + out_degree[idx]),
                1.0 if idx in root_set else 0.0,
                depth[idx] / max(max_depth, 1),
                math.log1p(delay[idx]) / delay_norm,
            ]
        )

    n_edges = len(td_src)
    possible_edges = float(n_nodes) * max(float(n_nodes - 1), 1.0)
    global_features = [
        math.log1p(n_nodes),
        math.log1p(max(n_edges, finite_float(sample.get("num_edges"), 0.0))),
        n_edges / possible_edges if n_nodes > 1 else 0.0,
        n_edges / max(float(n_nodes - 1), 1.0) if n_nodes > 1 else 0.0,
        float(max_depth),
        math.log1p(max_delay),
    ]

    return GraphExample(
        dataset=sample["dataset"],
        sample_id=sample["sample_id"],
        raw_label=sample["raw_label"],
        label=int(sample["label_id"]),
        x=torch.tensor(node_features, dtype=torch.float32),
        td_src=torch.tensor(td_src, dtype=torch.long),
        td_dst=torch.tensor(td_dst, dtype=torch.long),
        bu_src=torch.tensor(td_dst, dtype=torch.long),
        bu_dst=torch.tensor(td_src, dtype=torch.long),
        global_features=torch.tensor(global_features, dtype=torch.float32),
    )


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


class DirectionalGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.message_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if src.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])
            degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device))
            agg = agg / degree.clamp_min(1.0)
        return torch.relu(self.norm(self.self_linear(x) + self.message_linear(agg)))


class TorchBiGCNClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.td_layers = nn.ModuleList(
            DirectionalGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.bu_layers = nn.ModuleList(
            DirectionalGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        graph_dim = hidden_dim * 6 + GLOBAL_FEATURE_DIM
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = torch.relu(self.input_proj(batch.x))

        td = x
        bu = x
        for td_layer, bu_layer in zip(self.td_layers, self.bu_layers):
            td = td_layer(td, batch.td_src, batch.td_dst)
            bu = bu_layer(bu, batch.bu_src, batch.bu_dst)

        graph_repr = torch.cat(
            [
                pool_batch(td, batch.graph_id, batch.global_features.shape[0]),
                pool_batch(bu, batch.graph_id, batch.global_features.shape[0]),
                batch.global_features,
            ],
            dim=1,
        )
        return self.classifier(graph_repr)


def pool_batch(x: torch.Tensor, graph_id: torch.Tensor, num_graphs: int) -> torch.Tensor:
    pooled = []
    for idx in range(num_graphs):
        mask = graph_id == idx
        graph_x = x[mask]
        if graph_x.shape[0] == 0:
            zeros = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
            pooled.append(torch.cat([zeros, zeros, zeros], dim=0))
            continue
        std = torch.zeros_like(graph_x[0]) if graph_x.shape[0] == 1 else graph_x.std(dim=0, unbiased=False)
        pooled.append(torch.cat([graph_x.mean(dim=0), graph_x.max(dim=0).values, std], dim=0))
    return torch.stack(pooled, dim=0)


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


def build_graphs(args: argparse.Namespace) -> tuple[list[GraphExample], list[GraphExample], list[GraphExample]]:
    split_datasets = {
        "train": load_split_dataset(args, "train"),
        "val": load_split_dataset(args, "val"),
        "test": load_split_dataset(args, "test"),
    }
    sample_ids = {sample["sample_id"] for dataset in split_datasets.values() for sample in dataset}
    edges = read_edges_for_samples(
        Path(args.data_root) / args.dataset / "edges.csv",
        sample_ids,
        max_edges_per_graph=args.max_edges_per_graph,
    )

    graphs = {}
    for split_name, dataset in split_datasets.items():
        graphs[split_name] = [
            build_graph_example(sample, edges.get(sample["sample_id"], []))
            for sample in dataset
        ]
    return graphs["train"], graphs["val"], graphs["test"]


def iter_batches(graphs: list[GraphExample], batch_size: int, shuffle: bool, rng: random.Random) -> list[list[GraphExample]]:
    indices = list(range(len(graphs)))
    if shuffle:
        rng.shuffle(indices)
    return [list(graphs[idx] for idx in indices[start : start + batch_size]) for start in range(0, len(indices), batch_size)]


def labels_for_batch(batch: list[GraphExample], device: torch.device) -> torch.Tensor:
    return torch.tensor([graph.label for graph in batch], dtype=torch.long, device=device)


def collate_graphs(graphs: list[GraphExample], device: torch.device) -> GraphBatch:
    x_parts = []
    graph_id_parts = []
    global_parts = []
    labels = []
    td_src = []
    td_dst = []
    bu_src = []
    bu_dst = []
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        x = graph.x
        x_parts.append(x)
        graph_id_parts.append(torch.full((x.shape[0],), graph_idx, dtype=torch.long))
        global_parts.append(graph.global_features)
        labels.append(graph.label)
        if graph.td_src.numel() > 0:
            td_src.append(graph.td_src + node_offset)
            td_dst.append(graph.td_dst + node_offset)
            bu_src.append(graph.bu_src + node_offset)
            bu_dst.append(graph.bu_dst + node_offset)
        node_offset += x.shape[0]

    empty_edges = torch.empty((0,), dtype=torch.long)
    return GraphBatch(
        x=torch.cat(x_parts, dim=0).to(device),
        td_src=(torch.cat(td_src, dim=0) if td_src else empty_edges).to(device),
        td_dst=(torch.cat(td_dst, dim=0) if td_dst else empty_edges).to(device),
        bu_src=(torch.cat(bu_src, dim=0) if bu_src else empty_edges).to(device),
        bu_dst=(torch.cat(bu_dst, dim=0) if bu_dst else empty_edges).to(device),
        graph_id=torch.cat(graph_id_parts, dim=0).to(device),
        global_features=torch.stack(global_parts, dim=0).to(device),
        labels=torch.tensor(labels, dtype=torch.long, device=device),
    )


def train_one_epoch(
    model: TorchBiGCNClassifier,
    graphs: list[GraphExample],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in iter_batches(graphs, args.batch_size, True, rng):
        optimizer.zero_grad()
        graph_batch = collate_graphs(batch, device)
        logits = model(graph_batch)
        loss = loss_fn(logits, graph_batch.labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        total_examples += len(batch)
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def predict(model: TorchBiGCNClassifier, graphs: list[GraphExample], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    rows = []
    for batch in iter_batches(graphs, args.batch_size, False, random.Random(args.seed)):
        graph_batch = collate_graphs(batch, device)
        logits = model(graph_batch)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        for graph, pred, score in zip(batch, preds.cpu().tolist(), probs[:, 1].cpu().tolist()):
            y_true.append(graph.label)
            y_pred.append(int(pred))
            y_score.append(float(score))
            rows.append(
                {
                    "dataset": graph.dataset,
                    "split_strategy": args.split_strategy,
                    "model": "torch_bigcn",
                    "sample_id": graph.sample_id,
                    "raw_label": graph.raw_label,
                    "label_id": graph.label,
                    "pred_label_id": int(pred),
                    "score_label_1": float(score),
                }
            )
    return {"metrics": compute_metrics(y_true, y_pred, y_score), "predictions": rows}


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    precision_0 = tn / max(tn + fn, 1)
    recall_0 = tn / max(tn + fp, 1)
    f1_0 = 2 * precision_0 * recall_0 / max(precision_0 + recall_0, 1e-12)
    macro_f1 = (f1 + f1_0) / 2
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "auc": binary_auc(y_true, y_score),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def binary_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for y in y_true if y == 1)
    negatives = sum(1 for y in y_true if y == 0)
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2
        rank_sum += avg_rank * sum(1 for _score, label in pairs[idx:end] if label == 1)
        rank += end - idx
        idx = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def class_weights(graphs: list[GraphExample], device: torch.device) -> torch.Tensor:
    counts = [0, 0]
    for graph in graphs:
        counts[graph.label] += 1
    total = max(sum(counts), 1)
    weights = [total / max(2 * counts[idx], 1) for idx in range(2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_graphs, val_graphs, test_graphs = build_graphs(args)

    model = TorchBiGCNClassifier(args.hidden_dim, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_graphs, device))
    rng = random.Random(args.seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val_macro_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_graphs, optimizer, loss_fn, args, device, rng)
        val_metrics = predict(model, val_graphs, args, device)["metrics"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
        if float(val_metrics["macro_f1"]) > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    split_predictions = {
        "train": predict(model, train_graphs, args, device),
        "val": predict(model, val_graphs, args, device),
        "test": predict(model, test_graphs, args, device),
    }

    result = {
        "dataset": args.dataset,
        "task": args.task,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "model_type": "torch_bigcn",
        "device": str(device),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "max_edges_per_graph": args.max_edges_per_graph,
        "num_graphs": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        },
        "models": {
            "torch_bigcn": {
                split: payload["metrics"] for split, payload in split_predictions.items()
            }
        },
        "history": history,
    }

    edge_suffix = f"_maxe{args.max_edges_per_graph}" if args.max_edges_per_graph is not None else ""
    prefix = f"{args.dataset}_{args.task}_{args.split_strategy}_torch_bigcn_l{args.num_layers}{edge_suffix}_seed{args.seed}"
    metrics_path = Path(args.output_dir) / f"{prefix}_metrics.json"
    predictions_path = Path(args.output_dir) / f"{prefix}_predictions.csv"
    write_json(metrics_path, result)
    write_predictions(predictions_path, split_predictions["test"]["predictions"])
    result["outputs"] = {"metrics": str(metrics_path), "predictions": str(predictions_path)}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["weibo", "twitter15", "twitter16", "pheme"])
    parser.add_argument("--task", default="rumor_binary", choices=["rumor_binary"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--label-map", default="label_map.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--max-edges-per-graph", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
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
        "model_type": result["model_type"],
        "best_epoch": result["best_epoch"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "outputs": result["outputs"],
        "test_metrics": result["models"]["torch_bigcn"]["test"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
