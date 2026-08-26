import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dataset_loader import default_split_path, load_split_ids


DEFAULT_INPUT_ROOT = Path("data/processed/v1_inputs")
SUPPORTED_DATASETS = [
    "pheme",
    "twitter15",
    "twitter16",
    "twitter15_rumdetect2017",
    "twitter16_rumdetect2017",
    "weibo",
]


def artifact_stem(dataset: str, observation: int) -> str:
    unit = "events" if dataset == "weibo" else "m"
    return f"obs_{observation}{unit}"


class V1InputDataset:
    """Loads leakage-safe text, topology, temporal, and user inputs for V1."""

    def __init__(
        self,
        dataset: str,
        observation: int,
        split: str,
        split_strategy: str = "stratified",
        seed: int = 42,
        task: str = "rumor_binary",
        input_root: Path | str = DEFAULT_INPUT_ROOT,
        split_root: Path | str = "data/processed/splits",
        split_file: Path | str | None = None,
        text_feature_path: Path | str | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.observation = int(observation)
        self.split = split
        self.split_strategy = split_strategy
        self.task = task
        stem = artifact_stem(self.dataset, self.observation)
        dataset_dir = Path(input_root) / self.dataset
        self.npz_path = dataset_dir / f"{stem}.npz"
        self.metadata_path = dataset_dir / f"{stem}_metadata.json"
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"Missing V1 artifact: {self.npz_path}. Run scripts/build_v1_inputs.py first."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing V1 metadata: {self.metadata_path}")

        archive = np.load(self.npz_path, allow_pickle=False)
        try:
            self.arrays = {name: archive[name] for name in archive.files}
        finally:
            archive.close()
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.text_feature_path = Path(text_feature_path) if text_feature_path else None
        if self.text_feature_path is not None:
            text_archive = np.load(self.text_feature_path, allow_pickle=False)
            try:
                text_ids = [str(value) for value in text_archive["sample_ids"]]
                text_features = text_archive["text_features"].astype(np.float32)
            finally:
                text_archive.close()
            if text_features.ndim != 2 or len(text_ids) != text_features.shape[0]:
                raise ValueError(f"Invalid PLM text feature archive: {self.text_feature_path}")
            text_by_id = {sample_id: text_features[index] for index, sample_id in enumerate(text_ids)}
            artifact_ids = [str(value) for value in self.arrays["sample_ids"]]
            missing_text = [sample_id for sample_id in artifact_ids if sample_id not in text_by_id]
            if missing_text:
                raise ValueError(
                    f"{len(missing_text)} samples are missing PLM text features; first: {missing_text[0]}"
                )
            self.arrays["text_features"] = np.stack(
                [text_by_id[sample_id] for sample_id in artifact_ids]
            ).astype(np.float32)

        resolved_split_file = (
            Path(split_file)
            if split_file is not None
            else default_split_path(
                dataset=self.dataset,
                task=self.task,
                split_strategy=self.split_strategy,
                split_root=Path(split_root),
                seed=seed,
            )
        )
        split_ids, self.split_metadata = load_split_ids(
            split_file=resolved_split_file,
            split=split,
            expected_dataset=self.dataset,
            expected_task=self.task,
        )
        artifact_ids = [str(value) for value in self.arrays["sample_ids"]]
        index_by_id = {sample_id: index for index, sample_id in enumerate(artifact_ids)}
        missing = [sample_id for sample_id in split_ids if sample_id not in index_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} split samples are missing from {self.npz_path}; first: {missing[0]}"
            )
        self.indices = np.asarray([index_by_id[sample_id] for sample_id in split_ids], dtype=np.int64)
        self.split_file = resolved_split_file

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        node_start = int(self.arrays["node_ptr"][index])
        node_end = int(self.arrays["node_ptr"][index + 1])
        edge_start = int(self.arrays["edge_ptr"][index])
        edge_end = int(self.arrays["edge_ptr"][index + 1])
        return {
            "dataset": self.dataset,
            "sample_id": str(self.arrays["sample_ids"][index]),
            "raw_label": str(self.arrays["raw_labels"][index]),
            "label_id": int(self.arrays["label_ids"][index]),
            "source_text": str(self.arrays["source_texts"][index]),
            "text_features": self.arrays["text_features"][index].copy(),
            "node_features": self.arrays["node_features"][node_start:node_end].copy(),
            "edge_index": self.arrays["edge_index"][:, edge_start:edge_end].copy(),
            "global_features": self.arrays["global_features"][index].copy(),
            "temporal_features": self.arrays["temporal_features"][index].copy(),
            "temporal_mask": self.arrays["temporal_masks"][index].copy(),
            "user_features": self.arrays["user_features"][index].copy(),
            "modality_mask": self.arrays["modality_masks"][index].copy(),
            "final_size": float(self.arrays["final_sizes"][index]),
            "log_final_size": float(self.arrays["log_final_sizes"][index]),
            "observed_size": float(self.arrays["observed_sizes"][index]),
        }

    def summary(self) -> dict[str, Any]:
        masks = self.arrays["modality_masks"][self.indices]
        modality_names = self.metadata["modality_names"]
        return {
            "dataset": self.dataset,
            "observation": self.observation,
            "split": self.split,
            "split_strategy": self.split_strategy,
            "split_file": str(self.split_file),
            "num_samples": len(self),
            "text_dim": int(self.arrays["text_features"].shape[1]),
            "text_feature_source": (
                str(self.text_feature_path) if self.text_feature_path is not None else "stable_hash"
            ),
            "node_feature_dim": int(self.arrays["node_features"].shape[1]),
            "global_feature_dim": int(self.arrays["global_features"].shape[1]),
            "temporal_shape": list(self.arrays["temporal_features"].shape[1:]),
            "user_feature_dim": int(self.arrays["user_features"].shape[1]),
            "modality_coverage": {
                name: float(masks[:, index].mean())
                for index, name in enumerate(modality_names)
            },
        }


def collate_v1_batch(examples: list[dict[str, Any]], as_torch: bool = False) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty V1 batch")

    node_parts = []
    edge_parts = []
    graph_ids = []
    node_offset = 0
    for graph_index, example in enumerate(examples):
        nodes = example["node_features"]
        edges = example["edge_index"]
        node_parts.append(nodes)
        graph_ids.append(np.full(nodes.shape[0], graph_index, dtype=np.int64))
        if edges.shape[1] > 0:
            edge_parts.append(edges + node_offset)
        node_offset += nodes.shape[0]

    batch = {
        "sample_ids": [example["sample_id"] for example in examples],
        "source_texts": [example["source_text"] for example in examples],
        "text_features": np.stack([example["text_features"] for example in examples]),
        "node_features": np.concatenate(node_parts, axis=0),
        "edge_index": (
            np.concatenate(edge_parts, axis=1)
            if edge_parts
            else np.empty((2, 0), dtype=np.int64)
        ),
        "graph_id": np.concatenate(graph_ids, axis=0),
        "global_features": np.stack([example["global_features"] for example in examples]),
        "temporal_features": np.stack([example["temporal_features"] for example in examples]),
        "temporal_mask": np.stack([example["temporal_mask"] for example in examples]),
        "user_features": np.stack([example["user_features"] for example in examples]),
        "modality_mask": np.stack([example["modality_mask"] for example in examples]),
        "label_ids": np.asarray([example["label_id"] for example in examples], dtype=np.int64),
        "final_sizes": np.asarray([example["final_size"] for example in examples], dtype=np.float32),
        "log_final_sizes": np.asarray(
            [example["log_final_size"] for example in examples], dtype=np.float32
        ),
        "observed_sizes": np.asarray(
            [example["observed_size"] for example in examples], dtype=np.float32
        ),
    }
    if not as_torch:
        return batch

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required when collate_v1_batch(as_torch=True)") from exc

    torch_batch = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            torch_batch[key] = torch.from_numpy(value)
        else:
            torch_batch[key] = value
    return torch_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--observation", type=int, default=180)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="rumor_binary")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--split-root", default="data/processed/splits")
    parser.add_argument("--split-file")
    parser.add_argument("--text-feature-path")
    parser.add_argument("--limit", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    dataset = V1InputDataset(
        dataset=args.dataset,
        observation=args.observation,
        split=args.split,
        split_strategy=args.split_strategy,
        seed=args.seed,
        task=args.task,
        input_root=args.input_root,
        split_root=args.split_root,
        split_file=args.split_file,
        text_feature_path=args.text_feature_path,
    )
    examples = []
    for index in range(min(args.limit, len(dataset))):
        item = dataset[index]
        examples.append(
            {
                "sample_id": item["sample_id"],
                "label_id": item["label_id"],
                "source_text_preview": item["source_text"][:120],
                "text_shape": list(item["text_features"].shape),
                "node_shape": list(item["node_features"].shape),
                "edge_shape": list(item["edge_index"].shape),
                "temporal_shape": list(item["temporal_features"].shape),
                "modality_mask": item["modality_mask"].tolist(),
                "observed_size": item["observed_size"],
                "final_size": item["final_size"],
            }
        )
    print(json.dumps({"summary": dataset.summary(), "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
