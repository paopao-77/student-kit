import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_LABEL_MAP = Path("label_map.json")
DEFAULT_SPLIT_ROOT = DEFAULT_DATA_ROOT / "splits"


def _to_int(value: str, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float | None = None) -> float | None:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group_csv_by_sample(
    path: Path,
    max_rows_per_sample: int | None = None,
    allowed_sample_ids: set[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                continue
            if max_rows_per_sample is not None and len(grouped[sample_id]) >= max_rows_per_sample:
                continue
            grouped[sample_id].append(row)
    return dict(grouped)


def load_label_map(path: Path = DEFAULT_LABEL_MAP) -> dict[str, Any]:
    return _read_json(path)


def print_json(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    print(text)


def default_split_path(
    dataset: str,
    task: str,
    split_strategy: str,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    seed: int = 42,
) -> Path:
    dataset = dataset.lower()
    if split_strategy == "stratified":
        suffix = f"stratified_seed{seed}"
    elif split_strategy == "temporal":
        suffix = f"seed{seed}" if dataset == "weibo" else "time"
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    return split_root / f"{dataset}_{task}_{suffix}_split.json"


def load_split_ids(
    split_file: Path,
    split: str,
    expected_dataset: str,
    expected_task: str,
) -> tuple[list[str], dict[str, Any]]:
    split_data = _read_json(split_file)
    if split_data.get("dataset") != expected_dataset:
        raise ValueError(
            f"Split file dataset mismatch: expected {expected_dataset}, got {split_data.get('dataset')}"
        )
    if split_data.get("task") != expected_task:
        raise ValueError(f"Split file task mismatch: expected {expected_task}, got {split_data.get('task')}")
    if split not in split_data.get("splits", {}):
        raise ValueError(f"Unknown split {split!r}; available splits: {sorted(split_data.get('splits', {}))}")
    return list(split_data["splits"][split]), split_data


class RumorDataset:
    """Unified CSV loader for Weibo/BiGCN, Twitter15/16, and PHEME.

    The loader intentionally returns plain Python dictionaries so V0/V1 code can
    plug it into PyTorch, PyG, or baseline scripts without inheriting framework
    constraints too early.
    """

    def __init__(
        self,
        dataset: str,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        label_map_path: Path | str = DEFAULT_LABEL_MAP,
        task: str = "rumor_binary",
        include_edges: bool = False,
        include_events: bool = False,
        max_edges_per_sample: int | None = None,
        max_events_per_sample: int | None = None,
        filter_unlabeled: bool = True,
        limit: int | None = None,
        split: str | None = None,
        split_file: Path | str | None = None,
        split_root: Path | str = DEFAULT_SPLIT_ROOT,
        split_strategy: str = "stratified",
        split_seed: int = 42,
    ) -> None:
        self.dataset = dataset.lower()
        self.data_root = Path(data_root)
        self.dataset_dir = self.data_root / self.dataset
        self.task = task
        self.label_map = load_label_map(Path(label_map_path))
        self.split = split
        self.split_file: Path | None = None
        self.split_metadata: dict[str, Any] | None = None

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        sample_path = self.dataset_dir / "samples.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"samples.csv not found: {sample_path}")

        rows = _read_csv(sample_path)
        split_ids: list[str] | None = None
        if split is not None:
            self.split_file = (
                Path(split_file)
                if split_file is not None
                else default_split_path(
                    dataset=self.dataset,
                    task=self.task,
                    split_strategy=split_strategy,
                    split_root=Path(split_root),
                    seed=split_seed,
                )
            )
            if not self.split_file.exists():
                raise FileNotFoundError(f"Split file not found: {self.split_file}")
            split_ids, self.split_metadata = load_split_ids(
                split_file=self.split_file,
                split=split,
                expected_dataset=self.dataset,
                expected_task=self.task,
            )

        self.samples = []
        rows_by_id = {row.get("sample_id", ""): row for row in rows}
        if split_ids is None:
            selected_rows = rows
        else:
            missing_ids = [sample_id for sample_id in split_ids if sample_id not in rows_by_id]
            if missing_ids:
                raise ValueError(
                    f"{len(missing_ids)} split ids are missing from {sample_path}; first missing id: {missing_ids[0]}"
                )
            selected_rows = [rows_by_id[sample_id] for sample_id in split_ids]

        for row in selected_rows:
            item = self._normalize_sample(row)
            if filter_unlabeled and item["label_id"] < 0:
                continue
            self.samples.append(item)
            if limit is not None and len(self.samples) >= limit:
                break

        self.edges_by_sample: dict[str, list[dict[str, str]]] = {}
        self.events_by_sample: dict[str, list[dict[str, str]]] = {}
        selected_sample_ids = {sample["sample_id"] for sample in self.samples}
        if include_edges:
            self.edges_by_sample = _group_csv_by_sample(
                self.dataset_dir / "edges.csv",
                max_rows_per_sample=max_edges_per_sample,
                allowed_sample_ids=selected_sample_ids,
            )
        if include_events:
            self.events_by_sample = _group_csv_by_sample(
                self.dataset_dir / "events.csv",
                max_rows_per_sample=max_events_per_sample,
                allowed_sample_ids=selected_sample_ids,
            )

    def _raw_label_id(self, label: str) -> int:
        dataset_maps = self.label_map["datasets"].get(self.dataset, {})
        raw_map = dataset_maps.get("raw_label", {})
        return int(raw_map.get(label, -1))

    def _task_label_id(self, row: dict[str, str]) -> int:
        if self.task == "raw":
            return self._raw_label_id(row.get("label", ""))

        if self.task == "veracity":
            source = self.label_map["datasets"].get(self.dataset, {}).get("veracity_source")
            if not source:
                return -1
            label = row.get(source, "")
            return int(self.label_map["tasks"]["veracity"].get(label, -1))

        if self.task == "rumor_binary":
            label = row.get("label", "")
            return int(self.label_map["tasks"]["rumor_binary"].get(label, -1))

        raise ValueError(f"Unknown task: {self.task}")

    def _normalize_sample(self, row: dict[str, str]) -> dict[str, Any]:
        sample_id = row.get("sample_id", "")
        return {
            "dataset": self.dataset,
            "sample_id": sample_id,
            "event": row.get("event", ""),
            "source_text": row.get("source_text", ""),
            "raw_label": row.get("label", ""),
            "raw_label_id": self._raw_label_id(row.get("label", "")),
            "veracity": row.get("veracity", ""),
            "label_id": self._task_label_id(row),
            "num_nodes": _to_int(row.get("num_nodes", "")),
            "num_edges": _to_int(row.get("num_edges", "")),
            "max_delay_minutes": _to_float(row.get("max_delay_minutes", "")),
            "has_source_text": row.get("has_source_text", "0") == "1",
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.samples[index])
        sample_id = item["sample_id"]
        if self.edges_by_sample:
            item["edges"] = self.edges_by_sample.get(sample_id, [])
        if self.events_by_sample:
            item["events"] = self.events_by_sample.get(sample_id, [])
        return item

    def label_distribution(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for sample in self.samples:
            counts[sample["label_id"]] += 1
        return dict(sorted(counts.items()))

    def split_summary(self) -> dict[str, Any] | None:
        if self.split is None or self.split_metadata is None:
            return None
        summary = self.split_metadata.get("summary", {}).get(self.split, {})
        return {
            "split": self.split,
            "split_file": str(self.split_file),
            "strategy": self.split_metadata.get("strategy"),
            "seed": self.split_metadata.get("seed"),
            "expected_num_samples": summary.get("num_samples"),
            "loaded_num_samples": len(self.samples),
            "expected_label_distribution": summary.get("label_distribution", {}),
            "loaded_label_distribution": self.label_distribution(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["weibo", "twitter15", "twitter16", "pheme"])
    parser.add_argument("--task", default="rumor_binary", choices=["raw", "rumor_binary", "veracity"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--include-edges", action="store_true")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--max-edges-per-sample", type=int, default=200)
    parser.add_argument("--max-events-per-sample", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--split-file")
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    dataset = RumorDataset(
        dataset=args.dataset,
        data_root=args.data_root,
        label_map_path=args.label_map,
        task=args.task,
        include_edges=args.include_edges,
        include_events=args.include_events,
        max_edges_per_sample=args.max_edges_per_sample,
        max_events_per_sample=args.max_events_per_sample,
        limit=args.limit,
        split=args.split,
        split_file=args.split_file,
        split_root=args.split_root,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    print_json(
        {
            "dataset": args.dataset,
            "task": args.task,
            "split_summary": dataset.split_summary(),
            "num_loaded": len(dataset),
            "label_distribution": dataset.label_distribution(),
            "examples": [dataset[i] for i in range(min(len(dataset), args.limit))],
        }
    )


if __name__ == "__main__":
    main()
