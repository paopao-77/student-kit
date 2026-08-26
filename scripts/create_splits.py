import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_SPLIT_ROOT = DEFAULT_DATA_ROOT / "splits"
DEFAULT_LABEL_MAP = Path("label_map.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_twitter_datetime(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return ""
    return dt.isoformat()


def task_label(row: dict[str, str], dataset: str, task: str, label_map: dict[str, Any]) -> int:
    if task == "raw":
        return int(label_map["datasets"].get(dataset, {}).get("raw_label", {}).get(row.get("label", ""), -1))
    if task == "rumor_binary":
        return int(label_map["tasks"]["rumor_binary"].get(row.get("label", ""), -1))
    if task == "veracity":
        source = label_map["datasets"].get(dataset, {}).get("veracity_source")
        if not source:
            return -1
        return int(label_map["tasks"]["veracity"].get(row.get(source, ""), -1))
    raise ValueError(f"Unknown task: {task}")


def source_times_for_pheme(dataset_dir: Path) -> dict[str, str]:
    events_path = dataset_dir / "events.csv"
    result = {}
    with events_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("is_source") == "1":
                result[row["sample_id"]] = parse_twitter_datetime(row.get("created_at", ""))
    return result


def sort_key_for_sample(row: dict[str, str], dataset: str, pheme_times: dict[str, str]) -> str:
    sample_id = row["sample_id"]
    if dataset == "pheme":
        return pheme_times.get(sample_id, "")
    if dataset in {"twitter15", "twitter16"}:
        return sample_id.zfill(32)
    return sample_id.zfill(32)


def split_ids(items: list[dict[str, Any]], ratios: tuple[float, float, float]) -> dict[str, list[str]]:
    n = len(items)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    return {
        "train": [item["sample_id"] for item in items[:train_end]],
        "val": [item["sample_id"] for item in items[train_end:val_end]],
        "test": [item["sample_id"] for item in items[val_end:]],
    }


def stratified_split_ids(
    items: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        by_label.setdefault(item["label_id"], []).append(item)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for label_id in sorted(by_label):
        label_items = by_label[label_id]
        rng.shuffle(label_items)
        label_splits = split_ids(label_items, ratios)
        for split_name, ids in label_splits.items():
            splits[split_name].extend(ids)

    for split_name in splits:
        rng.shuffle(splits[split_name])
    return splits


def summarize_split(items_by_id: dict[str, dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, Any]:
    summary = {}
    for name, ids in splits.items():
        labels = Counter(items_by_id[sid]["label_id"] for sid in ids)
        raw_labels = Counter(items_by_id[sid]["raw_label"] for sid in ids)
        sort_keys = [items_by_id[sid]["sort_key"] for sid in ids if items_by_id[sid]["sort_key"]]
        summary[name] = {
            "num_samples": len(ids),
            "label_distribution": dict(sorted(labels.items())),
            "raw_label_distribution": dict(sorted(raw_labels.items())),
            "first_sort_key": min(sort_keys) if sort_keys else "",
            "last_sort_key": max(sort_keys) if sort_keys else "",
        }
    return summary


def create_split_for_dataset(
    dataset: str,
    task: str,
    data_root: Path,
    split_root: Path,
    label_map: dict[str, Any],
    ratios: tuple[float, float, float],
    seed: int,
    strategy: str,
) -> dict[str, Any]:
    dataset_dir = data_root / dataset
    rows = read_csv(dataset_dir / "samples.csv")
    pheme_times = source_times_for_pheme(dataset_dir) if dataset == "pheme" else {}
    items = []
    dropped_unlabeled = 0

    for row in rows:
        label_id = task_label(row, dataset, task, label_map)
        if label_id < 0:
            dropped_unlabeled += 1
            continue
        item = {
            "sample_id": row["sample_id"],
            "raw_label": row.get("label", ""),
            "label_id": label_id,
            "sort_key": sort_key_for_sample(row, dataset, pheme_times),
        }
        items.append(item)

    if strategy == "stratified":
        split_strategy = "stratified_seed_random"
        items.sort(key=lambda item: (item["label_id"], item["sample_id"]))
    elif dataset == "weibo":
        split_strategy = "fixed_seed_random"
        rng = random.Random(seed)
        rng.shuffle(items)
    else:
        split_strategy = "time_order"
        items.sort(key=lambda item: (item["sort_key"], item["sample_id"]))

    if split_strategy == "stratified_seed_random":
        splits = stratified_split_ids(items, ratios, seed)
    else:
        splits = split_ids(items, ratios)

    items_by_id = {item["sample_id"]: item for item in items}
    summary = summarize_split(items_by_id, splits)
    output = {
        "dataset": dataset,
        "task": task,
        "strategy": split_strategy,
        "seed": seed if split_strategy in {"fixed_seed_random", "stratified_seed_random"} else None,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "num_available_labeled_samples": len(items),
        "num_dropped_unlabeled_samples": dropped_unlabeled,
        "splits": splits,
        "summary": summary,
        "notes": strategy_notes(dataset, split_strategy),
    }

    split_root.mkdir(parents=True, exist_ok=True)
    if split_strategy == "stratified_seed_random":
        suffix = f"stratified_seed{seed}"
    elif split_strategy == "fixed_seed_random":
        suffix = f"seed{seed}"
    else:
        suffix = "time"
    out_path = split_root / f"{dataset}_{task}_{suffix}_split.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return {"path": str(out_path), **output}


def strategy_notes(dataset: str, strategy: str) -> list[str]:
    if strategy == "stratified_seed_random":
        return ["Fixed-seed stratified random split for model tuning and stable validation."]
    if dataset == "pheme":
        return ["Strict chronological split by source tweet created_at."]
    if dataset in {"twitter15", "twitter16"}:
        return ["Chronological proxy split by numeric source tweet id because absolute source timestamps are unavailable."]
    if dataset == "weibo":
        return ["Fixed-seed random split because the BiGCN Weibo version has no reliable timestamps."]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="pheme,twitter15,twitter16,weibo")
    parser.add_argument("--task", default="rumor_binary", choices=["raw", "rumor_binary", "veracity"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="temporal", choices=["temporal", "stratified"])
    parser.add_argument("--full-output", action="store_true", help="Print full split ids instead of compact summaries.")
    args = parser.parse_args()

    ratios = tuple(float(part) for part in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("--ratios must have three values summing to 1.0")

    label_map = read_json(Path(args.label_map))
    results = []
    for dataset in [d.strip().lower() for d in args.datasets.split(",") if d.strip()]:
        results.append(
            create_split_for_dataset(
                dataset=dataset,
                task=args.task,
                data_root=Path(args.data_root),
                split_root=Path(args.split_root),
                label_map=label_map,
                ratios=ratios,  # type: ignore[arg-type]
                seed=args.seed,
                strategy=args.strategy,
            )
        )
    if args.full_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        compact = [
            {
                "path": result["path"],
                "dataset": result["dataset"],
                "task": result["task"],
                "strategy": result["strategy"],
                "seed": result["seed"],
                "num_available_labeled_samples": result["num_available_labeled_samples"],
                "num_dropped_unlabeled_samples": result["num_dropped_unlabeled_samples"],
                "summary": result["summary"],
                "notes": result["notes"],
            }
            for result in results
        ]
        print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
