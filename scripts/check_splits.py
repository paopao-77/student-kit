import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_ROOT = Path("data/processed/splits")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_one(path: Path) -> dict[str, Any]:
    data = read_json(path)
    splits = data["splits"]
    sets = {name: set(ids) for name, ids in splits.items()}
    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    overlap_counts = {key: len(value) for key, value in overlap.items()}
    all_ids = sets["train"] | sets["val"] | sets["test"]
    expected = data["num_available_labeled_samples"]
    chronological_ok = True
    if data["strategy"] == "time_order":
        summary = data["summary"]
        train_last = summary["train"]["last_sort_key"]
        val_first = summary["val"]["first_sort_key"]
        val_last = summary["val"]["last_sort_key"]
        test_first = summary["test"]["first_sort_key"]
        chronological_ok = (not val_first or train_last <= val_first) and (not test_first or val_last <= test_first)

    ok = all(count == 0 for count in overlap_counts.values()) and len(all_ids) == expected and chronological_ok
    return {
        "file": str(path),
        "dataset": data["dataset"],
        "task": data["task"],
        "strategy": data["strategy"],
        "ok": ok,
        "num_unique_ids": len(all_ids),
        "num_expected": expected,
        "overlap_counts": overlap_counts,
        "chronological_ok": chronological_ok,
        "summary": data["summary"],
        "notes": data.get("notes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--pattern", default="*.json")
    args = parser.parse_args()

    paths = sorted(Path(args.split_root).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No split files found in {args.split_root}")
    results = [check_one(path) for path in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
