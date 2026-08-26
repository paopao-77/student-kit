import argparse
import ast
import csv
import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ZIP = Path("data/raw/rumdetect2017.zip")
DEFAULT_OUTPUT_ROOT = Path("data/processed")

DATASETS = {
    "twitter15": "twitter15_rumdetect2017",
    "twitter16": "twitter16_rumdetect2017",
}


def read_text(zip_file: zipfile.ZipFile, name: str) -> str:
    return zip_file.read(name).decode("utf-8", errors="replace")


def load_labels(zip_file: zipfile.ZipFile, base: str) -> dict[str, str]:
    labels = {}
    for line in read_text(zip_file, f"{base}/label.txt").splitlines():
        if ":" not in line:
            continue
        label, sample_id = line.strip().split(":", 1)
        labels[sample_id] = label
    return labels


def load_source_tweets(zip_file: zipfile.ZipFile, base: str) -> dict[str, str]:
    sources = {}
    for line in read_text(zip_file, f"{base}/source_tweets.txt").splitlines():
        if "\t" not in line:
            continue
        sample_id, text = line.split("\t", 1)
        sources[sample_id] = text
    return sources


def parse_node(value: str) -> tuple[str, str, float]:
    raw = ast.literal_eval(value)
    if len(raw) != 3:
        raise ValueError(f"Expected 3 node fields, got {raw!r}")
    user_id = str(raw[0])
    tweet_id = str(raw[1])
    delay = float(raw[2])
    return user_id, tweet_id, delay


def parse_tree_lines(lines: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    edges: list[dict[str, str]] = []
    events_by_tweet: dict[str, tuple[str, str, float]] = {}

    for line in lines:
        line = line.strip()
        if not line or "->" not in line:
            continue
        parent_raw, child_raw = line.split("->", 1)
        parent_user, parent_tweet, parent_delay = parse_node(parent_raw)
        child_user, child_tweet, child_delay = parse_node(child_raw)

        for user_id, tweet_id, delay in (
            (parent_user, parent_tweet, parent_delay),
            (child_user, child_tweet, child_delay),
        ):
            if tweet_id not in events_by_tweet:
                events_by_tweet[tweet_id] = (tweet_id, user_id, delay)

        edges.append(
            {
                "parent_user_id": parent_user,
                "parent_tweet_id": parent_tweet,
                "parent_delay_minutes": f"{parent_delay:g}",
                "child_user_id": child_user,
                "child_tweet_id": child_tweet,
                "child_delay_minutes": f"{child_delay:g}",
            }
        )

    events = [
        {
            "tweet_id": tweet_id,
            "user_id": user_id,
            "delay_minutes": f"{delay:g}",
        }
        for tweet_id, user_id, delay in sorted(
            events_by_tweet.values(), key=lambda item: (item[2], item[0])
        )
    ]
    summary = {
        "num_nodes": len(events_by_tweet),
        "num_edges": len(edges),
        "max_delay_minutes": max(
            (delay for tweet_id, _user_id, delay in events_by_tweet.values() if tweet_id != "ROOT"),
            default=0.0,
        ),
    }
    return events, edges, summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def convert_dataset(
    zip_file: zipfile.ZipFile,
    source_name: str,
    output_name: str,
    output_root: Path,
) -> dict[str, Any]:
    base = f"rumor_detection_acl2017/{source_name}"
    labels = load_labels(zip_file, base)
    source_tweets = load_source_tweets(zip_file, base)
    tree_names = sorted(
        name
        for name in zip_file.namelist()
        if name.startswith(f"{base}/tree/") and name.endswith(".txt")
    )
    tree_ids = {Path(name).stem for name in tree_names}
    all_ids = sorted(set(labels) | set(source_tweets) | tree_ids)

    output_dir = output_root / output_name
    samples: list[dict[str, Any]] = []
    events_handle, events_writer = make_writer(
        output_dir / "events.csv",
        ["dataset", "sample_id", "tweet_id", "user_id", "delay_minutes", "is_source", "text"],
    )
    edges_handle, edges_writer = make_writer(
        output_dir / "edges.csv",
        [
            "dataset",
            "sample_id",
            "parent_user_id",
            "parent_tweet_id",
            "parent_delay_minutes",
            "child_user_id",
            "child_tweet_id",
            "child_delay_minutes",
        ],
    )

    missing_tree = 0
    missing_label = 0
    missing_source = 0
    node_counts: list[int] = []
    edge_counts: list[int] = []
    max_delays: list[float] = []
    label_counts: Counter[str] = Counter()
    user_ids: set[str] = set()
    num_events = 0
    num_edges = 0

    tree_name_by_id = {Path(name).stem: name for name in tree_names}

    try:
        for sample_id in all_ids:
            label = labels.get(sample_id, "")
            source_text = source_tweets.get(sample_id, "")
            tree_name = tree_name_by_id.get(sample_id)

            if not label:
                missing_label += 1
            if not source_text:
                missing_source += 1
            if not tree_name:
                missing_tree += 1
                events, edges, tree_summary = [], [], {
                    "num_nodes": 0,
                    "num_edges": 0,
                    "max_delay_minutes": 0.0,
                }
            else:
                lines = read_text(zip_file, tree_name).splitlines()
                events, edges, tree_summary = parse_tree_lines(lines)

            for event in events:
                is_source = int(event["tweet_id"] == sample_id)
                if event["user_id"] != "ROOT":
                    user_ids.add(event["user_id"])
                events_writer.writerow(
                    {
                        "dataset": output_name,
                        "sample_id": sample_id,
                        "tweet_id": event["tweet_id"],
                        "user_id": event["user_id"],
                        "delay_minutes": event["delay_minutes"],
                        "is_source": is_source,
                        "text": source_text if is_source else "",
                    }
                )
                num_events += 1

            for edge in edges:
                edges_writer.writerow({"dataset": output_name, "sample_id": sample_id, **edge})
                num_edges += 1

            label_counts[label] += 1
            node_counts.append(int(tree_summary["num_nodes"]))
            edge_counts.append(int(tree_summary["num_edges"]))
            max_delays.append(float(tree_summary["max_delay_minutes"]))
            samples.append(
                {
                    "dataset": output_name,
                    "sample_id": sample_id,
                    "source_text": source_text,
                    "label": label,
                    "num_nodes": int(tree_summary["num_nodes"]),
                    "num_edges": int(tree_summary["num_edges"]),
                    "max_delay_minutes": round(float(tree_summary["max_delay_minutes"]), 6),
                    "has_source_text": int(bool(source_text)),
                    "source_dataset": source_name,
                    "time_reference": "relative_delay_minutes",
                }
            )
    finally:
        events_handle.close()
        edges_handle.close()

    write_csv(
        output_dir / "samples.csv",
        [
            "dataset",
            "sample_id",
            "source_text",
            "label",
            "num_nodes",
            "num_edges",
            "max_delay_minutes",
            "has_source_text",
            "source_dataset",
            "time_reference",
        ],
        samples,
    )
    stats = {
        "dataset": output_name,
        "source_dataset": source_name,
        "source_archive": str(DEFAULT_ZIP),
        "num_samples": len(samples),
        "num_labels": len([label for label in label_counts if label]),
        "label_distribution": dict(sorted(label_counts.items())),
        "num_users": len(user_ids),
        "num_edges": num_edges,
        "num_events": num_events,
        "avg_cascade_nodes": round(statistics.fmean(node_counts), 4) if node_counts else 0,
        "min_cascade_nodes": min(node_counts) if node_counts else 0,
        "max_cascade_nodes": max(node_counts) if node_counts else 0,
        "avg_max_delay_minutes": round(statistics.fmean(max_delays), 4) if max_delays else 0,
        "samples_missing_tree": missing_tree,
        "samples_missing_label": missing_label,
        "samples_with_source_text": sum(1 for row in samples if row["has_source_text"]),
        "samples_missing_source_text": missing_source,
        "notes": [
            "Only source tweet text is available; non-source tweet text is omitted by the original release.",
            "Tree node time is relative post delay in minutes, not an absolute timestamp.",
        ],
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--datasets",
        default="twitter15,twitter16",
        help="Comma-separated source datasets inside the archive.",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    selected = [part.strip().lower() for part in args.datasets.split(",") if part.strip()]
    with zipfile.ZipFile(zip_path) as zip_file:
        results = []
        for source_name in selected:
            if source_name not in DATASETS:
                raise ValueError(f"Unsupported dataset {source_name!r}; choose from {sorted(DATASETS)}")
            results.append(
                convert_dataset(
                    zip_file=zip_file,
                    source_name=source_name,
                    output_name=DATASETS[source_name],
                    output_root=Path(args.output_root),
                )
            )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
