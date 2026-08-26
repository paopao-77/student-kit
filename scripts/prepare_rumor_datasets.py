import argparse
import ast
import csv
from datetime import datetime
import json
import re
from collections import Counter
from pathlib import Path


EDGE_RE = re.compile(r"^(?P<parent>\[.*?\])->(?P<child>\[.*?\])$")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_source_tweets(dataset_dir: Path) -> dict[str, str]:
    source_file = dataset_dir / "source_tweets.txt"
    if not source_file.exists():
        return {}

    result = {}
    with source_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def read_labels(dataset_dir: Path) -> dict[str, str]:
    labels = {}
    label_file = dataset_dir / "label.txt"
    with label_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, source_id = line.split(":", 1)
            labels[source_id] = label
    return labels


def parse_node(text: str) -> tuple[str, str, float]:
    uid, tweet_id, delay = ast.literal_eval(text)
    return str(uid), str(tweet_id), float(delay)


def parse_tree_file(path: Path) -> tuple[list[dict], dict[str, dict]]:
    edges = []
    nodes = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            match = EDGE_RE.match(line)
            if not match:
                raise ValueError(f"Cannot parse edge at {path}:{line_no}: {line[:120]}")

            parent_uid, parent_tweet_id, parent_delay = parse_node(match.group("parent"))
            child_uid, child_tweet_id, child_delay = parse_node(match.group("child"))

            for uid, tweet_id, delay in [
                (parent_uid, parent_tweet_id, parent_delay),
                (child_uid, child_tweet_id, child_delay),
            ]:
                if tweet_id not in nodes:
                    nodes[tweet_id] = {
                        "user_id": uid,
                        "tweet_id": tweet_id,
                        "delay_minutes": delay,
                    }

            edges.append(
                {
                    "parent_user_id": parent_uid,
                    "parent_tweet_id": parent_tweet_id,
                    "parent_delay_minutes": parent_delay,
                    "child_user_id": child_uid,
                    "child_tweet_id": child_tweet_id,
                    "child_delay_minutes": child_delay,
                }
            )

    return edges, nodes


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_twitter_dataset(raw_root: Path, dataset_name: str, output_root: Path) -> dict:
    dataset_dir = raw_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    labels = read_labels(dataset_dir)
    source_texts = read_source_tweets(dataset_dir)
    tree_dir = dataset_dir / "tree"
    out_dir = output_root / dataset_name
    ensure_dir(out_dir)

    samples = []
    edges_out = []
    events_out = []
    label_counter = Counter()
    cascade_lengths = []
    max_delays = []
    all_users = set()
    missing_tree = []

    for source_id, label in sorted(labels.items()):
        tree_file = tree_dir / f"{source_id}.txt"
        if not tree_file.exists():
            missing_tree.append(source_id)
            continue

        edges, nodes = parse_tree_file(tree_file)
        label_counter[label] += 1
        cascade_lengths.append(len(nodes))
        max_delay = max((n["delay_minutes"] for n in nodes.values()), default=0.0)
        max_delays.append(max_delay)

        for node in nodes.values():
            if node["user_id"] != "ROOT":
                all_users.add(node["user_id"])
            events_out.append(
                {
                    "dataset": dataset_name,
                    "sample_id": source_id,
                    "tweet_id": node["tweet_id"],
                    "user_id": node["user_id"],
                    "delay_minutes": node["delay_minutes"],
                    "is_source": "1" if node["tweet_id"] == source_id else "0",
                }
            )

        for edge in edges:
            row = {"dataset": dataset_name, "sample_id": source_id}
            row.update(edge)
            edges_out.append(row)

        samples.append(
            {
                "dataset": dataset_name,
                "sample_id": source_id,
                "source_text": source_texts.get(source_id, ""),
                "label": label,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "max_delay_minutes": max_delay,
                "has_source_text": "1" if source_id in source_texts else "0",
            }
        )

    write_csv(
        out_dir / "samples.csv",
        samples,
        [
            "dataset",
            "sample_id",
            "source_text",
            "label",
            "num_nodes",
            "num_edges",
            "max_delay_minutes",
            "has_source_text",
        ],
    )
    write_csv(
        out_dir / "edges.csv",
        edges_out,
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
    write_csv(
        out_dir / "events.csv",
        events_out,
        ["dataset", "sample_id", "tweet_id", "user_id", "delay_minutes", "is_source"],
    )

    stats = {
        "dataset": dataset_name,
        "num_samples": len(samples),
        "num_labels": len(label_counter),
        "label_distribution": dict(sorted(label_counter.items())),
        "num_users": len(all_users),
        "num_edges": len(edges_out),
        "avg_cascade_nodes": round(sum(cascade_lengths) / len(cascade_lengths), 4)
        if cascade_lengths
        else 0,
        "min_cascade_nodes": min(cascade_lengths) if cascade_lengths else 0,
        "max_cascade_nodes": max(cascade_lengths) if cascade_lengths else 0,
        "avg_max_delay_minutes": round(sum(max_delays) / len(max_delays), 4)
        if max_delays
        else 0,
        "samples_missing_tree": len(missing_tree),
        "samples_with_source_text": sum(1 for row in samples if row["has_source_text"] == "1"),
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def read_weibo_labels(label_file: Path) -> dict[str, str]:
    labels = {}
    with label_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def read_weibo_original_id_list(path: Path, limit: int = 0) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="gb18030", errors="replace") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            ids.append(value)
            if limit and len(ids) >= limit:
                break
    return ids


def read_weibo_original_root_texts(path: Path, allowed_ids: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    if not path.exists():
        return texts

    current_id: str | None = None
    with path.open("r", encoding="gb18030", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.isdigit():
                current_id = stripped
                continue
            if current_id is None:
                continue
            if current_id in allowed_ids and current_id not in texts:
                if not stripped.startswith("@") and not stripped.startswith("link"):
                    texts[current_id] = stripped
            if len(texts) >= len(allowed_ids):
                break
    return texts


def read_weibo_original_uid_list(path: Path) -> list[str]:
    uids: list[str] = []
    with path.open("r", encoding="gb18030", errors="replace") as f:
        for line in f:
            value = line.strip()
            if value:
                uids.append(value)
    return uids


WEIBO_PROFILE_FIELDS = [
    "id",
    "bi_followers_count",
    "city",
    "verified",
    "followers_count",
    "location",
    "province",
    "friends_count",
    "name",
    "gender",
    "created_at",
    "verified_type",
    "statuses_count",
    "description",
]


def read_weibo_original_profiles(profile_dir: Path, needed_uids: set[str]) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    if not profile_dir.exists() or not needed_uids:
        return profiles

    for path in [profile_dir / "user_profile1.txt", profile_dir / "user_profile2.txt"]:
        if not path.exists():
            continue
        with path.open("r", encoding="gb18030", errors="replace") as f:
            fields: list[str] = []
            for raw_line in f:
                line = raw_line.rstrip("\n\r")
                if line.startswith("#"):
                    continue
                fields.append(line)
                if len(fields) < len(WEIBO_PROFILE_FIELDS):
                    continue
                row = dict(zip(WEIBO_PROFILE_FIELDS, fields[: len(WEIBO_PROFILE_FIELDS)]))
                uid = row.get("id", "").strip()
                if uid in needed_uids:
                    profiles[uid] = {
                        "followers_count": row.get("followers_count", ""),
                        "friends_count": row.get("friends_count", ""),
                        "statuses_count": row.get("statuses_count", ""),
                        "verified": row.get("verified", ""),
                    }
                    if len(profiles) >= len(needed_uids):
                        return profiles
                fields = []
    return profiles


def prepare_weibo_original_dataset(raw_dir: Path, output_root: Path, limit: int = 0) -> dict:
    diffusion_dir = raw_dir / "diffusion"
    repost_idlist = diffusion_dir / "repost_idlist.txt"
    repost_data = diffusion_dir / "repost_data.txt"
    uidlist_path = diffusion_dir / "uidlist.txt"
    if not repost_idlist.exists() or not repost_data.exists() or not uidlist_path.exists():
        raise FileNotFoundError(
            f"Expected {repost_idlist}, {repost_data}, and {uidlist_path}"
        )

    sample_ids = read_weibo_original_id_list(repost_idlist, limit=limit)
    allowed_post_ids = set(range(len(sample_ids)))
    source_texts = read_weibo_original_root_texts(raw_dir / "root_content.txt", set(sample_ids))
    if not source_texts and (raw_dir / "weibocontents" / "Root_Content.txt").exists():
        source_texts = read_weibo_original_root_texts(
            raw_dir / "weibocontents" / "Root_Content.txt",
            set(sample_ids),
        )
    uidlist = read_weibo_original_uid_list(uidlist_path)

    out_dir = output_root / "weibo"
    ensure_dir(out_dir)
    sample_fields = [
        "dataset",
        "sample_id",
        "source_text",
        "label",
        "num_nodes",
        "num_edges",
        "max_delay_minutes",
        "has_source_text",
    ]
    event_fields = [
        "dataset",
        "sample_id",
        "tweet_id",
        "user_id",
        "delay_minutes",
        "event_order",
        "is_source",
        "text",
        "followers_count",
        "friends_count",
        "statuses_count",
        "verified",
    ]
    edge_fields = [
        "dataset",
        "sample_id",
        "parent_user_id",
        "parent_tweet_id",
        "parent_delay_minutes",
        "child_user_id",
        "child_tweet_id",
        "child_delay_minutes",
    ]

    samples: list[dict] = []
    event_buffer: list[dict] = []
    edge_buffer: list[dict] = []
    needed_uids: set[str] = set()
    cascade_lengths: list[int] = []
    max_delays: list[float] = []

    with repost_data.open("r", encoding="gb18030", errors="replace") as f:
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            parts = header.split()
            if len(parts) < 2:
                continue
            post_id = int(parts[0])
            repost_count = int(parts[1])
            if post_id not in allowed_post_ids:
                for _ in range(repost_count):
                    f.readline()
                if limit and post_id >= limit:
                    break
                continue

            sample_id = sample_ids[post_id]
            source_text = source_texts.get(sample_id, "")
            rows: list[tuple[int, str, str]] = []
            for order in range(1, repost_count + 1):
                line = f.readline()
                if not line:
                    break
                values = line.strip().split()
                if len(values) < 2:
                    continue
                timestamp = int(values[0])
                internal_uid = int(values[1])
                uid = uidlist[internal_uid] if 0 <= internal_uid < len(uidlist) else ""
                if uid:
                    needed_uids.add(uid)
                rows.append((timestamp, uid, f"{sample_id}:r:{order}"))

            first_ts = min((row[0] for row in rows), default=0)
            max_delay = 0.0
            event_buffer.append(
                {
                    "dataset": "weibo",
                    "sample_id": sample_id,
                    "tweet_id": sample_id,
                    "user_id": "",
                    "delay_minutes": 0.0,
                    "event_order": 0,
                    "is_source": "1",
                    "text": source_text,
                    "followers_count": "",
                    "friends_count": "",
                    "statuses_count": "",
                    "verified": "",
                }
            )
            for order, (timestamp, uid, tweet_id) in enumerate(rows, 1):
                delay = max((timestamp - first_ts) / 60.0, 0.0) if first_ts else 0.0
                max_delay = max(max_delay, delay)
                event_buffer.append(
                    {
                        "dataset": "weibo",
                        "sample_id": sample_id,
                        "tweet_id": tweet_id,
                        "user_id": uid,
                        "delay_minutes": round(delay, 6),
                        "event_order": order,
                        "is_source": "0",
                        "text": "",
                        "followers_count": "",
                        "friends_count": "",
                        "statuses_count": "",
                        "verified": "",
                    }
                )
                edge_buffer.append(
                    {
                        "dataset": "weibo",
                        "sample_id": sample_id,
                        "parent_user_id": "",
                        "parent_tweet_id": sample_id,
                        "parent_delay_minutes": 0.0,
                        "child_user_id": uid,
                        "child_tweet_id": tweet_id,
                        "child_delay_minutes": round(delay, 6),
                    }
                )
            node_count = len(rows) + 1
            cascade_lengths.append(node_count)
            max_delays.append(max_delay)
            samples.append(
                {
                    "dataset": "weibo",
                    "sample_id": sample_id,
                    "source_text": source_text,
                    "label": "0",
                    "num_nodes": node_count,
                    "num_edges": len(rows),
                    "max_delay_minutes": round(max_delay, 6),
                    "has_source_text": "1" if source_text else "0",
                }
            )
            if limit and len(samples) >= limit:
                break

    profiles = read_weibo_original_profiles(raw_dir / "userProfile", needed_uids)
    for row in event_buffer:
        uid = row.get("user_id", "")
        profile = profiles.get(uid)
        if profile:
            row.update(profile)

    write_csv(out_dir / "samples.csv", samples, sample_fields)
    write_csv(out_dir / "events.csv", event_buffer, event_fields)
    write_csv(out_dir / "edges.csv", edge_buffer, edge_fields)

    stats = {
        "dataset": "weibo",
        "source": str(raw_dir),
        "num_samples": len(samples),
        "num_labels": 1,
        "label_distribution": {"0": len(samples)},
        "num_users": len(needed_uids),
        "num_users_with_profile": len(profiles),
        "num_edges": len(edge_buffer),
        "avg_cascade_nodes": round(sum(cascade_lengths) / len(cascade_lengths), 4)
        if cascade_lengths
        else 0,
        "min_cascade_nodes": min(cascade_lengths) if cascade_lengths else 0,
        "max_cascade_nodes": max(cascade_lengths) if cascade_lengths else 0,
        "avg_max_delay_minutes": round(sum(max_delays) / len(max_delays), 4)
        if max_delays
        else 0,
        "samples_missing_tree": 0,
        "samples_with_source_text": sum(1 for row in samples if row["has_source_text"] == "1"),
        "notes": [
            "Parsed directly from 数据集/微博 diffusion, root_content, and userProfile files.",
            "The raw repost data has cascade-size targets but no rumor/non-rumor labels; label is set to 0 for split compatibility.",
            "Edges are source-to-retweet star edges because diffusion/repost_data.txt does not encode retweet parent IDs.",
        ],
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def prepare_weibo_dataset(raw_dir: Path, output_root: Path) -> dict:
    label_file = raw_dir / "weibo_id_label.txt"
    tree_file = raw_dir / "weibotree.txt"
    if not label_file.exists() or not tree_file.exists():
        raise FileNotFoundError(f"Expected {label_file} and {tree_file}")

    labels = read_weibo_labels(label_file)
    out_dir = output_root / "weibo"
    ensure_dir(out_dir)

    samples_by_id = {}
    edges_out = []
    events_by_key = {}
    label_counter = Counter()
    all_nodes_by_sample = {}

    with tree_file.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                raise ValueError(f"Cannot parse Weibo row at {tree_file}:{line_no}: {line[:120]}")

            sample_id, parent_id, child_id, sparse_features = parts[0], parts[1], parts[2], parts[3]
            label = labels.get(sample_id, "")
            samples_by_id.setdefault(sample_id, label)
            all_nodes_by_sample.setdefault(sample_id, set()).add(child_id)
            if parent_id != "None":
                all_nodes_by_sample[sample_id].add(parent_id)

            feature_count = 0 if not sparse_features else len(sparse_features.split())
            events_by_key[(sample_id, child_id)] = {
                "dataset": "weibo",
                "sample_id": sample_id,
                "tweet_id": child_id,
                "user_id": "",
                "delay_minutes": "",
                "event_order": child_id,
                "is_source": "1" if parent_id == "None" else "0",
                "feature_count": feature_count,
                "sparse_features": sparse_features,
            }

            if parent_id != "None":
                edges_out.append(
                    {
                        "dataset": "weibo",
                        "sample_id": sample_id,
                        "parent_user_id": "",
                        "parent_tweet_id": parent_id,
                        "parent_delay_minutes": "",
                        "child_user_id": "",
                        "child_tweet_id": child_id,
                        "child_delay_minutes": "",
                    }
                )

    samples = []
    cascade_lengths = []
    edge_counter = Counter(edge["sample_id"] for edge in edges_out)
    for sample_id, label in sorted(samples_by_id.items()):
        node_count = len(all_nodes_by_sample.get(sample_id, set()))
        edge_count = edge_counter[sample_id]
        cascade_lengths.append(node_count)
        if label:
            label_counter[label] += 1
        samples.append(
            {
                "dataset": "weibo",
                "sample_id": sample_id,
                "source_text": "",
                "label": label,
                "num_nodes": node_count,
                "num_edges": edge_count,
                "max_delay_minutes": "",
                "has_source_text": "0",
            }
        )

    events_out = [events_by_key[key] for key in sorted(events_by_key)]
    write_csv(
        out_dir / "samples.csv",
        samples,
        [
            "dataset",
            "sample_id",
            "source_text",
            "label",
            "num_nodes",
            "num_edges",
            "max_delay_minutes",
            "has_source_text",
        ],
    )
    write_csv(
        out_dir / "edges.csv",
        edges_out,
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
    write_csv(
        out_dir / "events.csv",
        events_out,
        [
            "dataset",
            "sample_id",
            "tweet_id",
            "user_id",
            "delay_minutes",
            "event_order",
            "is_source",
            "feature_count",
            "sparse_features",
        ],
    )

    stats = {
        "dataset": "weibo",
        "num_samples": len(samples),
        "num_labels": len(label_counter),
        "label_distribution": dict(sorted(label_counter.items())),
        "num_users": 0,
        "num_edges": len(edges_out),
        "avg_cascade_nodes": round(sum(cascade_lengths) / len(cascade_lengths), 4)
        if cascade_lengths
        else 0,
        "min_cascade_nodes": min(cascade_lengths) if cascade_lengths else 0,
        "max_cascade_nodes": max(cascade_lengths) if cascade_lengths else 0,
        "avg_max_delay_minutes": "",
        "samples_missing_tree": max(len(labels) - len(samples_by_id), 0),
        "samples_with_source_text": 0,
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def parse_twitter_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def flatten_structure(tree: dict, parent: str | None = None) -> list[tuple[str, str]]:
    edges = []
    for node_id, children in tree.items():
        if parent is not None:
            edges.append((parent, node_id))
        if isinstance(children, dict):
            edges.extend(flatten_structure(children, node_id))
    return edges


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def tweet_event_row(dataset: str, sample_id: str, tweet: dict, source_time: datetime | None, is_source: str) -> dict:
    created_at = tweet.get("created_at", "")
    dt = parse_twitter_datetime(created_at)
    delay = ""
    if dt is not None and source_time is not None:
        delay = round((dt - source_time).total_seconds() / 60.0, 6)

    user = tweet.get("user") or {}
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "tweet_id": str(tweet.get("id_str") or tweet.get("id") or ""),
        "user_id": str(user.get("id_str") or user.get("id") or ""),
        "created_at": created_at,
        "delay_minutes": delay,
        "is_source": is_source,
        "text": tweet.get("text", ""),
        "followers_count": user.get("followers_count", ""),
        "friends_count": user.get("friends_count", ""),
        "statuses_count": user.get("statuses_count", ""),
        "verified": user.get("verified", ""),
    }


def prepare_pheme_dataset(raw_root: Path, output_root: Path, limit: int = 0) -> dict:
    root = raw_root / "all-rnr-annotated-threads"
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = output_root / "pheme"
    ensure_dir(out_dir)

    label_counter = Counter()
    veracity_counter = Counter()
    all_users = set()
    cascade_lengths = []
    max_delays = []
    processed = 0
    edge_total = 0
    source_text_total = 0

    sample_fields = [
        "dataset",
        "sample_id",
        "event",
        "source_text",
        "label",
        "veracity",
        "num_nodes",
        "num_edges",
        "max_delay_minutes",
        "has_source_text",
    ]
    edge_fields = [
        "dataset",
        "sample_id",
        "parent_user_id",
        "parent_tweet_id",
        "parent_delay_minutes",
        "child_user_id",
        "child_tweet_id",
        "child_delay_minutes",
    ]
    event_fields = [
        "dataset",
        "sample_id",
        "tweet_id",
        "user_id",
        "created_at",
        "delay_minutes",
        "is_source",
        "text",
        "followers_count",
        "friends_count",
        "statuses_count",
        "verified",
    ]

    sample_file = (out_dir / "samples.csv").open("w", encoding="utf-8", newline="")
    edge_file = (out_dir / "edges.csv").open("w", encoding="utf-8", newline="")
    event_file = (out_dir / "events.csv").open("w", encoding="utf-8", newline="")
    sample_writer = csv.DictWriter(sample_file, fieldnames=sample_fields)
    edge_writer = csv.DictWriter(edge_file, fieldnames=edge_fields)
    event_writer = csv.DictWriter(event_file, fieldnames=event_fields)
    sample_writer.writeheader()
    edge_writer.writeheader()
    event_writer.writeheader()

    try:
        for event_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("._")):
            event_name = event_dir.name.replace("-all-rnr-threads", "")
            for label_dir_name in ["rumours", "non-rumours"]:
                label_dir = event_dir / label_dir_name
                if not label_dir.exists():
                    continue
                for thread_dir in sorted(p for p in label_dir.iterdir() if p.is_dir() and not p.name.startswith("._")):
                    if limit and processed >= limit:
                        break
                    sample_id = thread_dir.name
                    annotation_path = thread_dir / "annotation.json"
                    structure_path = thread_dir / "structure.json"
                    source_dir = thread_dir / "source-tweets"
                    reactions_dir = thread_dir / "reactions"
                    source_path = source_dir / f"{sample_id}.json"
                    if not annotation_path.exists() or not structure_path.exists() or not source_path.exists():
                        continue
                    processed += 1
                    if processed % 500 == 0:
                        print(f"processed PHEME threads: {processed}", flush=True)

                    annotation = read_json(annotation_path)
                    structure = read_json(structure_path)
                    source_tweet = read_json(source_path)
                    source_time = parse_twitter_datetime(source_tweet.get("created_at", ""))
                    label = annotation.get("is_rumour", "rumour" if label_dir_name == "rumours" else "nonrumour")
                    if label == "nonrumour":
                        veracity = "nonrumour"
                    elif str(annotation.get("true", "0")) == "1":
                        veracity = "true"
                    elif str(annotation.get("misinformation", "0")) == "1":
                        veracity = "false"
                    else:
                        veracity = "unverified"

                    label_counter[label] += 1
                    veracity_counter[veracity] += 1

                    event_rows = [tweet_event_row("pheme", sample_id, source_tweet, source_time, "1")]
                    if reactions_dir.exists():
                        for reaction_path in sorted(reactions_dir.glob("*.json")):
                            if reaction_path.name.startswith("._"):
                                continue
                            try:
                                reaction = read_json(reaction_path)
                            except json.JSONDecodeError:
                                continue
                            event_rows.append(tweet_event_row("pheme", sample_id, reaction, source_time, "0"))

                    node_ids = {row["tweet_id"] for row in event_rows if row["tweet_id"]}
                    for row in event_rows:
                        if row["user_id"]:
                            all_users.add(row["user_id"])
                        event_writer.writerow(row)

                    structure_edges = flatten_structure(structure)
                    edge_total += len(structure_edges)
                    for parent_id, child_id in structure_edges:
                        edge_writer.writerow(
                            {
                            "dataset": "pheme",
                            "sample_id": sample_id,
                            "parent_user_id": "",
                            "parent_tweet_id": parent_id,
                            "parent_delay_minutes": "",
                            "child_user_id": "",
                            "child_tweet_id": child_id,
                            "child_delay_minutes": "",
                        }
                        )

                    numeric_delays = [
                        float(row["delay_minutes"])
                        for row in event_rows
                        if row["delay_minutes"] != ""
                    ]
                    max_delay = max(numeric_delays) if numeric_delays else 0.0
                    max_delays.append(max_delay)
                    cascade_lengths.append(len(node_ids))
                    has_source_text = "1" if source_tweet.get("text") else "0"
                    if has_source_text == "1":
                        source_text_total += 1

                    sample_writer.writerow(
                        {
                        "dataset": "pheme",
                        "sample_id": sample_id,
                        "event": event_name,
                        "source_text": source_tweet.get("text", ""),
                        "label": label,
                        "veracity": veracity,
                        "num_nodes": len(node_ids),
                        "num_edges": len(structure_edges),
                        "max_delay_minutes": round(max_delay, 6),
                        "has_source_text": has_source_text,
                    }
                    )
    finally:
        sample_file.close()
        edge_file.close()
        event_file.close()

    stats = {
        "dataset": "pheme",
        "num_samples": processed,
        "num_labels": len(label_counter),
        "label_distribution": dict(sorted(label_counter.items())),
        "veracity_distribution": dict(sorted(veracity_counter.items())),
        "num_users": len(all_users),
        "num_edges": edge_total,
        "avg_cascade_nodes": round(sum(cascade_lengths) / len(cascade_lengths), 4)
        if cascade_lengths
        else 0,
        "min_cascade_nodes": min(cascade_lengths) if cascade_lengths else 0,
        "max_cascade_nodes": max(cascade_lengths) if cascade_lengths else 0,
        "avg_max_delay_minutes": round(sum(max_delays) / len(max_delays), 4)
        if max_delays
        else 0,
        "samples_missing_tree": 0,
        "samples_with_source_text": source_text_total,
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def write_stats_csv(path: Path, stats_rows: list[dict]) -> None:
    fieldnames = [
        "dataset",
        "num_samples",
        "num_labels",
        "label_distribution",
        "veracity_distribution",
        "num_users",
        "num_edges",
        "avg_cascade_nodes",
        "min_cascade_nodes",
        "max_cascade_nodes",
        "avg_max_delay_minutes",
        "samples_missing_tree",
        "samples_with_source_text",
    ]
    rows = []
    for stats in stats_rows:
        row = dict(stats)
        row["label_distribution"] = json.dumps(
            row["label_distribution"], ensure_ascii=False, sort_keys=True
        )
        row["veracity_distribution"] = json.dumps(
            row.get("veracity_distribution", {}), ensure_ascii=False, sort_keys=True
        )
        rows.append({field: row.get(field, "") for field in fieldnames})
    write_csv(path, rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--twitter-root",
        default="data/raw/Twitter15_16_dataset-main",
        help="Path containing twitter15 and twitter16 directories.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed",
        help="Directory for normalized CSV outputs.",
    )
    parser.add_argument(
        "--weibo-root",
        default="数据集/微博",
        help="Path containing either the original Weibo diffusion files or BiGCN weibo_id_label.txt/weibotree.txt.",
    )
    parser.add_argument(
        "--pheme-root",
        default="data/raw/PHEME",
        help="Path containing all-rnr-annotated-threads.",
    )
    parser.add_argument(
        "--datasets",
        default="twitter15,twitter16,weibo,pheme",
            help="Comma-separated datasets to prepare: twitter15,twitter16,weibo,pheme.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional per-dataset sample limit for debugging.")
    args = parser.parse_args()

    raw_root = Path(args.twitter_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    requested = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}

    stats_rows = []
    for dataset_name in ["twitter15", "twitter16"]:
        if dataset_name in requested:
            stats_rows.append(prepare_twitter_dataset(raw_root, dataset_name, output_root))
    weibo_root = Path(args.weibo_root)
    if "weibo" in requested:
        if (
            (weibo_root / "diffusion" / "repost_data.txt").exists()
            and (weibo_root / "diffusion" / "repost_idlist.txt").exists()
            and (weibo_root / "diffusion" / "uidlist.txt").exists()
        ):
            stats_rows.append(prepare_weibo_original_dataset(weibo_root, output_root, limit=args.limit))
        elif (weibo_root / "weibo_id_label.txt").exists() and (weibo_root / "weibotree.txt").exists():
            stats_rows.append(prepare_weibo_dataset(weibo_root, output_root))
    pheme_root = Path(args.pheme_root)
    if "pheme" in requested and (pheme_root / "all-rnr-annotated-threads").exists():
        stats_rows.append(prepare_pheme_dataset(pheme_root, output_root, limit=args.limit))

    if stats_rows:
        write_stats_csv(output_root / "dataset_stats_latest.csv", stats_rows)
    print(json.dumps(stats_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
