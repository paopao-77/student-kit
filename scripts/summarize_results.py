import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "summary"
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "macro_f1", "auc"]
REGRESSION_METRICS = ["mae", "rmse", "mape", "smape", "r2", "median_ae"]
METRIC_NAMES = [*CLASSIFICATION_METRICS, *REGRESSION_METRICS]
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
DATASET_ORDER = {
    "weibo": 0,
    "twitter15": 1,
    "twitter16": 2,
    "twitter15_rumdetect2017": 3,
    "twitter16_rumdetect2017": 4,
    "pheme": 5,
}
MODEL_FAMILY_ORDER = {
    "structure_stats": 0,
    "propagation_graph": 1,
    "dynamics_seir": 2,
    "heterorumor_v1": 3,
    "heterorumor_v2_c1": 4,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_family(path: Path) -> str:
    parent = path.parent.name
    if parent == "baseline":
        return "structure_stats"
    if parent == "graph_baseline":
        return "propagation_graph"
    if parent == "seir_baseline":
        return "dynamics_seir"
    return parent


def split_size(payload: dict[str, Any], split: str) -> int | None:
    if "num_graphs" in payload:
        value = payload.get("num_graphs", {}).get(split)
        return int(value) if value is not None else None

    summary = payload.get("split_summaries", {}).get(split, {})
    value = summary.get("loaded_num_samples") or summary.get("expected_num_samples")
    return int(value) if value is not None else None


def label_distribution(payload: dict[str, Any], split: str) -> str:
    summary = payload.get("split_summaries", {}).get(split, {})
    distribution = summary.get("loaded_label_distribution") or summary.get("expected_label_distribution") or {}
    return json.dumps(distribution, ensure_ascii=False, sort_keys=True)


def confusion_matrix_value(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics.get("confusion_matrix"), ensure_ascii=False)


def metric_value(metrics: dict[str, Any], name: str) -> float | str:
    value = metrics.get(name)
    if value is None:
        return ""
    return float(value)


def flatten_metrics(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    family = payload.get("model_family") or model_family(path)
    rows = []

    for model_name, split_metrics in payload.get("models", {}).items():
        for split, metrics in split_metrics.items():
            row = {
                "dataset": payload.get("dataset", ""),
                "task": payload.get("task", ""),
                "split_strategy": payload.get("split_strategy", ""),
                "split": split,
                "model_family": family,
                "model": model_name,
                "seed": payload.get("seed", ""),
                "hops": payload.get("hops", payload.get("num_layers", "")),
                "max_edges_per_graph": payload.get("max_edges_per_graph", ""),
                "observation_window_minutes": metrics.get("observation_window_minutes", ""),
                "forecast_horizon_hours": metrics.get("forecast_horizon_hours", payload.get("forecast_horizon_hours", "")),
                "num_samples": split_size(payload, split),
                "label_distribution": label_distribution(payload, split),
                "confusion_matrix": confusion_matrix_value(metrics),
                "source_file": str(path),
            }
            for metric_name in METRIC_NAMES:
                row[metric_name] = metric_value(metrics, metric_name)
            fusion_weights = metrics.get("mean_fusion_weights", {})
            for modality in ("text", "topology", "temporal", "user_profile"):
                row[f"mean_{modality}_weight"] = metric_value(
                    fusion_weights, modality
                )
            rows.append(row)
    return rows


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.get(str(row.get("dataset")), 99),
        str(row.get("dataset")),
        str(row.get("split_strategy")),
        SPLIT_ORDER.get(str(row.get("split")), 99),
        MODEL_FAMILY_ORDER.get(str(row.get("model_family")), 99),
        str(row.get("model")),
    )


def best_test_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        key = (str(row.get("dataset")), str(row.get("split_strategy")))
        groups.setdefault(key, []).append(row)

    best_rows = []
    for key in sorted(groups):
        candidates = groups[key]
        best = max(candidates, key=lambda row: float(row.get("macro_f1") or -1.0))
        best_rows.append(best)
    return sorted(best_rows, key=sort_key)


def best_regression_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "test" or row.get("mape") in ("", None):
            continue
        window = row.get("observation_window_minutes")
        if window in ("", None):
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("split_strategy")),
            float(window),
        )
        groups.setdefault(key, []).append(row)

    best_rows = []
    for key in sorted(groups):
        best_rows.append(min(groups[key], key=lambda row: float(row["mape"])))
    return sorted(best_rows, key=sort_key)


def round_for_paper(row: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(row)
    for metric_name in METRIC_NAMES:
        value = rounded.get(metric_name)
        if value == "" or value is None:
            rounded[metric_name] = ""
        else:
            rounded[metric_name] = round(float(value), 4)
    return rounded


def v1_result_tables(
    test_rows: list[dict[str, Any]],
    dataset_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    v1_rows = [row for row in test_rows if row.get("model_family") == "heterorumor_v1"]
    if dataset_filter is not None:
        v1_rows = [row for row in v1_rows if row.get("dataset") == dataset_filter]
    primary = sorted(
        [row for row in v1_rows if row.get("model") == "heterorumor_v1_hurdle"],
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("split_strategy")),
            float(row.get("observation_window_minutes") or 0),
        ),
    )
    full_180 = next(
        (
            row
            for row in primary
            if float(row.get("observation_window_minutes") or 0) == 180.0
        ),
        None,
    )
    ablations = []
    if full_180 is not None:
        baseline_mape = float(full_180["mape"])
        for row in v1_rows:
            if not str(row.get("model", "")).startswith("heterorumor_v1_hurdle_wo_"):
                continue
            enriched = dict(row)
            enriched["mape_delta_vs_full"] = float(row["mape"]) - baseline_mape
            ablations.append(enriched)
        ablations.sort(key=lambda row: str(row.get("model")))
    split_comparison = sorted(
        [
            row
            for row in primary
            if float(row.get("observation_window_minutes") or 0) == 180.0
        ],
        key=lambda row: str(row.get("split_strategy")),
    )
    return primary, ablations, split_comparison


def v1_rumdetect2017_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in test_rows
            if row.get("model_family") == "heterorumor_v1"
            and row.get("model") == "heterorumor_v1_hurdle"
            and row.get("dataset") in {"twitter15_rumdetect2017", "twitter16_rumdetect2017"}
            and float(row.get("observation_window_minutes") or 0) == 180.0
        ],
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("split_strategy")),
        ),
    )


def v1_text_encoder_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_models = {
        "heterorumor_v1_hurdle": ("stable_hash", "full"),
        "heterorumor_v1_hurdle_wo_text": ("stable_hash", "no_text"),
        "heterorumor_v1_hurdle_multilingual_minilm": ("multilingual_minilm", "full"),
        "heterorumor_v1_hurdle_multilingual_minilm_wo_text": (
            "multilingual_minilm",
            "no_text",
        ),
    }
    rows = []
    for row in test_rows:
        if row.get("dataset") != "pheme":
            continue
        if float(row.get("observation_window_minutes") or 0) != 180.0:
            continue
        model = str(row.get("model"))
        if model not in selected_models:
            continue
        text_encoder, comparison_role = selected_models[model]
        enriched = dict(row)
        enriched["text_encoder"] = text_encoder
        enriched["comparison_role"] = comparison_role
        rows.append(enriched)
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("split_strategy")),
            str(row.get("text_encoder")),
            str(row.get("comparison_role")),
        ),
    )


def collect_rows(results_root: Path) -> list[dict[str, Any]]:
    paths = sorted(results_root.glob("*/*_metrics.json"))
    rows = []
    for path in paths:
        parent_name = path.parent.name.lower()
        if (
            path.parent.name == "summary"
            or "smoke" in parent_name
            or "multiseed" in parent_name
            or "sensitivity" in parent_name
        ):
            continue
        rows.extend(flatten_metrics(path))
    return sorted(rows, key=sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--round-paper", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = parse_args()
    rows = collect_rows(Path(args.results_root))
    if not rows:
        raise FileNotFoundError(f"No metrics files found under {args.results_root}")

    output_dir = Path(args.output_dir)
    all_fieldnames = [
        "dataset",
        "task",
        "split_strategy",
        "split",
        "model_family",
        "model",
        "seed",
        "hops",
        "max_edges_per_graph",
        "observation_window_minutes",
        "forecast_horizon_hours",
        "num_samples",
        "label_distribution",
        *METRIC_NAMES,
        "confusion_matrix",
        "mean_text_weight",
        "mean_topology_weight",
        "mean_temporal_weight",
        "mean_user_profile_weight",
        "source_file",
    ]
    test_fieldnames = [
        "dataset",
        "task",
        "split_strategy",
        "model_family",
        "model",
        "seed",
        "hops",
        "max_edges_per_graph",
        "observation_window_minutes",
        "forecast_horizon_hours",
        "num_samples",
        *METRIC_NAMES,
        "confusion_matrix",
        "mean_text_weight",
        "mean_topology_weight",
        "mean_temporal_weight",
        "mean_user_profile_weight",
        "source_file",
    ]

    all_metrics_path = output_dir / "all_metrics_long.csv"
    test_rows = [row for row in rows if row.get("split") == "test"]
    paper_rows = [round_for_paper(row) for row in test_rows]
    best_rows = [round_for_paper(row) for row in best_test_rows(rows)]
    best_regression = [round_for_paper(row) for row in best_regression_rows(rows)]
    v1_windows, v1_ablations, v1_splits = v1_result_tables(test_rows, dataset_filter="pheme")
    v1_text_rows = v1_text_encoder_rows(test_rows)
    v1_rumdetect_rows = v1_rumdetect2017_rows(test_rows)

    write_csv(all_metrics_path, rows, all_fieldnames)
    write_csv(output_dir / "paper_test_metrics.csv", paper_rows, test_fieldnames)
    write_csv(output_dir / "best_test_by_dataset_split.csv", best_rows, test_fieldnames)
    write_csv(
        output_dir / "best_regression_by_dataset_split_window.csv",
        best_regression,
        test_fieldnames,
    )
    write_csv(
        output_dir / "v1_pheme_window_comparison.csv",
        [round_for_paper(row) for row in v1_windows],
        test_fieldnames,
    )
    write_csv(
        output_dir / "v1_pheme_ablation.csv",
        [
            {**round_for_paper(row), "mape_delta_vs_full": round(row["mape_delta_vs_full"], 4)}
            for row in v1_ablations
        ],
        [*test_fieldnames, "mape_delta_vs_full"],
    )
    write_csv(
        output_dir / "v1_pheme_180_split_comparison.csv",
        [round_for_paper(row) for row in v1_splits],
        test_fieldnames,
    )
    write_csv(
        output_dir / "v1_pheme_text_encoder_comparison.csv",
        [round_for_paper(row) for row in v1_text_rows],
        [*test_fieldnames, "text_encoder", "comparison_role"],
    )
    rumdetect_paper_rows = [round_for_paper(row) for row in v1_rumdetect_rows]
    write_csv(
        output_dir / "v1_rumdetect2017_180_split_comparison.csv",
        rumdetect_paper_rows,
        test_fieldnames,
    )
    write_markdown(
        output_dir / "v1_rumdetect2017_180_split_comparison.md",
        rumdetect_paper_rows,
        [
            "dataset",
            "split_strategy",
            "mape",
            "mae",
            "rmse",
            "r2",
            "mean_text_weight",
            "mean_topology_weight",
            "mean_temporal_weight",
            "mean_user_profile_weight",
            "source_file",
        ],
    )

    compact = {
        "num_metric_files": len(
            [
                path
                for path in Path(args.results_root).glob("*/*_metrics.json")
                if path.parent.name != "summary"
                and "smoke" not in path.parent.name.lower()
                and "multiseed" not in path.parent.name.lower()
                and "sensitivity" not in path.parent.name.lower()
            ]
        ),
        "num_rows_all_splits": len(rows),
        "num_rows_test": len(test_rows),
        "outputs": {
            "all_metrics_long": str(all_metrics_path),
            "paper_test_metrics": str(output_dir / "paper_test_metrics.csv"),
            "best_test_by_dataset_split": str(output_dir / "best_test_by_dataset_split.csv"),
            "best_regression_by_dataset_split_window": str(
                output_dir / "best_regression_by_dataset_split_window.csv"
            ),
            "v1_pheme_window_comparison": str(output_dir / "v1_pheme_window_comparison.csv"),
            "v1_pheme_ablation": str(output_dir / "v1_pheme_ablation.csv"),
            "v1_pheme_180_split_comparison": str(
                output_dir / "v1_pheme_180_split_comparison.csv"
            ),
            "v1_pheme_text_encoder_comparison": str(
                output_dir / "v1_pheme_text_encoder_comparison.csv"
            ),
            "v1_rumdetect2017_180_split_comparison": str(
                output_dir / "v1_rumdetect2017_180_split_comparison.csv"
            ),
            "v1_rumdetect2017_180_split_comparison_md": str(
                output_dir / "v1_rumdetect2017_180_split_comparison.md"
            ),
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
