import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_CASE_DIR = DEFAULT_RESULTS_ROOT / "case_studies"
DEFAULT_FIGURE_DIR = DEFAULT_RESULTS_ROOT / "figures"

DATASETS = ["pheme", "twitter15", "twitter16"]
CASE_KINDS = [
    ("early_warning", "Early warning success"),
    ("control_gain", "High-control gain"),
    ("challenge", "False-alarm challenge"),
]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_2": "#E9A6A1",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "orange": "#D9853B",
}


def finite_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite_int(value: Any, default: int = 0) -> int:
    return int(round(finite_float(value, default)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def load_snapshots(data_root: Path, dataset: str) -> dict[str, list[dict[str, str]]]:
    rows = read_csv(data_root / dataset / "dynamic_snapshots" / "snapshots.csv")
    grouped = group_by(rows, "sample_id")
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: finite_int(row.get("window_index")))
    return grouped


def prepare_window_scores(args: argparse.Namespace) -> Path:
    from train_c2_breakout import (
        aggregate_sample_scores,
        build_examples,
        group_by_sample,
        make_models,
        positive_scores,
        read_csv as c2_read_csv,
        select_threshold,
        split_ids,
    )

    output_rows: list[dict[str, Any]] = []
    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    for dataset in args.datasets:
        dataset_dir = data_root / dataset
        snapshots = group_by_sample(c2_read_csv(dataset_dir / "dynamic_snapshots" / "snapshots.csv"))
        breakouts = {row["sample_id"]: row for row in c2_read_csv(dataset_dir / "breakout_events.csv")}
        c2_args = SimpleNamespace(
            data_root=str(data_root),
            split_root=str(split_root),
            split_file=None,
            task=args.task,
            split_seed=args.split_seed,
            seed=args.seed,
            max_windows_per_sample=args.max_windows_per_sample,
        )
        split_payloads = {}
        for split in ("train", "val", "test"):
            ids = split_ids(dataset, args.split_strategy, split, c2_args)
            x_rows, y_rows, meta_rows, _feature_names = build_examples(
                ids,
                snapshots,
                breakouts,
                "heterorumor_c2",
                args.max_windows_per_sample,
            )
            split_payloads[split] = (x_rows, y_rows, meta_rows)

        model = make_models(args.seed, "heterorumor_c2")["heterorumor_c2"]
        x_train, y_train, _train_meta = split_payloads["train"]
        x_val, y_val, val_meta = split_payloads["val"]
        x_test, y_test, test_meta = split_payloads["test"]
        model.fit(x_train, y_train)

        val_scores = positive_scores(model, x_val)
        val_sample_rows = aggregate_sample_scores(val_meta, y_val, val_scores, threshold=0.5)
        threshold = select_threshold(
            [int(row["label_id"]) for row in val_sample_rows],
            [float(row["score_label_1"]) for row in val_sample_rows],
        )

        test_scores = positive_scores(model, x_test)
        for meta, label, score in zip(test_meta, y_test, test_scores):
            output_rows.append(
                {
                    "dataset": dataset,
                    "split_strategy": args.split_strategy,
                    "seed": args.seed,
                    "model": "heterorumor_c2",
                    "sample_id": meta["sample_id"],
                    "label_id": int(label),
                    "window_index": meta["window_index"],
                    "window_start": meta["window_start"],
                    "window_end": meta["window_end"],
                    "score_label_1": float(score),
                    "threshold": threshold,
                    "breakout_window": meta.get("breakout_window", ""),
                    "breakout_time": meta.get("breakout_time", ""),
                }
            )

    path = Path(args.case_dir) / "c2_c3_case_window_scores.csv"
    write_csv(
        path,
        output_rows,
        [
            "dataset",
            "split_strategy",
            "seed",
            "model",
            "sample_id",
            "label_id",
            "window_index",
            "window_start",
            "window_end",
            "score_label_1",
            "threshold",
            "breakout_window",
            "breakout_time",
        ],
    )
    return path


def load_c2_predictions(results_root: Path, dataset: str, seed: int, split_strategy: str) -> dict[str, dict[str, str]]:
    path = results_root / "c2_breakout" / f"{dataset}_breakout_{split_strategy}_seed{seed}_predictions.csv"
    rows = [
        row
        for row in read_csv(path)
        if row.get("split") == "test" and row.get("model") == "heterorumor_c2"
    ]
    return {row["sample_id"]: row for row in rows}


def load_c3_simulations(results_root: Path, dataset: str, seed: int, split_strategy: str) -> dict[str, dict[str, dict[str, str]]]:
    path = results_root / "c3_control" / f"{dataset}_control_{split_strategy}_heterorumor_c2_seed{seed}_simulations.csv"
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(path):
        grouped[row["sample_id"]][row["strategy"]] = row
    return dict(grouped)


def choose_max(candidates: list[dict[str, Any]], key) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=key)


def select_cases(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root)
    all_rows: list[dict[str, Any]] = []
    per_dataset: dict[str, tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]] = {}
    for dataset in args.datasets:
        per_dataset[dataset] = (
            load_c2_predictions(results_root, dataset, args.seed, args.split_strategy),
            load_c3_simulations(results_root, dataset, args.seed, args.split_strategy),
        )

    def enrich(dataset: str, sample_id: str, kind: str, title: str) -> dict[str, Any]:
        c2_rows, c3_rows = per_dataset[dataset]
        pred = c2_rows[sample_id]
        sims = c3_rows.get(sample_id, {})
        full = sims.get("heterorumor_c3_event_pulse", {})
        random_budget = sims.get("random_same_budget", {})
        fixed_budget = sims.get("fixed_same_budget", {})
        return {
            "case_kind": kind,
            "case_title": title,
            "dataset": dataset,
            "sample_id": sample_id,
            "label_id": pred.get("label_id", ""),
            "pred_label_id": pred.get("pred_label_id", ""),
            "score_label_1": pred.get("score_label_1", ""),
            "first_warning_window": pred.get("first_warning_window", ""),
            "first_warning_time": pred.get("first_warning_time", ""),
            "breakout_window": pred.get("breakout_window", ""),
            "breakout_time": pred.get("breakout_time", ""),
            "lead_time_minutes": pred.get("lead_time_minutes", ""),
            "full_suppression_rate": full.get("suppression_rate", ""),
            "full_cost": full.get("cost", ""),
            "random_same_budget_suppression_rate": random_budget.get("suppression_rate", ""),
            "fixed_same_budget_suppression_rate": fixed_budget.get("suppression_rate", ""),
            "full_trigger_window": full.get("trigger_window", ""),
            "full_effective_window": full.get("effective_window", ""),
            "full_pulse_strength": full.get("pulse_strength", ""),
        }

    # Case 1: a clear PHEME early-warning success.
    c2_rows, c3_rows = per_dataset.get("pheme", ({}, {}))
    candidates = []
    for sample_id, pred in c2_rows.items():
        full = c3_rows.get(sample_id, {}).get("heterorumor_c3_event_pulse", {})
        if finite_int(pred.get("label_id")) == 1 and finite_int(pred.get("pred_label_id")) == 1 and finite_float(pred.get("lead_time_minutes")) > 0:
            candidates.append({"sample_id": sample_id, "pred": pred, "full": full})
    selected = choose_max(
        candidates,
        lambda row: (
            -abs(finite_float(row["pred"].get("lead_time_minutes")) - 360.0),
            finite_float(row["full"].get("suppression_rate")),
            finite_float(row["pred"].get("score_label_1")),
        ),
    )
    if selected:
        all_rows.append(enrich("pheme", selected["sample_id"], "early_warning", "Early warning success"))

    # Case 2: a Twitter15 sample where risk-aware control beats same-budget random control.
    c2_rows, c3_rows = per_dataset.get("twitter15", ({}, {}))
    candidates = []
    for sample_id, pred in c2_rows.items():
        sims = c3_rows.get(sample_id, {})
        full = sims.get("heterorumor_c3_event_pulse", {})
        random_budget = sims.get("random_same_budget", {})
        if finite_int(pred.get("pred_label_id")) == 1 and finite_int(full.get("triggered")) == 1:
            candidates.append({"sample_id": sample_id, "pred": pred, "full": full, "random": random_budget})
    selected = choose_max(
        candidates,
        lambda row: (
            finite_float(row["full"].get("suppression_rate")) - finite_float(row["random"].get("suppression_rate")),
            finite_float(row["full"].get("suppression_rate")),
            finite_float(row["pred"].get("score_label_1")),
        ),
    )
    if selected:
        all_rows.append(enrich("twitter15", selected["sample_id"], "control_gain", "High-control gain"))

    # Case 3: a Twitter16 false alarm, used as a limitation/challenge example.
    c2_rows, c3_rows = per_dataset.get("twitter16", ({}, {}))
    candidates = []
    for sample_id, pred in c2_rows.items():
        if finite_int(pred.get("label_id")) == 0 and finite_int(pred.get("pred_label_id")) == 1:
            full = c3_rows.get(sample_id, {}).get("heterorumor_c3_event_pulse", {})
            candidates.append({"sample_id": sample_id, "pred": pred, "full": full})
    selected = choose_max(
        candidates,
        lambda row: (
            finite_float(row["pred"].get("score_label_1")),
            finite_float(row["full"].get("cost")),
        ),
    )
    if selected:
        all_rows.append(enrich("twitter16", selected["sample_id"], "challenge", "False-alarm challenge"))
    else:
        fallback = []
        for dataset, (c2_rows, c3_rows) in per_dataset.items():
            for sample_id, pred in c2_rows.items():
                if finite_int(pred.get("label_id")) == 1 and finite_int(pred.get("pred_label_id")) == 0:
                    fallback.append({"dataset": dataset, "sample_id": sample_id, "pred": pred})
        selected = choose_max(fallback, lambda row: finite_float(row["pred"].get("score_label_1")))
        if selected:
            all_rows.append(enrich(selected["dataset"], selected["sample_id"], "challenge", "Missed-warning challenge"))

    path = Path(args.case_dir) / "c2_c3_selected_cases.csv"
    write_csv(
        path,
        all_rows,
        [
            "case_kind",
            "case_title",
            "dataset",
            "sample_id",
            "label_id",
            "pred_label_id",
            "score_label_1",
            "first_warning_window",
            "first_warning_time",
            "breakout_window",
            "breakout_time",
            "lead_time_minutes",
            "full_suppression_rate",
            "full_cost",
            "random_same_budget_suppression_rate",
            "fixed_same_budget_suppression_rate",
            "full_trigger_window",
            "full_effective_window",
            "full_pulse_strength",
        ],
    )
    return path


def controlled_curve(
    rows: list[dict[str, str]],
    effective_window: int | None,
    strength: float,
    effect_multiplier: float,
) -> list[float]:
    controlled = 1.0
    prev_cumulative = 1.0
    values = []
    for row in rows:
        window = finite_int(row.get("window_index"))
        current = finite_float(row.get("cumulative_nodes"), prev_cumulative)
        observed_new = max(0.0, current - prev_cumulative)
        if effective_window is not None and window >= effective_window:
            reduction = min(max(strength * effect_multiplier, 0.0), 0.95)
            kept_new = observed_new * (1.0 - reduction)
        else:
            kept_new = observed_new
        controlled += kept_new
        values.append(max(1.0, controlled))
        prev_cumulative = current
    return values


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.3,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig, figure_dir: Path, basename: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = figure_dir / f"{basename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        paths.append(path)
    return paths


def strategy_lookup(rows: list[dict[str, str]], strategy: str) -> dict[str, str]:
    return next((row for row in rows if row.get("strategy") == strategy), {})


def effect_multiplier(results_root: Path, dataset: str, seed: int, split_strategy: str) -> float:
    path = results_root / "c3_control" / f"{dataset}_control_{split_strategy}_heterorumor_c2_seed{seed}_metrics.json"
    if not path.exists():
        return 0.85
    return finite_float(read_json(path).get("effect_multiplier"), 0.85)


def plot_cases(args: argparse.Namespace) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    case_dir = Path(args.case_dir)
    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    selected_cases = read_csv(case_dir / "c2_c3_selected_cases.csv")
    window_scores = read_csv(case_dir / "c2_c3_case_window_scores.csv")
    scores_by_sample = group_by(window_scores, "sample_id")

    fig, axes = plt.subplots(2, len(selected_cases), figsize=(5.2 * len(selected_cases), 6.7), sharex=False)
    if len(selected_cases) == 1:
        axes = [[axes[0]], [axes[1]]]

    for idx, case in enumerate(selected_cases):
        dataset = case["dataset"]
        sample_id = case["sample_id"]
        snapshots = load_snapshots(data_root, dataset).get(sample_id, [])
        score_rows = sorted(scores_by_sample.get(sample_id, []), key=lambda row: finite_int(row.get("window_index")))
        threshold = finite_float(score_rows[0].get("threshold"), 0.5) if score_rows else 0.5

        sims = load_c3_simulations(results_root, dataset, args.seed, args.split_strategy).get(sample_id, {})
        sim_rows = list(sims.values())
        full_row = strategy_lookup(sim_rows, "heterorumor_c3_event_pulse")
        effective_time = None
        if full_row.get("effective_window", "") != "":
            effective_window = finite_int(full_row.get("effective_window"))
            effective_time = next(
                (
                    finite_float(row.get("window_start"))
                    for row in snapshots
                    if finite_int(row.get("window_index")) == effective_window
                ),
                None,
            )

        marker_times = [finite_float(row.get("window_start")) for row in score_rows]
        if case.get("first_warning_time", "") != "":
            marker_times.append(finite_float(case.get("first_warning_time")))
        if case.get("breakout_time", "") != "":
            marker_times.append(finite_float(case.get("breakout_time")))
        if effective_time is not None:
            marker_times.append(effective_time)
        final_time = max([finite_float(row.get("window_start")) for row in snapshots], default=0.0)
        plot_until = min(final_time, max(360.0, max(marker_times, default=0.0) + 240.0))
        snapshots_plot = [row for row in snapshots if finite_float(row.get("window_start")) <= plot_until]
        if not snapshots_plot and snapshots:
            snapshots_plot = snapshots[:1]
        snapshot_x = [finite_float(row.get("window_start")) for row in snapshots_plot]
        observed = [finite_float(row.get("cumulative_nodes")) for row in snapshots_plot]
        score_rows_plot = [row for row in score_rows if finite_float(row.get("window_start")) <= plot_until]
        score_x = [finite_float(row.get("window_start")) for row in score_rows_plot]
        score_y = [finite_float(row.get("score_label_1")) for row in score_rows_plot]

        ax = axes[0][idx]
        ax.plot(score_x, score_y, marker="o", color=PALETTE["blue_main"], linewidth=2.2, label="C2 risk")
        ax.axhline(threshold, color=PALETTE["red_strong"], linestyle="--", linewidth=1.4, label="Threshold")
        warning_time = finite_float(case.get("first_warning_time"), None)
        if case.get("first_warning_time", "") != "":
            ax.axvline(warning_time, color=PALETTE["orange"], linestyle="-.", linewidth=1.4, label="First warning")
        if case.get("breakout_time", "") != "":
            ax.axvline(finite_float(case.get("breakout_time")), color=PALETTE["violet"], linestyle=":", linewidth=1.8, label="Breakout")
        ax.set_ylim(0.0, 1.03)
        ax.set_xlim(left=min(0.0, min(score_x, default=0.0)), right=max(plot_until, max(score_x, default=0.0) + 30.0))
        ax.set_ylabel("Risk score" if idx == 0 else "")
        ax.set_title(
            f"{chr(ord('A') + idx)}. {case['case_title']}\n{dataset}, id={sample_id[-6:]}",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.18)

        multiplier = effect_multiplier(results_root, dataset, args.seed, args.split_strategy)

        ax2 = axes[1][idx]
        ax2.plot(snapshot_x, observed, color="#333333", linewidth=2.2, label="Observed")
        strategies = [
            ("heterorumor_c3_event_pulse", "HeteroRumorDyn-C3", PALETTE["blue_main"]),
            ("random_same_budget", "Random same-budget", PALETTE["red_2"]),
            ("fixed_same_budget", "Fixed same-budget", PALETTE["orange"]),
        ]
        curve_rows: list[dict[str, Any]] = []
        for strategy, label, color in strategies:
            row = strategy_lookup(sim_rows, strategy)
            effective_raw = row.get("effective_window", "")
            effective = finite_int(effective_raw) if effective_raw != "" else None
            strength = finite_float(row.get("pulse_strength"))
            curve = controlled_curve(snapshots_plot, effective, strength, multiplier)
            ax2.plot(snapshot_x, curve, linewidth=2.0, label=label, color=color)
            for x_value, y_value in zip(snapshot_x, curve):
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "case_kind": case.get("case_kind"),
                        "strategy": strategy,
                        "window_start": x_value,
                        "controlled_cumulative_nodes": y_value,
                    }
                )
        if effective_time is not None and effective_time <= plot_until:
            ax2.axvline(effective_time, color=PALETTE["blue_secondary"], linestyle="--", linewidth=1.2)
        ax2.set_xlim(left=min(0.0, min(snapshot_x, default=0.0)), right=max(plot_until, max(snapshot_x, default=0.0) + 30.0))
        ax2.set_xlabel("Time since source post (min)")
        ax2.set_ylabel("Cumulative nodes" if idx == 0 else "")
        ax2.set_title(f"{chr(ord('D') + idx)}. Control trajectory", loc="left", fontweight="bold")
        ax2.grid(axis="y", alpha=0.18)

    handles, labels = [], []
    for ax in [*axes[0], *axes[1]]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=1.0)
    paths = save_figure(fig, Path(args.figure_dir), "fig12_c2_c3_case_studies")
    plt.close(fig)

    write_case_notes(case_dir, selected_cases)
    return paths


def write_case_notes(case_dir: Path, selected_cases: list[dict[str, str]]) -> Path:
    lines = [
        "# C2/C3 Case Study Notes",
        "",
        "The selected examples are intended for qualitative explanation of the V3 experiments.",
        "",
    ]
    for case in selected_cases:
        label = "breakout" if finite_int(case.get("label_id")) == 1 else "non-breakout"
        prediction = "warned" if finite_int(case.get("pred_label_id")) == 1 else "not warned"
        lines.append(
            f"- {case['case_title']} ({case['dataset']}, sample {case['sample_id']}): "
            f"ground truth={label}, prediction={prediction}, score={finite_float(case.get('score_label_1')):.4f}, "
            f"lead time={case.get('lead_time_minutes') or 'NA'} min, "
            f"C3 suppression={finite_float(case.get('full_suppression_rate')):.4f}, "
            f"same-budget random suppression={finite_float(case.get('random_same_budget_suppression_rate')):.4f}."
        )
    lines.extend(
        [
            "",
            "Reporting caution: the third case is a limitation example. Use it to explain why high-risk structural signals can trigger unnecessary intervention when the cascade looks like a breakout but does not cross the breakout label threshold.",
            "",
        ]
    )
    path = case_dir / "c2_c3_case_study_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def percent(value: Any) -> str:
    return f"{finite_float(value) * 100:.2f}%"


def minutes(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{finite_float(value):.0f}"


def write_case_report(results_root: Path, figure_dir: Path, selected_cases: list[dict[str, str]]) -> Path:
    draft_dir = results_root / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "fig12_c2_c3_case_studies.png"

    by_kind = {case.get("case_kind", ""): case for case in selected_cases}
    early = by_kind.get("early_warning", {})
    gain = by_kind.get("control_gain", {})
    challenge = by_kind.get("challenge", {})

    lines = [
        "# C2/C3 案例分析汇报稿",
        "",
        "## 可以直接汇报的结论",
        "",
        (
            "C2/C3 的案例图不是为了再次证明平均指标，而是补充解释模型在单个传播事件上的行为："
            "C2 先给出破圈风险，C3 再根据风险触发控制。三个案例分别对应“提前预警有效”、"
            "“风险感知控制收益明显”和“高风险误报边界”。"
        ),
        "",
        f"- 论文图：`{figure_path.as_posix()}`",
        f"- 案例明细：`{(results_root / 'case_studies' / 'c2_c3_selected_cases.csv').as_posix()}`",
        "",
        "## 案例 1：PHEME 早预警成功",
        "",
        (
            f"PHEME 样本 `{early.get('sample_id', '')}` 是真实破圈事件。C2 在 "
            f"{minutes(early.get('first_warning_time'))} 分钟触发首次预警，真实破圈发生在 "
            f"{minutes(early.get('breakout_time'))} 分钟，因此获得约 "
            f"{minutes(early.get('lead_time_minutes'))} 分钟提前量。该样本的风险分数为 "
            f"{finite_float(early.get('score_label_1')):.4f}，高于阈值。"
        ),
        "",
        (
            f"C3 在该样本上的抑制率为 {percent(early.get('full_suppression_rate'))}，"
            f"而同预算随机干预为 {percent(early.get('random_same_budget_suppression_rate'))}。"
            "这说明在早期风险信号比较明确时，事件触发式控制可以把预算集中到关键增长阶段。"
        ),
        "",
        "## 案例 2：Twitter15 控制收益明显",
        "",
        (
            f"Twitter15 样本 `{gain.get('sample_id', '')}` 的风险分数达到 "
            f"{finite_float(gain.get('score_label_1')):.4f}，C2 在 "
            f"{minutes(gain.get('first_warning_time'))} 分钟预警，破圈发生在 "
            f"{minutes(gain.get('breakout_time'))} 分钟。虽然提前量只有 "
            f"{minutes(gain.get('lead_time_minutes'))} 分钟，但风险判断非常明确。"
        ),
        "",
        (
            f"该样本中 HeteroRumorDyn-C3 抑制率为 {percent(gain.get('full_suppression_rate'))}，"
            f"固定同预算策略为 {percent(gain.get('fixed_same_budget_suppression_rate'))}，"
            f"随机同预算策略为 {percent(gain.get('random_same_budget_suppression_rate'))}。"
            "这适合在汇报中说明：C3 的价值不只是“花预算”，而是根据 C2 风险把干预落到更有效的时间点。"
        ),
        "",
        "## 案例 3：Twitter16 误报挑战",
        "",
        (
            f"Twitter16 样本 `{challenge.get('sample_id', '')}` 是非破圈事件，但 C2 给出了 "
            f"{finite_float(challenge.get('score_label_1')):.4f} 的高风险分数并触发预警。"
            "这个案例应该主动作为局限性讲出来：当传播结构和增长曲线看起来接近破圈模式时，"
            "模型可能会对最终没有越过破圈阈值的事件进行不必要干预。"
        ),
        "",
        (
            f"该样本中 C3 抑制率为 {percent(challenge.get('full_suppression_rate'))}，"
            f"同预算随机干预为 {percent(challenge.get('random_same_budget_suppression_rate'))}。"
            "两者接近，说明在误报样本上，风险感知控制未必优于简单同预算策略。"
            "后续可以通过不确定性校准、二阶段阈值、人机复核或更真实的社区标签降低误报成本。"
        ),
        "",
        "## 汇报时建议这样说",
        "",
        "- C2 的强项是早期风险发现，C3 的强项是把风险分数转化成同预算下更有效的控制策略。",
        "- 案例图中第三列要主动解释为边界案例，不能把所有高风险预警都说成成功。",
        "- Weibo 当前缺少真实时间戳和真实社区，因此 C2/C3 的正式论证优先使用 PHEME、Twitter15、Twitter16。",
        "- 后续改进方向是加入风险校准和误报成本约束，使控制策略在高不确定样本上更保守。",
        "",
    ]
    path = draft_dir / "c2_c3_case_analysis_for_report.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "plot", "report", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--case-dir", default=str(DEFAULT_CASE_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--split-root", default=str(DEFAULT_DATA_ROOT / "splits"))
    parser.add_argument("--task", default="rumor_binary")
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows-per-sample", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    result: dict[str, Any] = {}
    if args.mode in {"prepare", "both"}:
        result["window_scores"] = str(prepare_window_scores(args))
        result["selected_cases"] = str(select_cases(args))
    if args.mode in {"plot", "both"}:
        result["figures"] = [str(path) for path in plot_cases(args)]
        result["notes"] = str(Path(args.case_dir) / "c2_c3_case_study_notes.md")
    if args.mode in {"plot", "report", "both"}:
        selected_path = Path(args.case_dir) / "c2_c3_selected_cases.csv"
        result["case_report"] = str(
            write_case_report(
                Path(args.results_root),
                Path(args.figure_dir),
                read_csv(selected_path),
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
