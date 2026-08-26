import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY_DIR = Path("results/summary")
DEFAULT_OUTPUT_DIR = Path("results/figures")
DATASET_ORDER = ["weibo", "twitter15", "twitter16", "pheme"]
SPLIT_ORDER = ["stratified", "temporal"]
FAMILY_LABELS = {
    "structure_stats": "Structure statistics",
    "propagation_graph": "Propagation graph",
}
DATASET_LABELS = {
    "weibo": "Weibo",
    "twitter15": "Twitter15",
    "twitter16": "Twitter16",
    "pheme": "PHEME",
}
SPLIT_LABELS = {
    "stratified": "Stratified",
    "temporal": "Temporal",
}
COLORS = {
    "structure_stats": "#4C78A8",
    "propagation_graph": "#F58518",
    "gain": "#54A24B",
    "loss": "#B279A2",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_metrics(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "paper_test_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}. Run scripts/summarize_results.py first.")
    df = pd.read_csv(path)
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df[df["model_family"].isin(FAMILY_LABELS)].dropna(subset=["macro_f1"]).copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["split_strategy"] = pd.Categorical(df["split_strategy"], categories=SPLIT_ORDER, ordered=True)
    return df


def best_by_family(df: pd.DataFrame) -> pd.DataFrame:
    idx = (
        df.sort_values("macro_f1", ascending=False)
        .groupby(["dataset", "split_strategy", "model_family"], observed=True)["macro_f1"]
        .idxmax()
    )
    return df.loc[idx].sort_values(["dataset", "split_strategy", "model_family"]).reset_index(drop=True)


def best_overall(df: pd.DataFrame) -> pd.DataFrame:
    idx = (
        df.sort_values("macro_f1", ascending=False)
        .groupby(["dataset", "split_strategy"], observed=True)["macro_f1"]
        .idxmax()
    )
    return df.loc[idx].sort_values(["dataset", "split_strategy"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def plot_family_comparison(best_family: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    bar_width = 0.34
    x = np.arange(len(DATASET_ORDER))

    for ax, split in zip(axes, SPLIT_ORDER):
        split_df = best_family[best_family["split_strategy"] == split]
        for offset, family in zip((-bar_width / 2, bar_width / 2), ("structure_stats", "propagation_graph")):
            values = []
            labels = []
            for dataset in DATASET_ORDER:
                row = split_df[(split_df["dataset"] == dataset) & (split_df["model_family"] == family)]
                if row.empty:
                    values.append(np.nan)
                    labels.append("")
                else:
                    values.append(float(row.iloc[0]["macro_f1"]))
                    labels.append(str(row.iloc[0]["model"]))
            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=FAMILY_LABELS[family],
                color=COLORS[family],
                edgecolor="white",
                linewidth=0.7,
            )
            for bar, model_name in zip(bars, labels):
                if not model_name:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(f"{SPLIT_LABELS[split]} split")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], rotation=20, ha="right")
        ax.set_ylim(0.35, 0.95)
        ax.set_ylabel("Test macro-F1")
        ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Propagation graph features improve test macro-F1 on most datasets", y=1.03)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig1_macro_f1_family_comparison")


def plot_graph_gain(best_family: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    rows = []
    for dataset in DATASET_ORDER:
        for split in SPLIT_ORDER:
            subset = best_family[(best_family["dataset"] == dataset) & (best_family["split_strategy"] == split)]
            structure = subset[subset["model_family"] == "structure_stats"]
            graph = subset[subset["model_family"] == "propagation_graph"]
            if structure.empty or graph.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "split_strategy": split,
                    "structure_macro_f1": float(structure.iloc[0]["macro_f1"]),
                    "graph_macro_f1": float(graph.iloc[0]["macro_f1"]),
                    "macro_f1_gain": float(graph.iloc[0]["macro_f1"]) - float(structure.iloc[0]["macro_f1"]),
                }
            )
    gain_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    labels = [f"{DATASET_LABELS[r.dataset]}\n{SPLIT_LABELS[r.split_strategy]}" for r in gain_df.itertuples()]
    x = np.arange(len(gain_df))
    colors = [COLORS["gain"] if value >= 0 else COLORS["loss"] for value in gain_df["macro_f1_gain"]]
    bars = ax.bar(x, gain_df["macro_f1_gain"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Macro-F1 gain over structure baseline")
    ax.set_title("Graph baseline gain by dataset and split")
    for bar in bars:
        value = bar.get_height()
        va = "bottom" if value >= 0 else "top"
        y = value + 0.01 if value >= 0 else value - 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.2f}", ha="center", va=va, fontsize=7)
    fig.tight_layout()

    gain_df.to_csv(output_dir / "plot_data_graph_gain.csv", index=False, encoding="utf-8-sig")
    return save_figure(fig, output_dir, "fig2_graph_gain")


def plot_split_robustness(best_rows: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = np.arange(len(SPLIT_ORDER))
    for dataset in DATASET_ORDER:
        subset = best_rows[best_rows["dataset"] == dataset].sort_values("split_strategy")
        if len(subset) != 2:
            continue
        values = [float(subset[subset["split_strategy"] == split].iloc[0]["macro_f1"]) for split in SPLIT_ORDER]
        ax.plot(x, values, marker="o", linewidth=1.8, label=DATASET_LABELS[dataset])
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_ylim(0.50, 0.92)
    ax.set_ylabel("Best test macro-F1")
    ax.set_title("Best model robustness from stratified to temporal split")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig3_split_robustness")


def write_analysis_tables(best_family: pd.DataFrame, best_rows: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_family_path = output_dir / "plot_data_best_by_family.csv"
    best_overall_path = output_dir / "plot_data_best_overall.csv"
    best_family.to_csv(best_family_path, index=False, encoding="utf-8-sig")
    best_rows.to_csv(best_overall_path, index=False, encoding="utf-8-sig")
    return {"best_by_family": str(best_family_path), "best_overall": str(best_overall_path)}


def build_analysis_summary(best_family: pd.DataFrame, best_rows: pd.DataFrame, output_dir: Path) -> str:
    lines = [
        "# Result Analysis Notes",
        "",
        "## Main Reading",
        "",
        "- Use `paper_test_metrics.csv` for the full test table.",
        "- Use `best_test_by_dataset_split.csv` to report the strongest baseline per dataset and split.",
        "- Use `fig1_macro_f1_family_comparison` to compare structure statistics against propagation graph features.",
        "- Use `fig2_graph_gain` to show the macro-F1 gain from graph propagation.",
        "- Use `fig3_split_robustness` to discuss whether model ranking survives temporal evaluation.",
        "",
        "## Best Test Macro-F1",
        "",
        "| Dataset | Split | Model family | Model | Macro-F1 | AUC |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[str(row.dataset)]} | {SPLIT_LABELS[str(row.split_strategy)]} | "
            f"{FAMILY_LABELS.get(str(row.model_family), row.model_family)} | `{row.model}` | "
            f"{float(row.macro_f1):.4f} | {float(row.auc):.4f} |"
        )

    lines.extend(["", "## Graph-vs-Structure Gain", ""])
    for dataset in DATASET_ORDER:
        for split in SPLIT_ORDER:
            subset = best_family[(best_family["dataset"] == dataset) & (best_family["split_strategy"] == split)]
            structure = subset[subset["model_family"] == "structure_stats"]
            graph = subset[subset["model_family"] == "propagation_graph"]
            if structure.empty or graph.empty:
                continue
            gain = float(graph.iloc[0]["macro_f1"]) - float(structure.iloc[0]["macro_f1"])
            lines.append(f"- {DATASET_LABELS[dataset]} / {SPLIT_LABELS[split]}: graph gain = {gain:+.4f}.")

    path = output_dir / "analysis_notes.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default=str(DEFAULT_SUMMARY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = parse_args()
    setup_matplotlib()
    output_dir = Path(args.output_dir)
    df = read_metrics(Path(args.summary_dir))
    family_df = best_by_family(df)
    overall_df = best_overall(df)

    outputs = {
        "fig1": plot_family_comparison(family_df, output_dir),
        "fig2": plot_graph_gain(family_df, output_dir),
        "fig3": plot_split_robustness(overall_df, output_dir),
        "tables": write_analysis_tables(family_df, overall_df, output_dir),
        "notes": build_analysis_summary(family_df, overall_df, output_dir),
    }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
