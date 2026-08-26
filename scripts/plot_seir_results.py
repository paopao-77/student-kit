import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUMMARY_DIR = Path("results/summary")
DEFAULT_OUTPUT_DIR = Path("results/figures")
SPLIT_ORDER = ["stratified", "temporal"]
SPLIT_LABELS = {
    "stratified": "Stratified",
    "temporal": "Temporal",
}
MODEL_LABELS = {
    "early_count": "Early count",
    "sir": "SIR",
    "seir": "SEIR",
}
MODEL_ORDER = ["early_count", "sir", "seir"]
COLORS = {
    "early_count": "#4C78A8",
    "sir": "#F58518",
    "seir": "#54A24B",
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


def normalize_model(model_name: str) -> str:
    if model_name.startswith("early_count"):
        return "early_count"
    if model_name.startswith("sir"):
        return "sir"
    if model_name.startswith("seir"):
        return "seir"
    return model_name


def read_seir_rows(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "paper_test_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}. Run scripts/summarize_results.py first.")
    df = pd.read_csv(path)
    df = df[
        (df["dataset"] == "pheme")
        & (df["task"] == "cascade_size_prediction")
        & (df["model_family"] == "dynamics_seir")
    ].copy()
    if df.empty:
        raise ValueError("No PHEME SEIR rows found. Run scripts/train_seir_baseline.py first.")

    df["model_type"] = df["model"].map(normalize_model)
    for column in ("observation_window_minutes", "mae", "rmse", "mape", "r2"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["split_strategy"] = pd.Categorical(df["split_strategy"], categories=SPLIT_ORDER, ordered=True)
    df["model_type"] = pd.Categorical(df["model_type"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["split_strategy", "model_type", "observation_window_minutes"])


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def plot_pheme_size_prediction(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    metrics = [("mape", "MAPE"), ("mae", "MAE")]

    for col_idx, split in enumerate(SPLIT_ORDER):
        split_df = df[df["split_strategy"] == split]
        for row_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for model_type in MODEL_ORDER:
                model_df = split_df[split_df["model_type"] == model_type]
                if model_df.empty:
                    continue
                ax.plot(
                    model_df["observation_window_minutes"],
                    model_df[metric],
                    marker="o",
                    linewidth=1.8,
                    color=COLORS[model_type],
                    label=MODEL_LABELS[model_type],
                )
            ax.set_title(f"{SPLIT_LABELS[split]} split")
            ax.set_ylabel(label)
            ax.set_xticks([60, 180, 360])
            ax.set_xticklabels(["60m", "180m", "360m"])
            if row_idx == 1:
                ax.set_xlabel("Observation window")
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, loc="upper right")

    fig.suptitle("PHEME early cascade-size prediction with SIR/SEIR baselines", y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig4_pheme_seir_size_prediction")


def write_tables_and_notes(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "plot_data_pheme_seir_size_prediction.csv"
    df.to_csv(table_path, index=False, encoding="utf-8-sig")

    best_rows = (
        df.sort_values("mape", ascending=True)
        .groupby(["split_strategy", "observation_window_minutes"], observed=True)
        .head(1)
        .sort_values(["split_strategy", "observation_window_minutes"])
    )
    lines = [
        "# PHEME SEIR Baseline Notes",
        "",
        "## Best Test MAPE by Observation Window",
        "",
        "| Split | Window | Best model | MAPE | MAE | R2 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"| {SPLIT_LABELS[str(row.split_strategy)]} | {row.observation_window_minutes:.0f}m | "
            f"{MODEL_LABELS[str(row.model_type)]} | {row.mape:.4f} | {row.mae:.4f} | {row.r2:.4f} |"
        )
    notes_path = output_dir / "seir_analysis_notes.md"
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"plot_data": str(table_path), "notes": str(notes_path)}


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
    df = read_seir_rows(Path(args.summary_dir))
    outputs = {
        "fig4": plot_pheme_size_prediction(df, output_dir),
        "tables": write_tables_and_notes(df, output_dir),
    }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
