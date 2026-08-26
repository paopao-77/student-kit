import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


PALETTE = {
    "blue": "#0F4D92",
    "red": "#B64342",
    "neutral": "#CFCECE",
}


def read_predictions(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    factor_names = sorted(
        [name for name in rows[0] if name.startswith("factor_")],
        key=lambda name: int(name.split("_")[1]),
    )
    if not factor_names:
        raise ValueError(f"No latent factors found in {path}")
    factors = np.asarray(
        [[float(row[name]) for name in factor_names] for row in rows], dtype=np.float64
    )
    return rows, factors


def save_plot_data(path: Path, rows: list[dict[str, str]], embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "raw_label", "final_size", "observed_size", "tsne_1", "tsne_2"],
        )
        writer.writeheader()
        for row, point in zip(rows, embedding):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "raw_label": row["raw_label"],
                    "final_size": row["final_size"],
                    "observed_size": row["observed_size"],
                    "tsne_1": float(point[0]),
                    "tsne_2": float(point[1]),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=(
            "results/heterorumor_v2_c1/"
            "pheme_cascade_size_stratified_heterorumor_v2_c1_vae_k16_"
            "multilingual_minilm_obs180_seed42_predictions.csv"
        ),
    )
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, factors = read_predictions(Path(args.predictions))
    final_size = np.asarray([float(row["final_size"]) for row in rows])
    observed_size = np.asarray([float(row["observed_size"]) for row in rows])
    log_growth = np.log1p(np.maximum(final_size - observed_size, 0.0))
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
        max_iter=1000,
    ).fit_transform(factors)
    correlations = np.asarray(
        [
            np.corrcoef(factors[:, index], log_growth)[0, 1]
            if factors[:, index].std() > 1e-12 and log_growth.std() > 1e-12
            else 0.0
            for index in range(factors.shape[1])
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    scatter = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=np.log1p(final_size),
        cmap="viridis",
        s=10,
        alpha=0.72,
        linewidths=0,
    )
    axes[0].set_title("a  VAE propagation-momentum factors", loc="left", fontweight="bold")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("log(1 + final cascade size)")

    order = np.argsort(np.abs(correlations))[::-1]
    labels = [f"z{index + 1}" for index in order]
    values = correlations[order]
    colors = [PALETTE["blue"] if value >= 0 else PALETTE["red"] for value in values]
    y = np.arange(len(values))
    axes[1].barh(y, values, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.9)
    axes[1].set_xlabel("Pearson r with log remaining growth")
    axes[1].set_title("b  Factor-growth association", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#E5E5E5", linewidth=0.8)

    fig.subplots_adjust(wspace=0.38)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_dir / "fig6_v2_c1_latent_factors"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(basename.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    save_plot_data(output_dir / "plot_data_v2_c1_tsne.csv", rows, embedding)
    with (output_dir / "plot_data_v2_c1_factor_correlations.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["factor", "pearson_r_log_remaining_growth"])
        for index in order:
            writer.writerow([f"z{index + 1}", float(correlations[index])])
    print(str(basename.with_suffix(".png")))


if __name__ == "__main__":
    main()
