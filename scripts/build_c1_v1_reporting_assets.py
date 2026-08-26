import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_DIR = Path("results/summary")
FIGURE_DIR = Path("results/figures")
DRAFT_DIR = Path("results/drafts")

V2_C1_TABLE = SUMMARY_DIR / "paper_v2_c1_main_table.csv"
TEXT_FAIRNESS = SUMMARY_DIR / "v1_rumdetect2017_text_fairness.csv"

TABLE_CSV = SUMMARY_DIR / "c1_v1_paper_table.csv"
TABLE_MD = SUMMARY_DIR / "c1_v1_paper_table.md"
TABLE_TEX = SUMMARY_DIR / "c1_v1_paper_table.tex"
DRAFT_OUT = DRAFT_DIR / "c1_v1_rumdetect2017_results_explanation.md"

FIGURE_BASENAME = "fig9_rumdetect2017_text_fairness"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_2": "#E9A6A1",
    "neutral": "#CFCECE",
    "dark": "#272727",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def fmt(value: float | str | None, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    return f"{float(value):.{digits}f}"


def fmt_pm(mean: float, std: float | None, digits: int = 4) -> str:
    if std is None or np.isnan(std):
        return fmt(mean, digits)
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.linewidth": 1.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_text_fairness(rows: list[dict[str, str]]) -> list[str]:
    publication_style()
    datasets = ["twitter15_rumdetect2017", "twitter16_rumdetect2017"]
    labels = ["Twitter15\nrumdetect2017", "Twitter16\nrumdetect2017"]
    method_keys = ["hash", "minilm", "no_text"]
    method_labels = ["Hash text", "MiniLM text", "w/o text"]
    colors = [PALETTE["neutral"], PALETTE["blue_main"], PALETTE["red_2"]]
    hatches = ["", "", "//"]

    by_dataset = {row["dataset"]: row for row in rows}
    means = np.asarray(
        [
            [to_float(by_dataset[dataset][f"{key}_mape_mean"]) for dataset in datasets]
            for key in method_keys
        ]
    )
    stds = np.asarray(
        [
            [to_float(by_dataset[dataset][f"{key}_mape_std"]) for dataset in datasets]
            for key in method_keys
        ]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.24
    for index, (key, label, color, hatch) in enumerate(
        zip(method_keys, method_labels, colors, hatches)
    ):
        offset = (index - 1) * width
        bars = ax.bar(
            x + offset,
            means[index],
            width,
            yerr=stds[index],
            capsize=4,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=1.1,
            hatch=hatch,
        )
        for bar, value in zip(bars, means[index]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=PALETTE["dark"],
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test MAPE")
    ax.set_title("A  Text encoder fairness comparison", loc="left", fontweight="bold")
    y_min = float(np.nanmin(means - stds)) - 0.008
    y_max = float(np.nanmax(means + stds)) + 0.018
    ax.set_ylim(max(y_min, 0.15), y_max)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=3, fontsize=9)

    ax = axes[1]
    delta_labels = ["MiniLM - Hash", "w/o text - MiniLM"]
    deltas = np.asarray(
        [
            [to_float(by_dataset[dataset]["minilm_minus_hash_mape"]) for dataset in datasets],
            [to_float(by_dataset[dataset]["no_text_minus_minilm_mape"]) for dataset in datasets],
        ]
    )
    y = np.arange(len(delta_labels))
    bar_height = 0.32
    dataset_colors = [PALETTE["blue_secondary"], PALETTE["green_3"]]
    for index, (dataset_label, color) in enumerate(zip(labels, dataset_colors)):
        offset = (index - 0.5) * bar_height
        bars = ax.barh(
            y + offset,
            deltas[:, index],
            bar_height,
            label=dataset_label.replace("\n", " "),
            color=color,
            edgecolor="black",
            linewidth=1.1,
        )
        for bar, value in zip(bars, deltas[:, index]):
            ax.text(
                value + (0.00035 if value >= 0 else -0.00035),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.4f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
    ax.axvline(0.0, color="black", linewidth=1.1)
    ax.set_yticks(y, delta_labels)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE difference")
    ax.set_title("B  Paired-condition mean deltas", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.0085, 0.0075)
    ax.text(
        0.5,
        -0.18,
        "Negative is better for the left condition; positive w/o-text delta means text helps.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
    )

    for ax in axes:
        ax.tick_params(axis="both", width=1.5, length=4)

    fig.tight_layout(pad=1.2, w_pad=2.0)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ["png", "pdf", "svg"]:
        path = FIGURE_DIR / f"{FIGURE_BASENAME}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def build_table_rows(v2_rows: list[dict[str, str]], fairness_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in v2_rows:
        rows.append(
            {
                "block": "C1/PHEME",
                "dataset": row["dataset"],
                "split": row["split"],
                "obs_min": row["obs_min"],
                "method": row["method"],
                "role": row["role"],
                "seeds": row["seeds"],
                "n_seeds": row["n_seeds"],
                "mape": row["mape_report"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "key_note": row["paper_note"],
            }
        )

    method_specs = [
        ("V1 HeteroRumorDyn + hash text", "Hash text baseline", "hash"),
        ("V1 HeteroRumorDyn + MiniLM", "Pretrained text encoder", "minilm"),
        ("V1 HeteroRumorDyn w/o text", "Text ablation", "no_text"),
    ]
    for source in fairness_rows:
        for method, role, prefix in method_specs:
            rows.append(
                {
                    "block": "V1/RumDetect2017",
                    "dataset": source["dataset"],
                    "split": source["split_strategy"],
                    "obs_min": source["observation_window_minutes"],
                    "method": method,
                    "role": role,
                    "seeds": source[f"{prefix}_seeds"],
                    "n_seeds": source[f"num_{prefix}"],
                    "mape": fmt_pm(
                        to_float(source[f"{prefix}_mape_mean"]),
                        to_float(source[f"{prefix}_mape_std"]),
                    ),
                    "mae": fmt(to_float(source[f"{prefix}_mae_mean"])),
                    "rmse": fmt(to_float(source[f"{prefix}_rmse_mean"])),
                    "r2": fmt(to_float(source[f"{prefix}_r2_mean"])),
                    "key_note": "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_marker(rows: list[dict[str, str]], block: str, dataset: str, value: str) -> str:
    if block != "V1/RumDetect2017":
        return value
    scoped = [
        row
        for row in rows
        if row["block"] == block
        and row["dataset"] == dataset
        and row["mape"]
        and "+/-" in row["mape"]
    ]
    if not scoped:
        return value
    parsed = [(row, to_float(row["mape"].split("+/-")[0].strip())) for row in scoped]
    best = min(metric for _, metric in parsed)
    second_values = sorted({metric for _, metric in parsed})
    second = second_values[1] if len(second_values) > 1 else None
    current = to_float(value.split("+/-")[0].strip()) if "+/-" in value else None
    if current == best:
        return f"**{value}**"
    if second is not None and current == second:
        return f"<u>{value}</u>"
    return value


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["block", "dataset", "method", "n_seeds", "mape", "mae", "rmse", "r2", "role"]
    lines = [
        "# C1/V1 paper table",
        "",
        "Lower MAPE, MAE and RMSE are better; higher R2 is better. Bold marks the lowest MAPE within each RumDetect2017 dataset block, and underline marks the second lowest.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered = dict(row)
        rendered["mape"] = best_marker(rows, row["block"], row["dataset"], row["mape"])
        lines.append("| " + " | ".join(rendered[col] for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_latex(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\caption{C1/V1 cascade-size prediction and text fairness results. Lower MAPE, MAE and RMSE are better; higher $R^2$ is better.}",
        "\\label{tab:c1-v1-main}",
        "\\centering",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Block & Dataset & Method & Seeds & MAPE $\\downarrow$ & MAE $\\downarrow$ & RMSE $\\downarrow$ & $R^2$ $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["block"]),
                    tex_escape(row["dataset"]),
                    tex_escape(row["method"]),
                    tex_escape(row["n_seeds"]),
                    tex_escape(row["mape"]).replace("+/-", "$\\pm$"),
                    tex_escape(row["mae"]),
                    tex_escape(row["rmse"]),
                    tex_escape(row["r2"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_explanation(path: Path, fairness_rows: list[dict[str, str]]) -> None:
    by_dataset = {row["dataset"]: row for row in fairness_rows}
    t15 = by_dataset["twitter15_rumdetect2017"]
    t16 = by_dataset["twitter16_rumdetect2017"]
    text = f"""# C1/V1 RumDetect2017 Results Explanation

## 论文结果解释段落

为检验 V1 多模态框架在新增 RumDetect2017 Twitter15/16 数据集上的可迁移性，并进一步分析文本语义编码的边际贡献，我们在 180 分钟早期传播窗口下进行了统一的五随机种子复验。结果显示，MiniLM 文本编码器在两个数据集上相较 hash 文本特征均降低了平均 MAPE：Twitter15 从 {fmt(to_float(t15['hash_mape_mean']))} 降至 {fmt(to_float(t15['minilm_mape_mean']))}，相对下降 {pct(to_float(t15['minilm_relative_mape_reduction_vs_hash_pct']))}；Twitter16 从 {fmt(to_float(t16['hash_mape_mean']))} 降至 {fmt(to_float(t16['minilm_mape_mean']))}，相对下降 {pct(to_float(t16['minilm_relative_mape_reduction_vs_hash_pct']))}。其中 Twitter16 的文本收益更稳定，MiniLM 在 4/5 个随机种子上优于 hash，且去除文本后 MAPE 增加 {fmt(to_float(t16['no_text_minus_minilm_mape']))}，配对 t-test 的 p 值为 {fmt(to_float(t16['text_ablation_paired_ttest_p']))}，说明源推文语义对早期传播规模预测具有可观贡献。

相比之下，Twitter15 的文本贡献呈现指标依赖性：MiniLM 相比 hash 的平均 MAPE 仅小幅下降 {fmt(abs(to_float(t15['minilm_minus_hash_mape'])))}，而 w/o text 的 MAPE 反而低于完整 MiniLM 模型 {fmt(abs(to_float(t15['no_text_minus_minilm_mape'])))}。但该消融模型的 MAE、RMSE 和 R2 分别退化到 {fmt(to_float(t15['no_text_mae_mean']))}、{fmt(to_float(t15['no_text_rmse_mean']))} 和 {fmt(to_float(t15['no_text_r2_mean']))}，弱于完整 MiniLM 的 {fmt(to_float(t15['minilm_mae_mean']))}、{fmt(to_float(t15['minilm_rmse_mean']))} 和 {fmt(to_float(t15['minilm_r2_mean']))}。这表明 Twitter15 上文本模态可能并未稳定改善相对误差，但有助于绝对级联规模拟合；该数据集中的早期时序与拓扑信号已经解释了较大比例的传播增长。

## 汇报 PPT 版本

- 新增 RumDetect2017 Twitter15/16 后，V1 多模态模型在两个数据集上都能稳定跑通 180 分钟早期传播规模预测。
- MiniLM 相比 hash 文本特征整体更优，Twitter16 上最明显：MAPE 从 {fmt(to_float(t16['hash_mape_mean']))} 降到 {fmt(to_float(t16['minilm_mape_mean']))}，去掉文本后 5/5 seeds 都变差。
- Twitter15 上文本增益不稳定：MAPE 指标下 w/o text 略优，但 MAE/RMSE/R2 更差，说明它更像“保守预测”，不是整体更强。
- 结论写法要稳：文本语义对 Twitter16 和绝对规模拟合有明确帮助；Twitter15 上传播时序/拓扑信号可能主导，文本贡献需要结合多指标解释。

## 可引用文件

- Figure: `results/figures/{FIGURE_BASENAME}.png`
- Table: `results/summary/c1_v1_paper_table.csv`
- Text fairness source: `results/summary/v1_rumdetect2017_text_fairness.csv`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    v2_rows = read_csv(V2_C1_TABLE)
    fairness_rows = read_csv(TEXT_FAIRNESS)
    figure_outputs = plot_text_fairness(fairness_rows)
    table_rows = build_table_rows(v2_rows, fairness_rows)
    write_csv(TABLE_CSV, table_rows)
    write_markdown(TABLE_MD, table_rows)
    write_latex(TABLE_TEX, table_rows)
    write_explanation(DRAFT_OUT, fairness_rows)
    print(
        json.dumps(
            {
                "figure_outputs": figure_outputs,
                "table_csv": str(TABLE_CSV),
                "table_md": str(TABLE_MD),
                "table_tex": str(TABLE_TEX),
                "draft": str(DRAFT_OUT),
                "table_rows": len(table_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
