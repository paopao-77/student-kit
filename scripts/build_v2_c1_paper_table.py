import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(".")
SUMMARY_DIR = ROOT / "results" / "summary"
DRAFT_DIR = ROOT / "results" / "drafts"

V1_SUMMARY = SUMMARY_DIR / "v1_plm_multiseed_summary.csv"
V2_SELECTED = (
    ROOT
    / "results"
    / "heterorumor_v2_c1_selected"
    / "pheme_cascade_size_stratified_heterorumor_v2_c1_vae_k4_multilingual_minilm_selected_obs180_seed42_metrics.json"
)
V2_DISENT_SUMMARY = SUMMARY_DIR / "v2_c1_disentangled_multiseed_summary.csv"

CSV_OUT = SUMMARY_DIR / "paper_v2_c1_main_table.csv"
MD_OUT = SUMMARY_DIR / "paper_v2_c1_main_table.md"
TEX_OUT = SUMMARY_DIR / "paper_v2_c1_main_table.tex"
DRAFT_OUT = DRAFT_DIR / "v2_c1_results_paragraph.md"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}%"


def metric_pm(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return fmt(mean, digits)
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def find_v1_full() -> dict[str, str]:
    for row in read_csv_rows(V1_SUMMARY):
        if row["model"] == "heterorumor_v1_hurdle_multilingual_minilm":
            return row
    raise ValueError(f"V1 full MiniLM row not found in {V1_SUMMARY}")


def load_old_v2() -> dict[str, Any]:
    with V2_SELECTED.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    model_type = payload["model_type"]
    test = payload["models"][model_type]["test"]
    robustness = payload.get("robustness", {})
    text_noise_03 = robustness.get("text_noise_0.3", {})
    return {
        "payload": payload,
        "test": test,
        "text_noise_0.3_mape": f(text_noise_03.get("mape")),
    }


def find_disentangled() -> dict[str, str]:
    rows = read_csv_rows(V2_DISENT_SUMMARY)
    if not rows:
        raise ValueError(f"No rows found in {V2_DISENT_SUMMARY}")
    return rows[0]


def build_rows() -> list[dict[str, str]]:
    v1 = find_v1_full()
    old_v2 = load_old_v2()
    disent = find_disentangled()

    v1_mape = f(v1["mape_mean"])
    old_test = old_v2["test"]

    rows = [
        {
            "row_id": "M1",
            "method": "V1 HeteroRumorDyn + MiniLM",
            "role": "strong multimodal baseline",
            "dataset": v1["dataset"],
            "split": v1["split_strategy"],
            "obs_min": v1["observation_window_minutes"],
            "seeds": v1["seeds"],
            "n_seeds": v1["num_seeds"],
            "mape": fmt(f(v1["mape_mean"])),
            "mape_std": fmt(f(v1["mape_std"])),
            "mape_report": metric_pm(f(v1["mape_mean"]), f(v1["mape_std"])),
            "mae": fmt(f(v1["mae_mean"])),
            "rmse": fmt(f(v1["rmse_mean"])),
            "r2": fmt(f(v1["r2_mean"])),
            "best_val_mape": "",
            "relative_mape_gain_vs_v1": "0.00%",
            "latent_factors": "",
            "content_factors": "",
            "text_noise_0.3_mape": "",
            "matched_swap_mape": "",
            "matched_swap_delta_mape": "",
            "paper_note": "Five-seed reference model with pretrained MiniLM text features.",
        },
        {
            "row_id": "M2",
            "method": "V2/C1 VAE factors",
            "role": "validation-selected predictive V2",
            "dataset": old_v2["payload"]["dataset"],
            "split": old_v2["payload"]["split_strategy"],
            "obs_min": str(old_v2["payload"]["observation_window_minutes"]),
            "seeds": str(old_v2["payload"]["seed"]),
            "n_seeds": "1",
            "mape": fmt(f(old_test["mape"])),
            "mape_std": "",
            "mape_report": metric_pm(f(old_test["mape"]), None),
            "mae": fmt(f(old_test["mae"])),
            "rmse": fmt(f(old_test["rmse"])),
            "r2": fmt(f(old_test["r2"])),
            "best_val_mape": fmt(f(old_v2["payload"]["best_val_mape"])),
            "relative_mape_gain_vs_v1": pct(
                100.0 * (v1_mape - f(old_test["mape"])) / v1_mape
            ),
            "latent_factors": str(old_test.get("active_latent_factors", "")),
            "content_factors": "",
            "text_noise_0.3_mape": fmt(old_v2["text_noise_0.3_mape"]),
            "matched_swap_mape": "",
            "matched_swap_delta_mape": "",
            "paper_note": "Single-seed selected model; best seed-42 predictive V2 result, not yet a multi-seed estimate.",
        },
        {
            "row_id": "M3",
            "method": "V2/C1 disentangled + matched-swap CF",
            "role": "interpretable and robustness-oriented V2",
            "dataset": disent["dataset"],
            "split": disent["split_strategy"],
            "obs_min": disent["observation_window_minutes"],
            "seeds": disent["seeds"],
            "n_seeds": disent["num_seeds"],
            "mape": fmt(f(disent["mape_mean"])),
            "mape_std": fmt(f(disent["mape_std"])),
            "mape_report": metric_pm(f(disent["mape_mean"]), f(disent["mape_std"])),
            "mae": fmt(f(disent["mae_mean"])),
            "rmse": fmt(f(disent["rmse_mean"])),
            "r2": fmt(f(disent["r2_mean"])),
            "best_val_mape": fmt(f(disent["best_val_mape_mean"])),
            "relative_mape_gain_vs_v1": pct(
                100.0 * (v1_mape - f(disent["mape_mean"])) / v1_mape
            ),
            "latent_factors": fmt(f(disent["active_latent_factors_mean"]), 1),
            "content_factors": fmt(f(disent["active_content_factors_mean"]), 1),
            "text_noise_0.3_mape": fmt(f(disent["text_noise_0.3_mape_mean"])),
            "matched_swap_mape": fmt(f(disent["matched_text_swap_mape_mean"])),
            "matched_swap_delta_mape": fmt(
                f(disent["matched_text_swap_delta_mape_mean"])
            ),
            "paper_note": "Five-seed model with explicit content/dynamics separation and target-matched text intervention.",
        },
    ]
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]]) -> None:
    columns = [
        "method",
        "n_seeds",
        "mape_report",
        "mae",
        "rmse",
        "r2",
        "relative_mape_gain_vs_v1",
        "text_noise_0.3_mape",
        "matched_swap_mape",
        "paper_note",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [
        "# Table. PHEME 180-min cascade-size prediction under the stratified split",
        "",
        "Lower MAPE, MAE and RMSE are better; higher R2 is better. Values with +/- report mean +/- standard deviation across seeds. The V2/C1 VAE row is a validation-selected single-seed result and should be interpreted as a seed-42 reference.",
        "",
        header,
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: list[dict[str, str]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\caption{PHEME 180-min cascade-size prediction under the stratified split. Lower MAPE, MAE and RMSE are better; higher $R^2$ is better.}",
        "\\label{tab:v2-c1-main}",
        "\\centering",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Seeds & MAPE $\\downarrow$ & MAE $\\downarrow$ & RMSE $\\downarrow$ & $R^2$ $\\uparrow$ & Gain vs V1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"].replace("_", "\\_")
        mape_report = row["mape_report"].replace("+/-", "$\\pm$")
        lines.append(
            f"{method} & {row['n_seeds']} & {mape_report} & {row['mae']} & {row['rmse']} & {row['r2']} & {row['relative_mape_gain_vs_v1']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Note: the V2/C1 VAE factors row is single-seed (seed=42), whereas V1 and the disentangled V2/C1 model are five-seed estimates.",
        ]
    )
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_results_draft(rows: list[dict[str, str]]) -> None:
    v1, old_v2, disent = rows
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# V2/C1 Results Paragraph Draft

## Intended Claim

V2/C1 adds a variational propagation-factor layer on top of the V1 multimodal model. The strongest seed-42 V2 model improves predictive accuracy, while the redesigned disentangled V2 model provides a more defensible interpretation of content factors and propagation-dynamics factors under a multi-seed protocol.

## Draft

To examine whether the multimodal representations learned by HeteroRumorDyn can be compressed into interpretable propagation-momentum factors, we evaluated two V2/C1 variants on the PHEME 180-min cascade-size prediction task under the stratified split (Table~\\ref{{tab:v2-c1-main}}). The MiniLM-enhanced V1 model provided a strong multimodal baseline, reaching a five-seed test MAPE of {v1['mape_report']}. A validation-selected V2/C1 VAE with four latent factors reduced the seed-42 test MAPE to {old_v2['mape']}, corresponding to a {old_v2['relative_mape_gain_vs_v1']} relative reduction against the V1 multi-seed mean, and all four latent factors remained active. This result indicates that the fused text, topology, temporal and user representations contain a compact low-dimensional signal predictive of future cascade growth, although the single-seed nature of this result means it should be interpreted as the strongest tuned V2 reference rather than as a fully replicated estimate.

We therefore further evaluated a disentangled V2/C1 variant that separates text-derived content factors from topology-, temporal- and user-derived dynamics factors and trains with a target-matched text-swap counterfactual constraint. Across five random seeds, this model achieved a test MAPE of {disent['mape_report']}, slightly improving on the V1 multi-seed baseline while preserving stable performance under 30% text-feature perturbation (MAPE {disent['text_noise_0.3_mape']}). The matched-swap intervention remained a harder stress test, increasing MAPE to {disent['matched_swap_mape']} (delta {disent['matched_swap_delta_mape']}), suggesting that content replacement can still shift predictions even when samples are matched by propagation target. Overall, V2/C1 supports the existence of compact propagation-momentum factors and provides a clearer separation between content and dynamics, but the current evidence is best framed as interpretability and robustness support rather than a definitive causal claim.

## 中文说明

- 主结果不要只说“V2 全面超过 V1”。更稳妥的说法是：旧 V2 在 seed=42 上预测最好，新 V2 disentangled 在五随机种子下略优于 V1，并提供更好的机制解释。
- “反事实”这里要写成 stress test / robustness support，不要写成 causal proof。
- 表格中必须标出 seed 数：旧 V2 是 n=1，V1 和 V2 disentangled 是 n=5。

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| V2/C1 learns compact propagation-momentum factors. | Selected V2 uses four active latent factors and obtains test MAPE {old_v2['mape']}. | supported |
| Disentanglement gives a more interpretable factor structure. | New V2 separates dynamics factors and content factors; five-seed row reports active dynamics/content factors in the CSV. | supported as model/diagnostic evidence |
| Counterfactual training improves causal robustness. | Matched-swap MAPE is {disent['matched_swap_mape']}, higher than clean MAPE {disent['mape']}. | not supported as strong causal gain |
| V2/C1 improves predictive performance over V1. | Seed-42 selected V2 improves more; five-seed disentangled V2 improves slightly over V1. | supported with boundary |
"""
    DRAFT_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_markdown(rows)
    write_latex(rows)
    write_results_draft(rows)
    print(
        json.dumps(
            {
                "csv": str(CSV_OUT),
                "markdown": str(MD_OUT),
                "latex": str(TEX_OUT),
                "draft": str(DRAFT_OUT),
                "rows": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
