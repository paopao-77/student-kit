import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(".")
OUTPUT = ROOT / "experiment_workflow_status.md"


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def count_files(pattern: str) -> int:
    return len(list(ROOT.glob(pattern)))


def read_json_file(path: str) -> dict[str, Any]:
    file_path = ROOT / path
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def weibo_raw_v1_ready() -> bool:
    metadata = read_json_file("data/processed/v1_inputs/weibo/obs_180events_metadata.json")
    coverage = metadata.get("modality_coverage", {})
    return (
        int(metadata.get("num_samples", 0)) >= 5000
        and float(coverage.get("text", 0.0)) > 0.0
        and float(coverage.get("user_profile", 0.0)) > 0.0
    )


def weibo_c2_preferred_foundation_ready() -> bool:
    stats = read_json_file("data/processed/weibo/c2_foundation_stats.json")
    return (
        int(stats.get("order_window_size", 0)) == 50
        and float(stats.get("theta_cross", -1.0)) == 0.2
        and float(stats.get("theta_branch_ratio", -1.0)) == 0.2
    )


def mark(value: bool) -> str:
    return "x" if value else " "


def status_text(value: bool) -> str:
    return "已完成" if value else "未完成"


def build_status() -> dict[str, Any]:
    processed_datasets = all(
        exists(f"data/processed/{dataset}/samples.csv")
        and exists(f"data/processed/{dataset}/edges.csv")
        and exists(f"data/processed/{dataset}/events.csv")
        for dataset in ["weibo", "twitter15", "twitter16", "pheme"]
    )
    return {
        "processed_datasets": processed_datasets,
        "split_files": count_files("data/processed/splits/*_split.json"),
        "label_map": exists("label_map.json"),
        "loader": exists("dataset_loader.py"),
        "structure_baselines": count_files("results/baseline/*_metrics.json"),
        "graph_baselines": count_files("results/graph_baseline/*_metrics.json"),
        "seir_script": exists("scripts/train_seir_baseline.py"),
        "seir_metrics": count_files("results/seir_baseline/*_metrics.json"),
        "summary_tables": all(
            exists(path)
            for path in [
                "results/summary/all_metrics_long.csv",
                "results/summary/paper_test_metrics.csv",
                "results/summary/best_test_by_dataset_split.csv",
            ]
        ),
        "classification_figures": all(
            exists(path)
            for path in [
                "results/figures/fig1_macro_f1_family_comparison.png",
                "results/figures/fig2_graph_gain.png",
                "results/figures/fig3_split_robustness.png",
            ]
        ),
        "seir_figure": exists("results/figures/fig4_pheme_seir_size_prediction.png"),
        "dynamic_snapshots": count_files("data/processed/*/dynamic_snapshots/*") > 0,
        "communities": count_files("data/processed/*/community_ids.csv") > 0,
        "breakout_events": count_files("data/processed/*/breakout_events.csv") > 0,
        "v1_inputs": count_files("data/processed/v1_inputs/*/obs_*.npz"),
        "v1_loader": exists("v1_dataset.py"),
        "v1_train_script": exists("scripts/train_heterorumor_v1.py"),
        "v1_metrics": (
            count_files("results/heterorumor_v1/*_metrics.json")
            + count_files("results/heterorumor_v1_hurdle/*_metrics.json")
            + count_files("results/heterorumor_v1_temporal/*_metrics.json")
            + count_files("results/heterorumor_v1_rumdetect2017/*_metrics.json")
        ),
        "v1_temporal_180": exists(
            "results/heterorumor_v1_temporal/"
            "pheme_cascade_size_temporal_heterorumor_v1_obs180_seed42_metrics.json"
        ),
        "plm_text_cache": exists(
            "data/processed/v1_text_features/pheme/multilingual_minilm.npz"
        ),
        "plm_multiseed_summary": exists("results/summary/v1_plm_multiseed_summary.csv"),
        "plm_multiseed_paired": exists(
            "results/summary/v1_plm_multiseed_paired_text.csv"
        ),
        "v2_c1_selected": exists(
            "results/heterorumor_v2_c1_selected/"
            "pheme_cascade_size_stratified_heterorumor_v2_c1_vae_k4_"
            "multilingual_minilm_selected_obs180_seed42_metrics.json"
        ),
        "v2_c1_factor_figure": exists("results/figures/fig6_v2_c1_latent_factors.png"),
        "v2_c1_k_sensitivity": exists("results/summary/v2_c1_k_sensitivity.csv"),
        "v2_c1_kl_sensitivity": exists("results/summary/v2_c1_kl_sensitivity.csv"),
        "v2_c1_cf_robustness": exists(
            "results/summary/v2_c1_counterfactual_robustness.csv"
        ),
        "v2_c1_sensitivity_figure": exists(
            "results/figures/fig7_v2_c1_sensitivity_and_counterfactual.png"
        ),
        "v2_disentangled_model": exists("models/heterorumor_v2_c1_disentangled.py"),
        "v2_disentangled_runs": count_files(
            "results/heterorumor_v2_c1_disentangled_multiseed/*_metrics.json"
        ),
        "v2_disentangled_summary": exists(
            "results/summary/v2_c1_disentangled_multiseed_summary.csv"
        ),
        "v2_disentangled_figure": exists(
            "results/figures/fig8_v2_disentangled_multiseed.png"
        ),
        "v2_c1_paper_table": exists("results/summary/paper_v2_c1_main_table.csv"),
        "v2_c1_results_draft": exists("results/drafts/v2_c1_results_paragraph.md"),
        "rumdetect2017_processed": all(
            exists(f"data/processed/{dataset}/{file_name}")
            for dataset in ["twitter15_rumdetect2017", "twitter16_rumdetect2017"]
            for file_name in ["samples.csv", "events.csv", "edges.csv", "stats.json"]
        ),
        "rumdetect2017_splits": count_files(
            "data/processed/splits/twitter*_rumdetect2017_*_split.json"
        ),
        "rumdetect2017_v1_inputs": count_files(
            "data/processed/v1_inputs/twitter*_rumdetect2017/obs_180m.npz"
        ),
        "rumdetect2017_v1_180": count_files(
            "results/heterorumor_v1_rumdetect2017/"
            "twitter*_rumdetect2017_cascade_size_*_heterorumor_v1_hurdle_obs180_seed42_metrics.json"
        ),
        "rumdetect2017_plm_features": count_files(
            "data/processed/v1_text_features/twitter*_rumdetect2017/multilingual_minilm.npz"
        ),
        "rumdetect2017_plm_multiseed_runs": count_files(
            "results/heterorumor_v1_rumdetect2017_plm_multiseed/"
            "twitter*_rumdetect2017_cascade_size_stratified_"
            "heterorumor_v1_hurdle_multilingual_minilm_obs180_seed*_metrics.json"
        ),
        "rumdetect2017_hash_multiseed_runs": count_files(
            "results/heterorumor_v1_rumdetect2017_hash_multiseed/"
            "twitter*_rumdetect2017_cascade_size_stratified_"
            "heterorumor_v1_hurdle_obs180_seed*_metrics.json"
        ),
        "rumdetect2017_plm_wo_text_runs": count_files(
            "results/heterorumor_v1_rumdetect2017_plm_multiseed/"
            "twitter*_rumdetect2017_cascade_size_stratified_"
            "heterorumor_v1_hurdle_multilingual_minilm_wo_text_obs180_seed*_metrics.json"
        ),
        "rumdetect2017_plm_multiseed_summary": exists(
            "results/summary/v1_rumdetect2017_plm_multiseed_summary.csv"
        ),
        "rumdetect2017_text_fairness_table": exists(
            "results/summary/v1_rumdetect2017_text_fairness.csv"
        ),
        "c1_v1_paper_table": exists("results/summary/c1_v1_paper_table.csv"),
        "rumdetect2017_text_fairness_figure": exists(
            "results/figures/fig9_rumdetect2017_text_fairness.png"
        ),
        "c1_v1_results_explanation": exists(
            "results/drafts/c1_v1_rumdetect2017_results_explanation.md"
        ),
        "c2_breakout_runs": count_files("results/c2_breakout/*_breakout_*_metrics.json"),
        "c2_breakout_multiseed_summary": exists(
            "results/summary/c2_breakout_multiseed_summary.csv"
        ),
        "c2_breakout_paper_table": exists(
            "results/summary/c2_breakout_paper_table.csv"
        ),
        "c2_breakout_figure": exists(
            "results/figures/fig10_c2_breakout_multiseed.png"
        ),
        "c3_control_runs": count_files("results/c3_control/*_control_*_metrics.json"),
        "c3_control_multiseed_summary": exists(
            "results/summary/c3_control_multiseed_summary.csv"
        ),
        "c3_control_paper_table": exists(
            "results/summary/c3_control_paper_table.csv"
        ),
        "c3_control_figure": exists(
            "results/figures/fig11_c3_control_multiseed.png"
        ),
        "c2_c3_temporal_table": exists(
            "results/summary/c2_c3_temporal_seed42_table.csv"
        ),
        "c2_c3_results_draft": exists(
            "results/drafts/c2_c3_results_explanation.md"
        ),
        "c2_c3_case_figure": exists(
            "results/figures/fig12_c2_c3_case_studies.png"
        ),
        "c2_c3_case_report": exists(
            "results/drafts/c2_c3_case_analysis_for_report.md"
        ),
        "paper_baselines_mapping": exists(
            "results/summary/paper_baselines_mapping.csv"
        ),
        "paper_dynamics_summary": exists(
            "results/summary/paper_dynamics_baseline_table.csv"
        ),
        "paper_midpms_summary": exists(
            "results/summary/paper_midpms_adapted_table.csv"
        ),
        "paper_dshcl_summary": exists(
            "results/summary/paper_dshcl_adapted_table.csv"
        ),
        "paper_edid_summary": exists(
            "results/summary/paper_ed_id_adapted_table.csv"
        ),
        "paper_inf_vae_script": exists("scripts/train_inf_vae_adapted_baseline.py"),
        "paper_inf_vae_runs": count_files(
            "results/paper_baselines/fair180/inf_vae/*_metrics.json"
        ),
        "paper_inf_vae_integrated": exists(
            "results/summary/paper_v1_fair180_main_table.csv"
        )
        and exists("results/summary/paper_v1_fair180_temporal_table.csv"),
        "v1_fair180_main_table": exists(
            "results/summary/paper_v1_fair180_main_table.csv"
        ),
        "v1_fair180_temporal_table": exists(
            "results/summary/paper_v1_fair180_temporal_table.csv"
        ),
        "v1_fair180_audit": exists(
            "results/summary/v1_fair180_fairness_audit.csv"
        ),
        "v1_fair180_bootstrap": exists(
            "results/summary/paper_v1_fair180_paired_bootstrap.csv"
        ),
        "weibo_v1_input_180": exists(
            "data/processed/v1_inputs/weibo/obs_180events.npz"
        ),
        "weibo_raw_v1_ready": weibo_raw_v1_ready(),
        "weibo_raw_smoke": count_files(
            "results/heterorumor_v1_weibo_raw_smoke/"
            "weibo_cascade_size_stratified_heterorumor_v1_hurdle_obs180_seed*_metrics.json"
        ),
        "weibo_v1_runs": count_files(
            "results/heterorumor_v1_weibo_multiseed/"
            "weibo_cascade_size_stratified_heterorumor_v1_hurdle_obs180_seed*_metrics.json"
        ),
        "weibo_v1_summary": exists("results/summary/v1_weibo_multiseed_summary.csv"),
        "weibo_v2_selected_runs": count_files(
            "results/heterorumor_v2_c1_weibo_selected_multiseed/"
            "weibo_cascade_size_stratified_heterorumor_v2_c1_vae_k4_weibo_selected_obs180_seed*_metrics.json"
        ),
        "weibo_v2_selected_summary": exists(
            "results/summary/v2_c1_weibo_selected_multiseed_summary.csv"
        ),
        "weibo_raw_c2_runs": count_files(
            "results/c2_breakout_weibo_raw/weibo_breakout_stratified_seed*_metrics.json"
        ),
        "weibo_raw_c2_summary": exists(
            "results/summary/c2_breakout_weibo_raw_summary.csv"
        ),
        "weibo_raw_c3_runs": count_files(
            "results/c3_control_weibo_raw/weibo_control_stratified_heterorumor_c2_seed*_metrics.json"
        ),
        "weibo_raw_c3_summary": exists(
            "results/summary/c3_control_weibo_raw_summary.csv"
        ),
        "weibo_raw_c2_c3_note": exists(
            "results/drafts/weibo_raw_c2_c3_experiment_note.md"
        ),
        "weibo_raw_c2_c3_order_window_sensitivity": exists(
            "results/summary/weibo_raw_c2_c3_order_window_sensitivity.csv"
        )
        and exists("results/drafts/weibo_raw_c2_c3_order_window_sensitivity.md"),
        "weibo_raw_c2_c3_threshold_sensitivity": exists(
            "results/summary/weibo_raw_c2_c3_threshold_sensitivity.csv"
        )
        and exists("results/drafts/weibo_raw_c2_c3_threshold_sensitivity.md"),
        "weibo_raw_c2_threshold_audit": exists(
            "results/summary/weibo_raw_c2_threshold_distribution.csv"
        )
        and exists("results/summary/weibo_raw_c2_threshold_label_flip_audit.csv")
        and exists("results/summary/weibo_raw_c2_threshold_condition_hits.csv")
        and exists("results/drafts/weibo_raw_c2_threshold_audit.md"),
        "weibo_raw_c2_c3_preferred": exists(
            "results/summary/c2_breakout_weibo_raw_preferred_summary.csv"
        )
        and exists("results/summary/c3_control_weibo_raw_preferred_summary.csv")
        and exists("results/drafts/weibo_raw_c2_c3_preferred_setting.md")
        and exists("scripts/promote_weibo_raw_c2_c3_preferred.py")
        and weibo_c2_preferred_foundation_ready(),
        "weibo_raw_preferred_artifact_validation": exists(
            "results/summary/weibo_raw_preferred_artifact_validation.csv"
        )
        and exists("results/summary/weibo_raw_preferred_artifact_map.csv")
        and exists("results/drafts/weibo_raw_preferred_artifact_validation.md")
        and exists("scripts/validate_weibo_raw_preferred_artifact.py"),
        "weibo_raw_reporting_entrypoints": exists(
            "results/summary/weibo_raw_reporting_entrypoints.csv"
        )
        and exists("results/summary/weibo_raw_reporting_entrypoints_validation.csv")
        and exists("results/drafts/weibo_raw_reporting_entrypoints.md")
        and exists("scripts/build_weibo_raw_reporting_entrypoints.py"),
        "weibo_raw_efficiency_benchmark": exists(
            "results/summary/weibo_raw_efficiency_summary.csv"
        )
        and exists("results/summary/weibo_raw_v1_efficiency_runs.csv")
        and exists("results/summary/weibo_raw_v2_c1_efficiency_runs.csv")
        and exists("results/drafts/weibo_raw_efficiency_benchmark.md")
        and exists("results/efficiency_benchmark/weibo_raw_c2_c3_benchmark_commands.json")
        and exists("scripts/benchmark_weibo_raw_efficiency.py"),
        "weibo_raw_external_holdout_validation": exists(
            "results/summary/v1_weibo_external_holdout_summary.csv"
        )
        and exists("results/summary/v2_c1_weibo_external_holdout_summary.csv")
        and exists("results/summary/c2_breakout_weibo_external_holdout_summary.csv")
        and exists("results/summary/c3_control_weibo_external_holdout_summary.csv")
        and exists("results/summary/weibo_raw_external_holdout_comparison.csv")
        and exists("results/drafts/weibo_raw_external_holdout_validation.md")
        and exists("scripts/summarize_weibo_raw_external_holdout.py"),
        "weibo_raw_e9_visual_diagnostics": exists(
            "results/figures/fig_weibo_raw_e9_diagnostics.png"
        )
        and exists("results/figures/fig_weibo_raw_e9_diagnostics.pdf")
        and exists("results/figures/fig_weibo_raw_e9_diagnostics.svg")
        and exists("results/figures/plot_data_weibo_raw_e9_diagnostics.csv")
        and exists("results/drafts/weibo_raw_e9_visual_diagnostics.md")
        and exists("scripts/plot_weibo_raw_experiment_diagnostics.py"),
        "weibo_raw_e10_case_studies": exists(
            "results/case_studies/weibo_raw_e10_cases.csv"
        )
        and exists("results/case_studies/weibo_raw_e10_case_curves.csv")
        and exists("results/figures/fig_weibo_raw_e10_case_studies.png")
        and exists("results/figures/fig_weibo_raw_e10_case_studies.pdf")
        and exists("results/figures/fig_weibo_raw_e10_case_studies.svg")
        and exists("results/drafts/weibo_raw_e10_case_analysis.md")
        and exists("scripts/build_weibo_raw_e10_case_studies.py"),
        "weibo_raw_e12_early_warning": exists(
            "results/summary/weibo_raw_e12_early_warning_seed_summary.csv"
        )
        and exists("results/summary/weibo_raw_e12_early_warning_summary.csv")
        and exists("results/summary/weibo_raw_e12_early_warning_recall_curve.csv")
        and exists("results/summary/weibo_raw_e12_early_warning_window_coverage.csv")
        and exists("results/figures/fig_weibo_raw_e12_early_warning.png")
        and exists("results/figures/fig_weibo_raw_e12_early_warning.pdf")
        and exists("results/figures/fig_weibo_raw_e12_early_warning.svg")
        and exists("results/drafts/weibo_raw_e12_early_warning.md")
        and exists("scripts/build_weibo_raw_e12_early_warning.py"),
        "weibo_raw_e4_significance": exists(
            "results/summary/weibo_raw_e4_significance_tests.csv"
        )
        and exists("results/drafts/weibo_raw_e4_significance_tests.md")
        and exists("scripts/build_weibo_raw_e4_significance.py"),
        "weibo_raw_e14_reproducibility": exists(
            "results/summary/weibo_raw_e14_reproducibility_manifest.json"
        )
        and exists("results/summary/weibo_raw_e14_reproducibility_files.csv")
        and exists("results/summary/weibo_raw_e14_reproducibility_checklist.csv")
        and exists("results/drafts/weibo_raw_e14_reproducibility_audit.md")
        and exists("scripts/build_weibo_raw_e14_reproducibility.py"),
        "weibo_raw_final_index": exists(
            "results/summary/weibo_raw_final_experiment_index.csv"
        )
        and exists("results/summary/weibo_raw_final_integrity_audit.csv")
        and exists("results/drafts/weibo_raw_final_experiment_index.md")
        and exists("scripts/build_weibo_raw_final_experiment_index.py"),
        "weibo_c1_note": exists("results/drafts/weibo_c1_results_note.md"),
        "weibo_raw_c1_table": exists("results/summary/weibo_raw_c1_paper_table.csv"),
        "weibo_raw_c1_insert": exists("results/drafts/weibo_c1_paper_insert.md"),
        "chapter5_experiment_discussion": exists(
            "results/drafts/chapter5_experiment_discussion_draft.md"
        ),
    }


def workflow_phase(s: dict[str, Any]) -> tuple[str, str]:
    if not s["processed_datasets"] or int(s["split_files"]) < 4 or not s["loader"]:
        return "V0 数据准备", "补齐统一数据、标签、split 和 dataset_loader。"
    if int(s["structure_baselines"]) < 8 or int(s["graph_baselines"]) < 8:
        return "V0 baseline 复现", "补齐结构统计 baseline 与传播图 baseline。"
    if not (s["dynamic_snapshots"] and s["communities"] and s["breakout_events"]):
        return "V0 -> V1 数据基础", "补齐 C2 动态快照、社区近似和爆发事件标签。"
    if not s["seir_script"] or int(s["seir_metrics"]) < 2:
        return "V0 -> V1 经典动力学 baseline", "实现并运行 PHEME SIR/SEIR 传播规模预测。"
    if not (s["summary_tables"] and s["classification_figures"] and s["seir_figure"]):
        return "阶段 E 结果整理", "重新生成 summary 表、分类图和 SEIR 预测图。"
    if int(s["v1_inputs"]) < 9 or not s["v1_loader"]:
        return "V1 多模态输入构建", "补齐文本、拓扑、时序、用户模态输入包和 split loader。"
    if not s["v1_train_script"] or int(s["v1_metrics"]) < 1:
        return "V1 融合模型入口", "训练 HeteroRumorDyn V1，先跑通 PHEME 180 分钟窗口。"
    if not s["v1_temporal_180"]:
        return "V1 三窗口与消融完成", "在 temporal split 上复验 PHEME 180 分钟早期传播规模预测。"
    if not (s["plm_text_cache"] and s["plm_multiseed_summary"] and s["plm_multiseed_paired"]):
        return "V1 预训练文本编码升级", "完成 MiniLM 文本特征、多随机种子和 paired text ablation。"
    if not s["v2_c1_selected"]:
        return "V2/C1 VAE 传播动能因子分解", "训练 K=4/K=16 初版，再跑 K/KL 敏感性。"
    if not (
        s["v2_c1_factor_figure"]
        and s["v2_c1_k_sensitivity"]
        and s["v2_c1_kl_sensitivity"]
    ):
        return "V2/C1 因子敏感性", "生成因子图，并完成 K 与 KL 权重敏感性分析。"
    if not (s["v2_c1_cf_robustness"] and s["v2_c1_sensitivity_figure"]):
        return "V2/C1 反事实初版", "评估文本扰动鲁棒性，并生成反事实敏感性图。"
    if not (
        s["v2_disentangled_model"]
        and int(s["v2_disentangled_runs"]) >= 5
        and s["v2_disentangled_summary"]
        and s["v2_disentangled_figure"]
    ):
        return "V2/C1 反事实重设计", "完成目标匹配文本替换、内容/动力因子分离和五随机种子复验。"
    if not (
        int(s["c2_breakout_runs"]) >= 24
        and int(s["c3_control_runs"]) >= 24
        and s["c2_breakout_multiseed_summary"]
        and s["c3_control_multiseed_summary"]
        and s["c2_breakout_figure"]
        and s["c3_control_figure"]
        and s["c2_c3_temporal_table"]
    ):
        return "V3/C2-C3 初版实验推进中", "补齐 C2/C3 多随机种子、temporal/proxy 压力测试、论文表格和图。"
    if not (s["c2_c3_case_figure"] and s["c2_c3_case_report"]):
        return "V3/C2-C3 多随机种子与 temporal 初版结果已完成", "优化 C2 消融解释、补案例分析，并写入第五章 C2/C3 实验段落。"
    if not (s["weibo_v1_input_180"] and s["weibo_raw_v1_ready"]):
        return (
            "Raw Weibo adapter in progress",
            "Regenerate Weibo V1 inputs from raw files with nonzero text/profile coverage.",
        )
    if int(s["weibo_raw_smoke"]) < 1:
        return (
            "Raw Weibo V1 smoke pending",
            "Run a short V1 smoke training job on the raw Weibo artifact.",
        )
    if not (
        int(s["weibo_v1_runs"]) >= 5
        and s["weibo_v1_summary"]
        and int(s["weibo_v2_selected_runs"]) >= 5
        and s["weibo_v2_selected_summary"]
    ):
        return (
            "Raw Weibo adapter ready; full rerun pending",
            "Rerun Weibo V1 and V2/C1 five-seed experiments from the raw data artifact; old BiGCN summaries are superseded.",
        )
    if not (
        int(s["weibo_raw_c2_runs"]) >= 5
        and s["weibo_raw_c2_summary"]
        and int(s["weibo_raw_c3_runs"]) >= 5
        and s["weibo_raw_c3_summary"]
        and s["weibo_raw_c2_c3_note"]
    ):
        return (
            "Raw Weibo C2/C3 rerun pending",
            "Rebuild raw-Weibo C2 foundation, run C2/C3 five-seed proxy experiments, and summarize the results separately from the old BiGCN-adapter outputs.",
        )
    if not s["weibo_raw_c2_c3_order_window_sensitivity"]:
        return (
            "Raw Weibo C2/C3 sensitivity pending",
            "Run raw-Weibo C2/C3 order-window sensitivity checks for event-order window sizes 50, 100, and 200.",
        )
    if not s["weibo_raw_c2_c3_threshold_sensitivity"]:
        return (
            "Raw Weibo C2/C3 threshold sensitivity pending",
            "Run raw-Weibo C2/C3 breakout-threshold sensitivity checks for theta_cross and theta_branch_ratio.",
        )
    if not s["weibo_raw_c2_threshold_audit"]:
        return (
            "Raw Weibo C2 threshold audit pending",
            "Audit why raw-Weibo threshold perturbations are invariant under the star-edge proxy.",
        )
    if not s["weibo_raw_c2_c3_preferred"]:
        return (
            "Raw Weibo C2/C3 preferred setting pending",
            "Promote order_window_size=50 as the preferred raw-Weibo C2/C3 setting and regenerate the preferred summary.",
        )
    if not s["weibo_raw_preferred_artifact_validation"]:
        return (
            "Raw Weibo preferred artifact validation pending",
            "Validate preferred raw-Weibo C2/C3 summaries, seed directories, and legacy ow100 mapping.",
        )
    if not s["weibo_raw_reporting_entrypoints"]:
        return (
            "Raw Weibo reporting entrypoints pending",
            "Align raw-Weibo V1/V2-C1/C2/C3 reporting entry points and validate all preferred seed directories.",
        )
    if not s["weibo_raw_efficiency_benchmark"]:
        return (
            "Raw Weibo E8 efficiency benchmark pending",
            "Benchmark raw-Weibo V1/V2-C1/C2/C3 runtime and summarize the timing outliers.",
        )
    if not s["weibo_raw_external_holdout_validation"]:
        return (
            "Raw Weibo external holdout validation pending",
            "Rerun raw-Weibo V1/V2-C1/C2/C3 on the non-stratified fixed-seed holdout and compare against preferred stratified results.",
        )
    if not s["weibo_raw_e9_visual_diagnostics"]:
        return (
            "Raw Weibo E9 visual diagnostics pending",
            "Generate raw-Weibo diagnostic figures for holdout robustness, order-window sensitivity, C3 strategy comparison, and runtime.",
        )
    if not s["weibo_raw_e10_case_studies"]:
        return (
            "Raw Weibo E10 case studies pending",
            "Select raw-Weibo C2/C3 success and failure cases, generate case curves, and export the case-study figure.",
        )
    if not s["weibo_raw_e12_early_warning"]:
        return (
            "Raw Weibo E12 early-warning validation pending",
            "Quantify C2 lead-time recall, false-alarm rate, and breakout-window coverage on preferred raw-Weibo outputs.",
        )
    if not s["weibo_raw_e4_significance"]:
        return (
            "Raw Weibo E4 significance tests pending",
            "Run paired seed-level significance tests for raw-Weibo V1/V2-C1, C2, and C3 comparisons.",
        )
    if not s["weibo_raw_e14_reproducibility"]:
        return (
            "Raw Weibo E14 reproducibility audit pending",
            "Build the raw-Weibo reproducibility manifest, file checksum table, and audit checklist.",
        )
    if not s["weibo_raw_final_index"]:
        return (
            "Raw Weibo final index pending",
            "Build the final raw-Weibo experiment index and verify no historical BiGCN/ow100 outputs are marked preferred.",
        )
    return (
        "Raw Weibo final experiment index complete",
        "Raw-Weibo experiment line is complete; use the final index for reporting.",
    )


def checklist_rows(s: dict[str, Any]) -> list[tuple[str, bool]]:
    return [
        ("统一数据集", bool(s["processed_datasets"])),
        ("label_map.json", bool(s["label_map"])),
        ("统一 split 文件 >= 8", int(s["split_files"]) >= 8),
        ("dataset_loader.py", bool(s["loader"])),
        ("结构统计 baseline >= 8", int(s["structure_baselines"]) >= 8),
        ("传播图 baseline >= 8", int(s["graph_baselines"]) >= 8),
        ("SIR/SEIR baseline", bool(s["seir_script"]) and int(s["seir_metrics"]) >= 2),
        ("结果汇总表与基础图", bool(s["summary_tables"]) and bool(s["classification_figures"])),
        ("V1 多模态输入", int(s["v1_inputs"]) >= 9 and bool(s["v1_loader"])),
        ("V1 融合模型训练", bool(s["v1_train_script"]) and int(s["v1_metrics"]) >= 1),
        ("V1 temporal split 180", bool(s["v1_temporal_180"])),
        ("V1 MiniLM 多随机种子", bool(s["plm_multiseed_summary"]) and bool(s["plm_multiseed_paired"])),
        ("V2/C1 selected VAE", bool(s["v2_c1_selected"])),
        ("V2/C1 K/KL 敏感性", bool(s["v2_c1_k_sensitivity"]) and bool(s["v2_c1_kl_sensitivity"])),
        ("V2/C1 反事实初版", bool(s["v2_c1_cf_robustness"]) and bool(s["v2_c1_sensitivity_figure"])),
        (
            "V2/C1 disentangled 多种子",
            bool(s["v2_disentangled_model"])
            and int(s["v2_disentangled_runs"]) >= 5
            and bool(s["v2_disentangled_summary"])
            and bool(s["v2_disentangled_figure"]),
        ),
        ("V2/C1 论文主表与结果段落", bool(s["v2_c1_paper_table"]) and bool(s["v2_c1_results_draft"])),
        (
            "rumdetect2017 Twitter15/16 转换与 split",
            bool(s["rumdetect2017_processed"]) and int(s["rumdetect2017_splits"]) >= 8,
        ),
        (
            "V3/C2 破圈预警多种子与 temporal",
            int(s["c2_breakout_runs"]) >= 24
            and bool(s["c2_breakout_multiseed_summary"])
            and bool(s["c2_breakout_figure"]),
        ),
        (
            "V3/C3 闭环控制多种子与 temporal",
            int(s["c3_control_runs"]) >= 24
            and bool(s["c3_control_multiseed_summary"])
            and bool(s["c3_control_figure"]),
        ),
        (
            "V3/C2-C3 论文表格与结果段落",
            bool(s["c2_breakout_paper_table"])
            and bool(s["c3_control_paper_table"])
            and bool(s["c2_c3_temporal_table"])
            and bool(s["c2_c3_results_draft"]),
        ),
        ("V3/C2-C3 案例分析图与汇报稿", bool(s["c2_c3_case_figure"]) and bool(s["c2_c3_case_report"])),
        (
            "Paper baselines 5/5: dynamics + MIDPMS + DSHCL + ED-ID + Inf-VAE",
            bool(s["paper_baselines_mapping"])
            and bool(s["paper_dynamics_summary"])
            and bool(s["paper_midpms_summary"])
            and bool(s["paper_dshcl_summary"])
            and bool(s["paper_edid_summary"])
            and bool(s["paper_inf_vae_script"])
            and int(s["paper_inf_vae_runs"]) >= 18
            and bool(s["paper_inf_vae_integrated"]),
        ),
        (
            "V1 fair comparison: fixed 180 min + exact-ID audit",
            bool(s["v1_fair180_main_table"])
            and bool(s["v1_fair180_temporal_table"])
            and bool(s["v1_fair180_audit"])
            and bool(s["v1_fair180_bootstrap"]),
        ),
        ("新增微博原始数据集：V1 obs_180events 输入生成与 loader 自检", bool(s["weibo_raw_v1_ready"])),
        ("新增微博原始数据集：V1 训练烟测", int(s["weibo_raw_smoke"]) >= 1),
        ("新增微博原始数据集：V1 五随机种子复验", int(s["weibo_v1_runs"]) >= 5 and bool(s["weibo_v1_summary"])),
        (
            "新增微博原始数据集：V2/C1 selected VAE 五随机种子复验",
            int(s["weibo_v2_selected_runs"]) >= 5 and bool(s["weibo_v2_selected_summary"]),
        ),
        (
            "新增微博原始数据集：C2/C3 五随机种子代理复验",
            int(s["weibo_raw_c2_runs"]) >= 5
            and bool(s["weibo_raw_c2_summary"])
            and int(s["weibo_raw_c3_runs"]) >= 5
            and bool(s["weibo_raw_c3_summary"])
            and bool(s["weibo_raw_c2_c3_note"]),
        ),
        (
            "新增微博原始数据集：C2/C3 order-window 敏感性",
            bool(s["weibo_raw_c2_c3_order_window_sensitivity"]),
        ),
        (
            "新增微博原始数据集：C2/C3 breakout-threshold 敏感性",
            bool(s["weibo_raw_c2_c3_threshold_sensitivity"]),
        ),
        (
            "新增微博原始数据集：C2 阈值不敏感原因审计",
            bool(s["weibo_raw_c2_threshold_audit"]),
        ),
        (
            "新增微博原始数据集：C2/C3 推荐口径 order_window_size=50",
            bool(s["weibo_raw_c2_c3_preferred"]),
        ),
        (
            "新增微博原始数据集：preferred artifact 一致性验证",
            bool(s["weibo_raw_preferred_artifact_validation"]),
        ),
        (
            "新增微博原始数据集：V1/V2-C1/C2/C3 reporting entrypoints 对齐",
            bool(s["weibo_raw_reporting_entrypoints"]),
        ),
        (
            "新增微博原始数据集：E8 efficiency benchmark",
            bool(s["weibo_raw_efficiency_benchmark"]),
        ),
        (
            "新增微博原始数据集：external holdout validation",
            bool(s["weibo_raw_external_holdout_validation"]),
        ),
        (
            "新增微博原始数据集：E9 visual diagnostics",
            bool(s["weibo_raw_e9_visual_diagnostics"]),
        ),
        (
            "新增微博原始数据集：E10 case studies",
            bool(s["weibo_raw_e10_case_studies"]),
        ),
        (
            "新增微博原始数据集：E12 early-warning validation",
            bool(s["weibo_raw_e12_early_warning"]),
        ),
        (
            "新增微博原始数据集：E4 significance tests",
            bool(s["weibo_raw_e4_significance"]),
        ),
        (
            "新增微博原始数据集：E14 reproducibility audit",
            bool(s["weibo_raw_e14_reproducibility"]),
        ),
        (
            "新增微博原始数据集：final experiment index and integrity audit",
            bool(s["weibo_raw_final_index"]),
        ),
        (
            "新增微博原始数据集：C1 论文表格与讨论段落",
            bool(s["weibo_raw_c1_table"]) and bool(s["weibo_raw_c1_insert"]),
        ),
        (
            "第五章 C1/C2/C3 实验讨论整合草稿",
            bool(s["chapter5_experiment_discussion"]),
        ),
    ]


def render_markdown(s: dict[str, Any]) -> str:
    phase, next_step = workflow_phase(s)
    lines = [
        "# Experiment Workflow Status",
        "",
        "> 每次开始新实验前，先运行 python scripts/workflow_status.py，再根据当前阶段决定下一步。",
        "",
        "## 当前定位",
        "",
        f"- 当前阶段：**{phase}**",
        f"- 下一步优先级：**{next_step}**",
        "",
        "## 与四个指导文档的对应",
        "",
        "| 文档 | 当前用途 |",
        "|---|---|",
        "| 00-quickstart.md | 判断当前实验阶段和最小可运行命令。 |",
        "| 01-development-guide.md | 对齐阶段路线图、开发任务和结果交付物。 |",
        "| 02-debug-manual.md | 结果异常、指标异常、数据疑似泄漏时排查。 |",
        "| HeteroRumorDyn_experiment_guide.md | 对齐 V0/V1/V2/C1/C2/C3 与论文实验矩阵。 |",
        "",
        "## 已完成产物",
        "",
    ]
    lines.extend(
        f"- [{mark(done)}] {name}：{status_text(done)}"
        for name, done in checklist_rows(s)
    )
    lines.extend(
        [
            "",
            "## 关键结果文件",
            "",
            "- results/summary/v1_plm_multiseed_summary.csv",
            "- results/summary/v2_c1_disentangled_multiseed_summary.csv",
            "- results/summary/paper_v2_c1_main_table.csv",
            "- results/drafts/v2_c1_results_paragraph.md",
            "- results/drafts/rumdetect2017_audit.md",
            "- results/figures/fig8_v2_disentangled_multiseed.png",
            "- results/summary/c2_breakout_paper_table.csv",
            "- results/summary/c3_control_paper_table.csv",
            "- results/paper_baselines/fair180/inf_vae/",
            "- results/drafts/inf_vae_adapted_results_explanation.md",
            "- results/drafts/c2_c3_results_explanation.md",
            "- results/drafts/c2_c3_case_analysis_for_report.md",
            "- results/figures/fig12_c2_c3_case_studies.png",
            "- data/processed/v1_inputs/weibo/obs_180events_metadata.json",
            "- results/heterorumor_v1_weibo_raw_smoke/",
            "- results/summary/v1_weibo_multiseed_summary.csv",
            "- results/summary/v2_c1_weibo_selected_multiseed_summary.csv",
            "- results/summary/c2_breakout_weibo_raw_summary.csv",
            "- results/summary/c3_control_weibo_raw_summary.csv",
            "- results/drafts/weibo_raw_c2_c3_experiment_note.md",
            "- results/summary/weibo_raw_c2_c3_order_window_sensitivity.csv",
            "- results/drafts/weibo_raw_c2_c3_order_window_sensitivity.md",
            "- results/summary/weibo_raw_c2_c3_threshold_sensitivity.csv",
            "- results/drafts/weibo_raw_c2_c3_threshold_sensitivity.md",
            "- results/summary/weibo_raw_c2_threshold_distribution.csv",
            "- results/summary/weibo_raw_c2_threshold_label_flip_audit.csv",
            "- results/summary/weibo_raw_c2_threshold_condition_hits.csv",
            "- results/drafts/weibo_raw_c2_threshold_audit.md",
            "- results/summary/c2_breakout_weibo_raw_preferred_summary.csv",
            "- results/summary/c3_control_weibo_raw_preferred_summary.csv",
            "- results/drafts/weibo_raw_c2_c3_preferred_setting.md",
            "- results/summary/weibo_raw_preferred_artifact_validation.csv",
            "- results/summary/weibo_raw_preferred_artifact_map.csv",
            "- results/drafts/weibo_raw_preferred_artifact_validation.md",
            "- results/summary/weibo_raw_reporting_entrypoints.csv",
            "- results/summary/weibo_raw_reporting_entrypoints_validation.csv",
            "- results/drafts/weibo_raw_reporting_entrypoints.md",
            "- results/summary/weibo_raw_efficiency_summary.csv",
            "- results/summary/weibo_raw_v1_efficiency_runs.csv",
            "- results/summary/weibo_raw_v2_c1_efficiency_runs.csv",
            "- results/drafts/weibo_raw_efficiency_benchmark.md",
            "- results/summary/v1_weibo_external_holdout_summary.csv",
            "- results/summary/v2_c1_weibo_external_holdout_summary.csv",
            "- results/summary/c2_breakout_weibo_external_holdout_summary.csv",
            "- results/summary/c3_control_weibo_external_holdout_summary.csv",
            "- results/summary/weibo_raw_external_holdout_comparison.csv",
            "- results/drafts/weibo_raw_external_holdout_validation.md",
            "- results/figures/fig_weibo_raw_e9_diagnostics.png",
            "- results/figures/fig_weibo_raw_e9_diagnostics.pdf",
            "- results/figures/fig_weibo_raw_e9_diagnostics.svg",
            "- results/figures/plot_data_weibo_raw_e9_diagnostics.csv",
            "- results/drafts/weibo_raw_e9_visual_diagnostics.md",
            "- results/case_studies/weibo_raw_e10_cases.csv",
            "- results/case_studies/weibo_raw_e10_case_curves.csv",
            "- results/figures/fig_weibo_raw_e10_case_studies.png",
            "- results/figures/fig_weibo_raw_e10_case_studies.pdf",
            "- results/figures/fig_weibo_raw_e10_case_studies.svg",
            "- results/drafts/weibo_raw_e10_case_analysis.md",
            "- results/summary/weibo_raw_e12_early_warning_seed_summary.csv",
            "- results/summary/weibo_raw_e12_early_warning_summary.csv",
            "- results/summary/weibo_raw_e12_early_warning_recall_curve.csv",
            "- results/summary/weibo_raw_e12_early_warning_window_coverage.csv",
            "- results/figures/fig_weibo_raw_e12_early_warning.png",
            "- results/figures/fig_weibo_raw_e12_early_warning.pdf",
            "- results/figures/fig_weibo_raw_e12_early_warning.svg",
            "- results/drafts/weibo_raw_e12_early_warning.md",
            "- results/summary/weibo_raw_e4_significance_tests.csv",
            "- results/drafts/weibo_raw_e4_significance_tests.md",
            "- results/summary/weibo_raw_e14_reproducibility_manifest.json",
            "- results/summary/weibo_raw_e14_reproducibility_files.csv",
            "- results/summary/weibo_raw_e14_reproducibility_checklist.csv",
            "- results/drafts/weibo_raw_e14_reproducibility_audit.md",
            "- results/summary/weibo_raw_final_experiment_index.csv",
            "- results/summary/weibo_raw_final_integrity_audit.csv",
            "- results/drafts/weibo_raw_final_experiment_index.md",
            "- results/summary/weibo_raw_c1_paper_table.csv",
            "- results/drafts/weibo_c1_paper_insert.md",
            "- results/drafts/chapter5_experiment_discussion_draft.md",
            "- results/drafts/weibo_c1_results_note.md",
            "",
            "## 下一步建议",
            "",
            f"下一步：{next_step}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    status = build_status()
    OUTPUT.write_text(render_markdown(status), encoding="utf-8-sig")
    phase, next_step = workflow_phase(status)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "phase": phase,
                "next_step": next_step,
                "status": status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
