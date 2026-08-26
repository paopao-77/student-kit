# Experiment Workflow Status

> 每次开始新实验前，先运行 python scripts/workflow_status.py，再根据当前阶段决定下一步。

## 当前定位

- 当前阶段：**Raw Weibo final experiment index complete**
- 下一步优先级：**Raw-Weibo experiment line is complete; use the final index for reporting.**

## 与四个指导文档的对应

| 文档 | 当前用途 |
|---|---|
| 00-quickstart.md | 判断当前实验阶段和最小可运行命令。 |
| 01-development-guide.md | 对齐阶段路线图、开发任务和结果交付物。 |
| 02-debug-manual.md | 结果异常、指标异常、数据疑似泄漏时排查。 |
| HeteroRumorDyn_experiment_guide.md | 对齐 V0/V1/V2/C1/C2/C3 与论文实验矩阵。 |

## 已完成产物

- [x] 统一数据集：已完成
- [x] label_map.json：已完成
- [x] 统一 split 文件 >= 8：已完成
- [x] dataset_loader.py：已完成
- [x] 结构统计 baseline >= 8：已完成
- [x] 传播图 baseline >= 8：已完成
- [x] SIR/SEIR baseline：已完成
- [x] 结果汇总表与基础图：已完成
- [x] V1 多模态输入：已完成
- [x] V1 融合模型训练：已完成
- [x] V1 temporal split 180：已完成
- [x] V1 MiniLM 多随机种子：已完成
- [x] V2/C1 selected VAE：已完成
- [x] V2/C1 K/KL 敏感性：已完成
- [x] V2/C1 反事实初版：已完成
- [x] V2/C1 disentangled 多种子：已完成
- [x] V2/C1 论文主表与结果段落：已完成
- [x] rumdetect2017 Twitter15/16 转换与 split：已完成
- [x] V3/C2 破圈预警多种子与 temporal：已完成
- [x] V3/C3 闭环控制多种子与 temporal：已完成
- [x] V3/C2-C3 论文表格与结果段落：已完成
- [x] V3/C2-C3 案例分析图与汇报稿：已完成
- [x] Paper baselines 5/5: dynamics + MIDPMS + DSHCL + ED-ID + Inf-VAE：已完成
- [x] V1 fair comparison: fixed 180 min + exact-ID audit：已完成
- [x] 新增微博原始数据集：V1 obs_180events 输入生成与 loader 自检：已完成
- [x] 新增微博原始数据集：V1 训练烟测：已完成
- [x] 新增微博原始数据集：V1 五随机种子复验：已完成
- [x] 新增微博原始数据集：V2/C1 selected VAE 五随机种子复验：已完成
- [x] 新增微博原始数据集：C2/C3 五随机种子代理复验：已完成
- [x] 新增微博原始数据集：C2/C3 order-window 敏感性：已完成
- [x] 新增微博原始数据集：C2/C3 breakout-threshold 敏感性：已完成
- [x] 新增微博原始数据集：C2 阈值不敏感原因审计：已完成
- [x] 新增微博原始数据集：C2/C3 推荐口径 order_window_size=50：已完成
- [x] 新增微博原始数据集：preferred artifact 一致性验证：已完成
- [x] 新增微博原始数据集：V1/V2-C1/C2/C3 reporting entrypoints 对齐：已完成
- [x] 新增微博原始数据集：E8 efficiency benchmark：已完成
- [x] 新增微博原始数据集：external holdout validation：已完成
- [x] 新增微博原始数据集：E9 visual diagnostics：已完成
- [x] 新增微博原始数据集：E10 case studies：已完成
- [x] 新增微博原始数据集：E12 early-warning validation：已完成
- [x] 新增微博原始数据集：E4 significance tests：已完成
- [x] 新增微博原始数据集：E14 reproducibility audit：已完成
- [x] 新增微博原始数据集：final experiment index and integrity audit：已完成
- [x] 新增微博原始数据集：C1 论文表格与讨论段落：已完成
- [x] 第五章 C1/C2/C3 实验讨论整合草稿：已完成

## 关键结果文件

- results/summary/v1_plm_multiseed_summary.csv
- results/summary/v2_c1_disentangled_multiseed_summary.csv
- results/summary/paper_v2_c1_main_table.csv
- results/drafts/v2_c1_results_paragraph.md
- results/drafts/rumdetect2017_audit.md
- results/figures/fig8_v2_disentangled_multiseed.png
- results/summary/c2_breakout_paper_table.csv
- results/summary/c3_control_paper_table.csv
- results/paper_baselines/fair180/inf_vae/
- results/drafts/inf_vae_adapted_results_explanation.md
- results/drafts/c2_c3_results_explanation.md
- results/drafts/c2_c3_case_analysis_for_report.md
- results/figures/fig12_c2_c3_case_studies.png
- data/processed/v1_inputs/weibo/obs_180events_metadata.json
- results/heterorumor_v1_weibo_raw_smoke/
- results/summary/v1_weibo_multiseed_summary.csv
- results/summary/v2_c1_weibo_selected_multiseed_summary.csv
- results/summary/c2_breakout_weibo_raw_summary.csv
- results/summary/c3_control_weibo_raw_summary.csv
- results/drafts/weibo_raw_c2_c3_experiment_note.md
- results/summary/weibo_raw_c2_c3_order_window_sensitivity.csv
- results/drafts/weibo_raw_c2_c3_order_window_sensitivity.md
- results/summary/weibo_raw_c2_c3_threshold_sensitivity.csv
- results/drafts/weibo_raw_c2_c3_threshold_sensitivity.md
- results/summary/weibo_raw_c2_threshold_distribution.csv
- results/summary/weibo_raw_c2_threshold_label_flip_audit.csv
- results/summary/weibo_raw_c2_threshold_condition_hits.csv
- results/drafts/weibo_raw_c2_threshold_audit.md
- results/summary/c2_breakout_weibo_raw_preferred_summary.csv
- results/summary/c3_control_weibo_raw_preferred_summary.csv
- results/drafts/weibo_raw_c2_c3_preferred_setting.md
- results/summary/weibo_raw_preferred_artifact_validation.csv
- results/summary/weibo_raw_preferred_artifact_map.csv
- results/drafts/weibo_raw_preferred_artifact_validation.md
- results/summary/weibo_raw_reporting_entrypoints.csv
- results/summary/weibo_raw_reporting_entrypoints_validation.csv
- results/drafts/weibo_raw_reporting_entrypoints.md
- results/summary/weibo_raw_efficiency_summary.csv
- results/summary/weibo_raw_v1_efficiency_runs.csv
- results/summary/weibo_raw_v2_c1_efficiency_runs.csv
- results/drafts/weibo_raw_efficiency_benchmark.md
- results/summary/v1_weibo_external_holdout_summary.csv
- results/summary/v2_c1_weibo_external_holdout_summary.csv
- results/summary/c2_breakout_weibo_external_holdout_summary.csv
- results/summary/c3_control_weibo_external_holdout_summary.csv
- results/summary/weibo_raw_external_holdout_comparison.csv
- results/drafts/weibo_raw_external_holdout_validation.md
- results/figures/fig_weibo_raw_e9_diagnostics.png
- results/figures/fig_weibo_raw_e9_diagnostics.pdf
- results/figures/fig_weibo_raw_e9_diagnostics.svg
- results/figures/plot_data_weibo_raw_e9_diagnostics.csv
- results/drafts/weibo_raw_e9_visual_diagnostics.md
- results/case_studies/weibo_raw_e10_cases.csv
- results/case_studies/weibo_raw_e10_case_curves.csv
- results/figures/fig_weibo_raw_e10_case_studies.png
- results/figures/fig_weibo_raw_e10_case_studies.pdf
- results/figures/fig_weibo_raw_e10_case_studies.svg
- results/drafts/weibo_raw_e10_case_analysis.md
- results/summary/weibo_raw_e12_early_warning_seed_summary.csv
- results/summary/weibo_raw_e12_early_warning_summary.csv
- results/summary/weibo_raw_e12_early_warning_recall_curve.csv
- results/summary/weibo_raw_e12_early_warning_window_coverage.csv
- results/figures/fig_weibo_raw_e12_early_warning.png
- results/figures/fig_weibo_raw_e12_early_warning.pdf
- results/figures/fig_weibo_raw_e12_early_warning.svg
- results/drafts/weibo_raw_e12_early_warning.md
- results/summary/weibo_raw_e4_significance_tests.csv
- results/drafts/weibo_raw_e4_significance_tests.md
- results/summary/weibo_raw_e14_reproducibility_manifest.json
- results/summary/weibo_raw_e14_reproducibility_files.csv
- results/summary/weibo_raw_e14_reproducibility_checklist.csv
- results/drafts/weibo_raw_e14_reproducibility_audit.md
- results/summary/weibo_raw_final_experiment_index.csv
- results/summary/weibo_raw_final_integrity_audit.csv
- results/drafts/weibo_raw_final_experiment_index.md
- results/summary/weibo_raw_c1_paper_table.csv
- results/drafts/weibo_c1_paper_insert.md
- results/drafts/chapter5_experiment_discussion_draft.md
- results/drafts/weibo_c1_results_note.md

## 下一步建议

下一步：Raw-Weibo experiment line is complete; use the final index for reporting.
