# V2/C1 Propagation-Momentum Factorization

## Current Objective

V2/C1 maps the four V1 modality representations into a low-dimensional variational latent space. The latent variables are treated as propagation-momentum factors and are used to predict remaining cascade growth.

## Model

```text
MiniLM text + directional topology + temporal LSTM + user features
  -> masked modality representations
  -> VAE encoder
  -> K-dimensional propagation-momentum factors
  -> hurdle cascade-growth prediction
```

The decoder reconstructs the masked four-modality representation. Training combines the V1 regression, growth classification, and relative-error losses with reconstruction and KL losses. KL weight is warmed up during the first epochs to reduce posterior collapse.

## Primary Command

```bash
conda run -n myenv python scripts/train_heterorumor_v1.py --model-version v2_c1_vae --dataset pheme --observation 180 --split-strategy stratified --seed 42 --split-seed 42 --epochs 40 --patience 8 --batch-size 64 --hidden-dim 64 --graph-layers 2 --latent-dim 16 --kl-weight 0.01 --reconstruction-weight 0.1 --kl-warmup-epochs 5 --text-feature-name multilingual_minilm --text-feature-path data/processed/v1_text_features/pheme/multilingual_minilm.npz --output-dir results/heterorumor_v2_c1
```

## First Formal Result

| Model | MAPE | MAE | R2 |
|---|---:|---:|---:|
| MiniLM V1 | 0.1241 | 2.9727 | 0.7413 |
| V2/C1 VAE K=16 | 0.1232 | 2.9445 | 0.7451 |

All 16 latent factors are active under the current standard-deviation check. This is a valid VAE foundation, but the MAPE reduction is below the final C1 target of 5%. The next experiments are K sensitivity, KL/reconstruction-weight sensitivity, and counterfactual robustness training.

## Sensitivity Results

Latent dimension selection is based on validation MAPE, not test MAPE. `K=4` is selected even though the single-seed K=16 test result is slightly lower.

| K | Validation MAPE | Test MAPE | Active factors |
|---:|---:|---:|---:|
| 4 | 0.1331 | 0.1246 | 4 |
| 8 | 0.1338 | 0.1249 | 8 |
| 16 | 0.1335 | 0.1232 | 16 |
| 32 | 0.1333 | 0.1251 | 32 |

With K fixed to 4, KL weight 0.1 gives the lowest validation MAPE (`0.1320`). All four factors remain active, so this setting is used for the counterfactual experiment.

The separately rerun selected configuration obtains test MAPE `0.1231`, MAE `2.9521`, and R2 `0.7436`. Sensitivity-run files are excluded from the automatic paper-table ranking; only this validation-selected configuration is treated as the formal tuned V2 result.

## Counterfactual Constraint V1

The first counterfactual constraint masks 20% of MiniLM embedding dimensions during training and enforces supervised prediction, prediction consistency, probability consistency, and latent-factor consistency between clean and intervened samples.

Counterfactual weights `0, 0.1, 0.5, 1.0` were compared. Selection used the mean validation MAPE over clean and 10%/20%/30% masked-text conditions. The best validation-robust model remained `lambda_CF=0`.

`lambda_CF=0.5` nearly eliminates test degradation at 30% masking, but starts from a worse clean MAPE. Therefore the first constraint improves local invariance without improving overall robust accuracy. It does not pass the C1 target of at least 8% improvement on noisy data and must not be presented as a successful causal result.

The next counterfactual version should use target-matched content replacement or an explicit content/dynamics factor split, then repeat the same validation-only selection and noisy test protocol.

## Factor Figure

```bash
python scripts/plot_v2_c1_factors.py
```

The script exports t-SNE coordinates, factor-growth correlations, and PNG/PDF/SVG figures.

## Counterfactual Constraint V2: Target-Matched Text Swap + Disentanglement

The first masking-based counterfactual was replaced by a stronger intervention. For each batch, the training script now finds target-matched samples with similar remaining-growth targets and swaps in the most text-dissimilar MiniLM feature among the nearest target neighbors. This avoids unrealistic random embedding masking while keeping the intervention approximately label-preserving.

A new model, models/heterorumor_v2_c1_disentangled.py, explicitly separates content factors and dynamics factors.

Text representation maps to content VAE factors. Topology, temporal, and user representations map to dynamics VAE factors. The predictor uses both factor groups for hurdle cascade-growth prediction.

Training adds reconstruction, KL, content/dynamics cross-covariance penalty, dynamics latent consistency under matched text swap, and content-change encouragement under swapped text.

Single-seed selection compared no counterfactual, lambda_CF=0.1, and lambda_CF=0.5. The selected configuration is K=4, content K=4, KL=0.1, lambda_CF=0.5, disentangle weight=0.1.

| Variant | Validation MAPE | Test MAPE | Text noise 0.3 MAPE | Matched-swap MAPE |
|---|---:|---:|---:|---:|
| Disentangled, no CF | 0.133821 | 0.122977 | 0.123507 | 0.126122 |
| Disentangled, CF=0.1 | 0.132850 | 0.122964 | 0.123295 | 0.126513 |
| Disentangled, CF=0.5 | 0.132203 | 0.122998 | 0.123010 | 0.126159 |

The selected V2 disentangled configuration was then rerun with seeds 7, 21, 42, 84, and 2024 under fixed split_seed=42.

| Model | Seeds | Test MAPE mean | Test MAPE std | MAE mean | R2 mean | Matched-swap MAPE mean |
|---|---:|---:|---:|---:|---:|---:|
| V2/C1 disentangled matched-swap CF | 5 | 0.124309 | 0.001513 | 3.000878 | 0.737547 | 0.126300 |

Compared with V1 MiniLM multi-seed MAPE 0.124710, the redesigned V2 is slightly better on clean MAPE and provides a clearer factor-separation mechanism. It does not yet dominate the earlier single-seed selected V2 result of 0.123094, so it should be framed as the C1 interpretability and robustness branch rather than the sole predictive winner.

Outputs:

- results/summary/v2_c1_disentangled_multiseed_summary.csv
- results/figures/fig8_v2_disentangled_multiseed.png
- results/figures/fig8_v2_disentangled_multiseed.svg
