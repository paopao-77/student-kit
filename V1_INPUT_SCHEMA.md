# HeteroRumorDyn V1 Input Schema

## Purpose

The V1 input layer supports early cascade-size prediction with four aligned modalities:

```text
source text + observed propagation graph + observed temporal sequence + observed user profile statistics
```

The target is the final unique cascade size. Inputs are truncated at the selected observation window to prevent future leakage.

## Generated Artifacts

```text
data/processed/v1_inputs/
  pheme/
    obs_60m.npz
    obs_180m.npz
    obs_360m.npz
    obs_*_metadata.json
  twitter15/
    obs_60m.npz
    obs_180m.npz
    obs_360m.npz
    obs_*_metadata.json
  twitter16/
    obs_60m.npz
    obs_180m.npz
    obs_360m.npz
    obs_*_metadata.json
```

Build or rebuild them with:

```bash
python scripts/build_v1_inputs.py --datasets pheme,twitter15,twitter16 --observations 60,180,360
```

## Array Schema

| Array | Shape | Meaning |
|---|---|---|
| `sample_ids` | `[N]` | Cascade ids. |
| `source_texts` | `[N]` | Raw source-post text for a future BERT tokenizer. |
| `text_features` | `[N, 256]` | Stable multilingual hash features for a dependency-free V1 smoke run. |
| `node_features` | `[sum(V), 8]` | Concatenated observed graph-node features. |
| `edge_index` | `[2, sum(E)]` | Concatenated local edge indices. |
| `node_ptr` | `[N + 1]` | Node offsets for each cascade. |
| `edge_ptr` | `[N + 1]` | Edge offsets for each cascade. |
| `global_features` | `[N, 8]` | Observed graph-level statistics. |
| `temporal_features` | `[N, T, 10]` | Fixed-window propagation sequence. |
| `temporal_masks` | `[N, T]` | Valid temporal positions. |
| `user_features` | `[N, 8]` | Aggregated user-profile features available before the cutoff. |
| `modality_masks` | `[N, 4]` | Availability of text, topology, temporal, and user-profile modalities. |
| `final_sizes` | `[N]` | Final unique cascade size. |
| `log_final_sizes` | `[N]` | `log(1 + final_size)` regression target. |
| `observed_sizes` | `[N]` | Number of nodes visible at the observation cutoff. |

## Feature Definitions

Node features:

```text
log in-degree, log out-degree, log total degree, root flag,
normalized depth, normalized delay, branch-community flag, root-child flag
```

Temporal features per window:

```text
new/cumulative nodes, new/cumulative edges, new/cumulative users,
new/active communities, cross-edge ratio, branch-community ratio
```

User features:

```text
mean/max followers, mean friends, mean statuses, verified ratio,
unique-user ratio, profile coverage, source-user followers
```

PHEME has profile attributes. Twitter15/16 do not, so their user-profile modality mask is zero. Text coverage is 100% for all three generated datasets.

## Leakage Rules

- Text uses the source post only, which is available at cascade start.
- Graph nodes and edges must have relative time no later than the observation cutoff.
- Temporal features use only events and edges visible before the cutoff.
- User aggregates use only users attached to observed events.
- Final cascade size is used only as the prediction target.
- Negative relative delays in raw PHEME records are clipped to zero.
- Final size uses the maximum unique-node count across `samples.csv`, `events.csv`, and `edges.csv` to handle minor source-file inconsistencies.

## Split Loader

Inspect an input split:

```bash
python v1_dataset.py --dataset pheme --observation 180 --split train --split-strategy stratified
python v1_dataset.py --dataset twitter15 --observation 60 --split test --split-strategy temporal
```

Use in Python:

```python
from v1_dataset import V1InputDataset, collate_v1_batch

dataset = V1InputDataset(
    dataset="pheme",
    observation=180,
    split="train",
    split_strategy="stratified",
)
batch = collate_v1_batch([dataset[0], dataset[1]], as_torch=True)
```

## Text Encoder Upgrade

The original 256-dimensional hash vector remains available as a dependency-free baseline. The upgraded encoder uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with attention-mask mean pooling and L2 normalization to produce frozen 384-dimensional source-post embeddings.

Build the reusable PHEME cache:

```bash
conda run -n myenv python scripts/build_plm_text_features.py --dataset pheme --observation 180 --feature-name multilingual_minilm --batch-size 32 --max-length 192
```

Run the PLM-backed model:

```bash
conda run -n myenv python scripts/train_heterorumor_v1.py --dataset pheme --observation 180 --split-strategy stratified --text-feature-name multilingual_minilm --text-feature-path data/processed/v1_text_features/pheme/multilingual_minilm.npz --epochs 30 --patience 6 --batch-size 64 --hidden-dim 64 --graph-layers 2 --output-dir results/heterorumor_v1_plm
```

Only source-post text is encoded, so the same leakage rule is preserved. The cache is aligned by `sample_id` and can be reused by every observation window and split.

## V1 Model And Training

The implemented V1 model uses:

```text
text MLP/BERT encoder
  + directional graph encoder
  + LSTM temporal encoder
  + optional user-profile MLP
  -> modality-mask-aware fusion
  -> zero-inflated remaining-growth regression
```

Run the primary PHEME experiment from the project root:

```bash
conda run -n myenv python scripts/train_heterorumor_v1.py --dataset pheme --observation 180 --split-strategy stratified --epochs 30 --patience 6 --batch-size 64 --hidden-dim 64 --graph-layers 2 --output-dir results/heterorumor_v1_hurdle
```

The model jointly predicts whether the cascade will continue growing and how much remaining growth to expect. The stopping threshold is selected on the validation split only. The first full PHEME 180-minute run reached test MAPE `0.1261`, MAE `3.0193`, and R2 `0.7372` on 1,286 test cascades.

After the primary run, evaluate the 60/360-minute windows and run single-modality ablations with `--disable-text`, `--disable-topology`, `--disable-temporal`, or `--disable-user`.
