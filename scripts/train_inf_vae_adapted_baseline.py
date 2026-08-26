import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v1_dataset import V1InputDataset


DEFAULT_OUTPUT_DIR = Path("results/paper_baselines/inf_vae_adapted")


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer value, got {raw!r}")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one numeric value, got {raw!r}")
    return values


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    denom = np.maximum(np.abs(y_true), 1.0)
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
    return {
        "num_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(abs_errors / denom)),
        "smape": float(np.mean(2.0 * abs_errors / smape_denom)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "median_ae": float(np.median(abs_errors)),
    }


def graph_summary(example: dict[str, Any]) -> np.ndarray:
    nodes = np.asarray(example["node_features"], dtype=np.float32)
    edges = np.asarray(example["edge_index"], dtype=np.int64)
    if nodes.size == 0:
        node_mean = np.zeros(8, dtype=np.float32)
        node_std = np.zeros(8, dtype=np.float32)
        node_max = np.zeros(8, dtype=np.float32)
        node_count = 0.0
    else:
        node_mean = nodes.mean(axis=0)
        node_std = nodes.std(axis=0)
        node_max = nodes.max(axis=0)
        node_count = float(nodes.shape[0])
    edge_count = float(edges.shape[1])
    density = edge_count / max(node_count * max(node_count - 1.0, 1.0), 1.0)
    return np.concatenate(
        [
            node_mean,
            node_std,
            node_max,
            np.asarray(
                [
                    math.log1p(node_count),
                    math.log1p(edge_count),
                    edge_count / max(node_count, 1.0),
                    density,
                    float(example["observed_size"]),
                    math.log1p(float(example["observed_size"])),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def flatten_temporal(example: dict[str, Any]) -> np.ndarray:
    temporal = np.asarray(example["temporal_features"], dtype=np.float32)
    mask = np.asarray(example["temporal_mask"], dtype=np.float32)
    if temporal.ndim == 2 and mask.ndim == 1 and len(mask) == temporal.shape[0]:
        temporal = temporal * mask[:, None]
    return temporal.reshape(-1).astype(np.float32)


def sample_features(example: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(example["text_features"], dtype=np.float32).reshape(-1),
            np.asarray(example["global_features"], dtype=np.float32).reshape(-1),
            flatten_temporal(example),
            np.asarray(example["user_features"], dtype=np.float32).reshape(-1),
            np.asarray(example["modality_mask"], dtype=np.float32).reshape(-1),
            graph_summary(example),
        ]
    ).astype(np.float32)


def load_split_rows(args: argparse.Namespace, split: str) -> dict[str, Any]:
    dataset = V1InputDataset(
        dataset=args.dataset,
        observation=args.observation,
        split=split,
        split_strategy=args.split_strategy,
        seed=args.split_seed,
        task=args.label_task,
        input_root=args.input_root,
        split_root=args.split_root,
        text_feature_path=args.text_feature_path,
    )
    features = []
    y = []
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        example = dataset[index]
        features.append(sample_features(example))
        y.append(float(example["final_size"]))
        rows.append(
            {
                "dataset": args.dataset,
                "split_strategy": args.split_strategy,
                "split": split,
                "sample_id": example["sample_id"],
                "raw_label": example["raw_label"],
                "label_id": example["label_id"],
                "final_size": float(example["final_size"]),
                "observed_size": float(example["observed_size"]),
            }
        )
    return {
        "x": np.stack(features).astype(np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "rows": rows,
        "summary": dataset.summary(),
    }


def load_data(args: argparse.Namespace) -> dict[str, Any]:
    return {split: load_split_rows(args, split) for split in ("train", "val", "test")}


class InfVAERegressor(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, latent_dim * 2)),
            nn.ReLU(),
        )
        encoded_dim = max(hidden_dim // 2, latent_dim * 2)
        self.mu = nn.Linear(encoded_dim, latent_dim)
        self.logvar = nn.Linear(encoded_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoded_dim),
            nn.ReLU(),
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 16)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim * 2, 16), 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)
        return pred, recon, mu, logvar


def train_one_model(
    args: argparse.Namespace,
    data: dict[str, Any],
    seed: int,
    latent_dim: int,
) -> tuple[InfVAERegressor, StandardScaler, StandardScaler, dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(data["train"]["x"]).astype(np.float32)
    y_train_log = np.log1p(data["train"]["y"]).reshape(-1, 1)
    y_train = y_scaler.fit_transform(y_train_log).reshape(-1).astype(np.float32)

    x_val = x_scaler.transform(data["val"]["x"]).astype(np.float32)
    y_val_log = np.log1p(data["val"]["y"]).reshape(-1, 1)
    y_val = y_scaler.transform(y_val_log).reshape(-1).astype(np.float32)

    model = InfVAERegressor(
        input_dim=x_train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    mse = nn.MSELoss()

    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred, recon, mu, logvar = model(xb)
            pred_loss = mse(pred, yb)
            recon_loss = mse(recon, xb)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = pred_loss + args.reconstruction_weight * recon_loss + args.kl_weight * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_scaled, _recon, _mu, _logvar = model(torch.from_numpy(x_val))
            val_pred_log = y_scaler.inverse_transform(val_pred_scaled.numpy().reshape(-1, 1)).reshape(-1)
            val_pred = np.clip(np.expm1(val_pred_log), 1.0, None)
            val_mape = regression_metrics(data["val"]["y"], val_pred)["mape"]
        if val_mape + 1e-7 < best_val:
            best_val = val_mape
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for split in ("train", "val", "test"):
            x_split = x_scaler.transform(data[split]["x"]).astype(np.float32)
            pred_scaled, recon, mu, logvar = model(torch.from_numpy(x_split))
            pred_log = y_scaler.inverse_transform(pred_scaled.numpy().reshape(-1, 1)).reshape(-1)
            pred = np.clip(np.expm1(pred_log), 1.0, None)
            split_metrics = regression_metrics(data[split]["y"], pred)
            split_metrics.update(
                {
                    "observation_window_minutes": args.observation,
                    "latent_dim": latent_dim,
                    "reconstruction_mse": float(torch.mean((recon - torch.from_numpy(x_split)) ** 2).item()),
                    "kl_divergence": float(
                        (-0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())).item()
                    ),
                }
            )
            metrics[split] = split_metrics
            predictions[split] = pred
    return model, x_scaler, y_scaler, metrics, predictions


def run(args: argparse.Namespace, data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or load_data(args)
    latent_dims = parse_int_list(args.latent_dims)
    models_payload: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []

    for latent_dim in latent_dims:
        model_name = f"inf_vae_adapted_z{latent_dim}_w{int(args.observation)}m"
        _model, _x_scaler, _y_scaler, metrics, predictions = train_one_model(
            args,
            data,
            args.seed,
            latent_dim,
        )
        models_payload[model_name] = metrics
        for split in ("train", "val", "test"):
            for row, pred in zip(data[split]["rows"], predictions[split]):
                truth = float(row["final_size"])
                all_predictions.append(
                    {
                        **row,
                        "model": model_name,
                        "latent_dim": latent_dim,
                        "observation_window_minutes": args.observation,
                        "pred_final_size": float(pred),
                        "abs_error": float(abs(pred - truth)),
                        "absolute_percentage_error": float(abs(pred - truth) / max(truth, 1.0)),
                    }
                )

    payload = {
        "dataset": args.dataset,
        "task": "cascade_size_prediction",
        "label_task": args.label_task,
        "split_strategy": args.split_strategy,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "model_family": "inf_vae_adapted",
        "target": "final_cascade_size",
        "observation_windows_minutes": [args.observation],
        "input_summary": data["train"]["summary"],
        "hyperparameters": {
            "latent_dims": latent_dims,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "kl_weight": args.kl_weight,
            "reconstruction_weight": args.reconstruction_weight,
        },
        "notes": [
            "This is an Inf-VAE-adapted baseline using V1 text/topology/temporal/user proxy inputs.",
            "It approximates influence and homophily with available cascade graph, community/time-window, and source-user features.",
            "It is not an original-code reproduction because complete user-user social links and original influence labels are unavailable.",
        ],
        "models": models_payload,
    }

    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset}_cascade_size_{args.split_strategy}_inf_vae_adapted_seed{args.seed}"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    write_json(metrics_path, payload)
    write_csv(
        predictions_path,
        all_predictions,
        [
            "dataset",
            "split_strategy",
            "split",
            "sample_id",
            "raw_label",
            "label_id",
            "final_size",
            "observed_size",
            "model",
            "latent_dim",
            "observation_window_minutes",
            "pred_final_size",
            "abs_error",
            "absolute_percentage_error",
        ],
    )
    payload["outputs"] = {"metrics": str(metrics_path), "predictions": str(predictions_path)}
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pheme", "twitter15", "twitter16", "weibo"])
    parser.add_argument("--label-task", default="rumor_binary", choices=["rumor_binary", "veracity", "raw"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--observation", type=int, default=180)
    parser.add_argument("--latent-dims", default="8,16")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--reconstruction-weight", type=float, default=0.10)
    parser.add_argument("--input-root", default="data/processed/v1_inputs")
    parser.add_argument("--split-root", default="data/processed/splits")
    parser.add_argument("--text-feature-path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    seeds = parse_int_list(args.seeds) if args.seeds else [args.seed]
    data = load_data(args)
    outputs = []
    for seed in seeds:
        args.seed = seed
        payload = run(args, data=data)
        outputs.append(payload["outputs"])
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split_strategy": args.split_strategy,
                "seeds": seeds,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
