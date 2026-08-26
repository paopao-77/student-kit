import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from models.heterorumor_v1 import HeteroRumorV1
from models.heterorumor_v2_c1 import HeteroRumorV2C1
from models.heterorumor_v2_c1_disentangled import HeteroRumorV2C1Disentangled
from v1_dataset import V1InputDataset, collate_v1_batch


DEFAULT_OUTPUT_DIR = Path("results/heterorumor_v1")
MODALITY_NAMES = ["text", "topology", "temporal", "user_profile"]
SUPPORTED_DATASETS = [
    "pheme",
    "twitter15",
    "twitter16",
    "twitter15_rumdetect2017",
    "twitter16_rumdetect2017",
    "weibo",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def selected_indices(length: int, limit: int, seed: int) -> list[int]:
    indices = list(range(length))
    if limit <= 0 or limit >= length:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return sorted(indices[:limit])


def iter_index_batches(
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    rng: random.Random,
) -> list[list[int]]:
    ordered = list(indices)
    if shuffle:
        rng.shuffle(ordered)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def perturb_text_features(
    features: torch.Tensor,
    noise_scale: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if noise_scale <= 0:
        return features
    if not 0.0 <= noise_scale < 1.0:
        raise ValueError("text corruption rate must be in [0, 1)")
    random_values = torch.rand(
        features.shape,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    )
    keep_mask = (random_values >= noise_scale).to(features.dtype)
    perturbed = features * keep_mask
    empty_rows = perturbed.abs().sum(dim=1) <= 1e-12
    if empty_rows.any():
        perturbed[empty_rows] = features[empty_rows]
    return nn.functional.normalize(perturbed, p=2, dim=1)


def matched_text_replacement(
    batch: dict[str, Any],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = batch["text_features"]
    batch_size = features.shape[0]
    if batch_size < 2:
        return features, torch.zeros(batch_size, dtype=features.dtype, device=features.device)
    targets = growth_targets(batch)
    target_distance = torch.abs(targets[:, None] - targets[None, :])
    target_distance.fill_diagonal_(float("inf"))
    candidate_count = min(max(top_k, 1), batch_size - 1)
    candidates = torch.topk(
        target_distance,
        k=candidate_count,
        largest=False,
        dim=1,
    ).indices
    normalized = nn.functional.normalize(features, p=2, dim=1)
    candidate_features = normalized[candidates]
    cosine_similarity = (normalized[:, None, :] * candidate_features).sum(dim=2)
    selected_position = cosine_similarity.argmin(dim=1)
    row_indices = torch.arange(batch_size, device=features.device)
    partners = candidates[row_indices, selected_position]
    return features[partners], torch.abs(targets - targets[partners])


def apply_ablation(batch: dict[str, Any], args: argparse.Namespace) -> None:
    disabled = [
        args.disable_text,
        args.disable_topology,
        args.disable_temporal,
        args.disable_user,
    ]
    for index, is_disabled in enumerate(disabled):
        if is_disabled:
            batch["modality_mask"][:, index] = 0.0


def make_batch(
    dataset: V1InputDataset,
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
    text_noise_scale: float = 0.0,
) -> dict[str, Any]:
    examples = [dataset[index] for index in indices]
    batch = collate_v1_batch(examples, as_torch=True)
    apply_ablation(batch, args)
    moved = move_batch(batch, device)
    if text_noise_scale > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed * 1_000_003 + sum(indices))
        moved["text_features"] = perturb_text_features(
            moved["text_features"], text_noise_scale, generator
        )
    return moved


def growth_targets(batch: dict[str, Any]) -> torch.Tensor:
    growth = (batch["final_sizes"] - batch["observed_sizes"]).clamp_min(0.0)
    return torch.log1p(growth)


def prediction_loss_components(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_fn: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = growth_targets(batch)
    regression_loss = loss_fn(output["log_growth"], target)
    has_growth = (batch["final_sizes"] > batch["observed_sizes"]).float()
    classification_loss = nn.functional.binary_cross_entropy_with_logits(
        output["growth_logit"], has_growth
    )
    relative_loss = torch.mean(
        torch.abs(output["predicted_final_size"] - batch["final_sizes"])
        / batch["final_sizes"].clamp_min(1.0)
    )
    loss = (
        regression_loss
        + args.growth_classification_weight * classification_loss
        + args.relative_loss_weight * relative_loss
    )
    return loss, {
        "regression": regression_loss,
        "growth_classification": classification_loss,
        "relative": relative_loss,
    }


def module_gradient_norms(model: HeteroRumorV1) -> dict[str, float]:
    result = {}
    modules = {
        "text": model.text_encoder,
        "topology": model.graph_encoder,
        "temporal": model.temporal_encoder,
        "user": model.user_encoder,
        "fusion": model.fusion,
        "prediction": model.prediction_head,
        "growth_probability": model.growth_probability_head,
    }
    if isinstance(model, HeteroRumorV2C1):
        modules.update(
            {
                "factor_encoder": model.factor_encoder,
                "factor_mu": model.factor_mu,
                "factor_logvar": model.factor_logvar,
                "factor_decoder": model.factor_decoder,
                "latent_prediction": model.latent_prediction_head,
                "latent_growth_probability": model.latent_growth_probability_head,
            }
        )
    if isinstance(model, HeteroRumorV2C1Disentangled):
        modules.update(
            {
                "content_factor_encoder": model.content_factor_encoder,
                "dynamics_factor_encoder": model.dynamics_factor_encoder,
                "content_decoder": model.content_decoder,
                "dynamics_decoder": model.dynamics_decoder,
                "disentangled_prediction": model.disentangled_prediction_head,
            }
        )
    for name, module in modules.items():
        squared = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().pow(2).sum().item())
        result[name] = math.sqrt(squared)
    return result


def train_one_epoch(
    model: HeteroRumorV1,
    dataset: V1InputDataset,
    indices: list[int],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
    epoch: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    gradient_totals: dict[str, float] = {}
    num_batches = 0
    component_totals: dict[str, float] = {}
    for batch_indices in iter_index_batches(indices, args.batch_size, True, rng):
        batch = make_batch(dataset, batch_indices, args, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, components = prediction_loss_components(output, batch, loss_fn, args)
        if "latent_mu" in output:
            reconstruction_loss = nn.functional.mse_loss(
                output["reconstruction"], output["reconstruction_target"]
            )
            vae_mu = output.get("vae_mu", output["latent_mu"])
            vae_logvar = output.get("vae_logvar", output["latent_logvar"])
            kl_loss = -0.5 * torch.mean(
                1.0
                + vae_logvar
                - vae_mu.pow(2)
                - vae_logvar.exp()
            )
            kl_scale = min(1.0, epoch / max(args.kl_warmup_epochs, 1))
            loss = (
                loss
                + args.reconstruction_weight * reconstruction_loss
                + args.kl_weight * kl_scale * kl_loss
            )
            components["reconstruction"] = reconstruction_loss
            components["kl"] = kl_loss
            components["kl_scale"] = torch.as_tensor(kl_scale, device=device)
            if "content_latent_mu" in output:
                dynamics_centered = output["latent_mu"] - output["latent_mu"].mean(dim=0)
                content_centered = (
                    output["content_latent_mu"] - output["content_latent_mu"].mean(dim=0)
                )
                cross_covariance = dynamics_centered.T @ content_centered / max(
                    dynamics_centered.shape[0] - 1, 1
                )
                disentangle_loss = cross_covariance.pow(2).mean()
                loss = loss + args.disentangle_weight * disentangle_loss
                components["disentangle_cross_covariance"] = disentangle_loss
        if args.counterfactual_weight > 0:
            counterfactual_batch = dict(batch)
            if args.counterfactual_mode == "matched_swap":
                replacement, target_gap = matched_text_replacement(
                    batch, args.counterfactual_match_top_k
                )
                counterfactual_batch["text_features"] = replacement
                components["counterfactual_mean_target_gap"] = target_gap.mean()
            else:
                counterfactual_batch["text_features"] = perturb_text_features(
                    batch["text_features"], args.counterfactual_text_noise
                )
            counterfactual_output = model(counterfactual_batch)
            counterfactual_supervised, _ = prediction_loss_components(
                counterfactual_output, counterfactual_batch, loss_fn, args
            )
            prediction_consistency = nn.functional.smooth_l1_loss(
                counterfactual_output["log_growth"], output["log_growth"].detach()
            )
            probability_consistency = nn.functional.mse_loss(
                counterfactual_output["growth_probability"],
                output["growth_probability"].detach(),
            )
            counterfactual_loss = (
                args.counterfactual_supervised_weight * counterfactual_supervised
                + prediction_consistency
                + args.counterfactual_probability_weight * probability_consistency
            )
            components["counterfactual_supervised"] = counterfactual_supervised
            components["counterfactual_prediction_consistency"] = prediction_consistency
            components["counterfactual_probability_consistency"] = probability_consistency
            if "latent_mu" in output:
                latent_consistency = nn.functional.smooth_l1_loss(
                    counterfactual_output["latent_mu"], output["latent_mu"].detach()
                )
                counterfactual_loss = (
                    counterfactual_loss
                    + args.counterfactual_latent_weight * latent_consistency
                )
                components["counterfactual_latent_consistency"] = latent_consistency
            if "content_latent_mu" in output:
                content_change = nn.functional.cosine_similarity(
                    counterfactual_output["content_latent_mu"],
                    output["content_latent_mu"].detach(),
                    dim=1,
                ).mean()
                counterfactual_loss = (
                    counterfactual_loss
                    + args.counterfactual_content_change_weight * content_change
                )
                components["counterfactual_content_similarity"] = content_change
            loss = loss + args.counterfactual_weight * counterfactual_loss
        loss.backward()
        norms = module_gradient_norms(model)
        for name, value in norms.items():
            gradient_totals[name] = gradient_totals.get(name, 0.0) + value
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_indices)
        total_examples += len(batch_indices)
        num_batches += 1
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(value.item())
    mean_norms = {name: value / max(num_batches, 1) for name, value in gradient_totals.items()}
    mean_components = {
        name: value / max(num_batches, 1) for name, value in component_totals.items()
    }
    return total_loss / max(total_examples, 1), mean_norms, mean_components


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    denominator = np.maximum(np.abs(y_true), 1.0)
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
    total_ss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_ss = float(np.sum(errors**2))
    return {
        "num_samples": int(len(y_true)),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(absolute_errors / denominator)),
        "smape": float(np.mean(2.0 * absolute_errors / smape_denominator)),
        "r2": None if total_ss <= 1e-12 else float(1.0 - residual_ss / total_ss),
        "median_ae": float(np.median(absolute_errors)),
        "mean_final_size": float(np.mean(y_true)),
        "mean_predicted_size": float(np.mean(y_pred)),
    }


@torch.no_grad()
def evaluate(
    model: HeteroRumorV1,
    dataset: V1InputDataset,
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
    split: str,
    growth_threshold: float | None = None,
    text_noise_scale: float = 0.0,
    matched_text_swap: bool = False,
) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    rows = []
    fusion_weight_sum = np.zeros(4, dtype=np.float64)
    num_examples = 0
    latent_parts = []
    content_latent_parts = []
    matched_target_gap_total = 0.0
    for batch_indices in iter_index_batches(
        indices, args.batch_size, False, random.Random(args.seed)
    ):
        batch = make_batch(
            dataset,
            batch_indices,
            args,
            device,
            text_noise_scale=text_noise_scale,
        )
        if matched_text_swap:
            replacement, target_gap = matched_text_replacement(
                batch, args.counterfactual_match_top_k
            )
            batch["text_features"] = replacement
            matched_target_gap_total += float(target_gap.sum().item())
        output = model(batch)
        predicted_final_size = output["predicted_final_size"]
        threshold = args.growth_threshold if growth_threshold is None else growth_threshold
        if threshold > 0:
            predicted_final_size = torch.where(
                output["growth_probability"] >= threshold,
                predicted_final_size,
                batch["observed_sizes"],
            )
        predictions = predicted_final_size.cpu().numpy()
        targets = batch["final_sizes"].cpu().numpy()
        observed = batch["observed_sizes"].cpu().numpy()
        weights = output["fusion_weights"].cpu().numpy()
        growth_probabilities = output["growth_probability"].cpu().numpy()
        latent_mu = output.get("latent_mu")
        latent_values = latent_mu.cpu().numpy() if latent_mu is not None else None
        content_latent_mu = output.get("content_latent_mu")
        content_latent_values = (
            content_latent_mu.cpu().numpy() if content_latent_mu is not None else None
        )
        if latent_values is not None:
            latent_parts.append(latent_values)
        if content_latent_values is not None:
            content_latent_parts.append(content_latent_values)
        fusion_weight_sum += weights.sum(axis=0)
        num_examples += len(batch_indices)
        for local_index, dataset_index in enumerate(batch_indices):
            example = dataset[dataset_index]
            target = float(targets[local_index])
            prediction = float(predictions[local_index])
            y_true.append(target)
            y_pred.append(prediction)
            row = {
                    "dataset": args.dataset,
                    "split_strategy": args.split_strategy,
                    "split": split,
                    "model": model_name(args),
                    "observation_window_minutes": args.observation,
                    "sample_id": example["sample_id"],
                    "raw_label": example["raw_label"],
                    "label_id": example["label_id"],
                    "observed_size": float(observed[local_index]),
                    "final_size": target,
                    "pred_final_size": prediction,
                    "abs_error": abs(prediction - target),
                    "absolute_percentage_error": abs(prediction - target) / max(target, 1.0),
                    "growth_probability": float(growth_probabilities[local_index]),
                    "text_weight": float(weights[local_index, 0]),
                    "topology_weight": float(weights[local_index, 1]),
                    "temporal_weight": float(weights[local_index, 2]),
                    "user_weight": float(weights[local_index, 3]),
                }
            if latent_values is not None:
                for factor_index, value in enumerate(latent_values[local_index]):
                    row[f"factor_{factor_index}"] = float(value)
            if content_latent_values is not None:
                for factor_index, value in enumerate(content_latent_values[local_index]):
                    row[f"content_factor_{factor_index}"] = float(value)
            rows.append(row)
    true_array = np.asarray(y_true, dtype=np.float64)
    pred_array = np.asarray(y_pred, dtype=np.float64)
    metrics = regression_metrics(true_array, pred_array)
    metrics["observation_window_minutes"] = args.observation
    metrics["mean_fusion_weights"] = {
        name: float(fusion_weight_sum[index] / max(num_examples, 1))
        for index, name in enumerate(MODALITY_NAMES)
    }
    metrics["text_noise_scale"] = text_noise_scale
    metrics["matched_text_swap"] = matched_text_swap
    if matched_text_swap:
        metrics["mean_matched_target_gap"] = matched_target_gap_total / max(
            num_examples, 1
        )
    if latent_parts:
        latent_array = np.concatenate(latent_parts, axis=0)
        factor_std = latent_array.std(axis=0)
        metrics["latent_dim"] = int(latent_array.shape[1])
        metrics["latent_mean_abs"] = float(np.abs(latent_array.mean(axis=0)).mean())
        metrics["latent_std_mean"] = float(factor_std.mean())
        metrics["active_latent_factors"] = int(np.sum(factor_std > args.active_factor_std))
    if content_latent_parts:
        content_array = np.concatenate(content_latent_parts, axis=0)
        content_std = content_array.std(axis=0)
        metrics["content_latent_dim"] = int(content_array.shape[1])
        metrics["content_latent_std_mean"] = float(content_std.mean())
        metrics["active_content_factors"] = int(
            np.sum(content_std > args.active_factor_std)
        )
        if latent_parts:
            dynamics_centered = latent_array - latent_array.mean(axis=0, keepdims=True)
            content_centered = content_array - content_array.mean(axis=0, keepdims=True)
            cross_covariance = dynamics_centered.T @ content_centered / max(
                len(latent_array) - 1, 1
            )
            metrics["content_dynamics_cross_covariance_mse"] = float(
                np.mean(cross_covariance**2)
            )
    return {"metrics": metrics, "predictions": rows}


def model_name(args: argparse.Namespace) -> str:
    disabled = []
    if args.disable_text:
        disabled.append("text")
    if args.disable_topology:
        disabled.append("topology")
    if args.disable_temporal:
        disabled.append("temporal")
    if args.disable_user:
        disabled.append("user")
    if args.model_version == "v2_c1_disentangled":
        base = (
            f"heterorumor_v2_c1_disentangled_d{args.latent_dim}"
            f"_c{args.content_latent_dim}"
        )
    elif args.model_version == "v2_c1_vae":
        base = f"heterorumor_v2_c1_vae_k{args.latent_dim}"
    else:
        base = "heterorumor_v1_hurdle"
    if args.text_feature_name != "hash":
        base += "_" + args.text_feature_name
    if args.run_tag:
        base += "_" + args.run_tag
    return base if not disabled else base + "_wo_" + "_".join(disabled)


def select_growth_threshold(
    model: HeteroRumorV1,
    dataset: V1InputDataset,
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    candidates = np.linspace(0.0, 0.9, 19)
    search = []
    for threshold in candidates:
        result = evaluate(
            model,
            dataset,
            indices,
            args,
            device,
            split="val",
            growth_threshold=float(threshold),
        )
        search.append({"threshold": float(threshold), "mape": float(result["metrics"]["mape"])})
    best = min(search, key=lambda row: (row["mape"], row["threshold"]))
    return float(best["threshold"]), search


def build_model(dataset: V1InputDataset, args: argparse.Namespace) -> HeteroRumorV1:
    arrays = dataset.arrays
    if args.model_version == "v2_c1_disentangled":
        model_class = HeteroRumorV2C1Disentangled
    elif args.model_version == "v2_c1_vae":
        model_class = HeteroRumorV2C1
    else:
        model_class = HeteroRumorV1
    kwargs = dict(
        text_dim=int(arrays["text_features"].shape[1]),
        node_dim=int(arrays["node_features"].shape[1]),
        global_dim=int(arrays["global_features"].shape[1]),
        temporal_dim=int(arrays["temporal_features"].shape[2]),
        user_dim=int(arrays["user_features"].shape[1]),
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
    )
    if model_class in (HeteroRumorV2C1, HeteroRumorV2C1Disentangled):
        kwargs["latent_dim"] = args.latent_dim
    if model_class is HeteroRumorV2C1Disentangled:
        kwargs["content_latent_dim"] = args.content_latent_dim
    return model_class(**kwargs)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split_strategy",
        "split",
        "model",
        "observation_window_minutes",
        "sample_id",
        "raw_label",
        "label_id",
        "observed_size",
        "final_size",
        "pred_final_size",
        "abs_error",
        "absolute_percentage_error",
        "growth_probability",
        "text_weight",
        "topology_weight",
        "temporal_weight",
        "user_weight",
    ]
    factor_fields = sorted(
        {key for row in rows for key in row if key.startswith("factor_")},
        key=lambda name: int(name.split("_")[1]),
    )
    content_factor_fields = sorted(
        {key for row in rows for key in row if key.startswith("content_factor_")},
        key=lambda name: int(name.split("_")[2]),
    )
    fieldnames.extend([*factor_fields, *content_factor_fields])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    datasets = {
        split: V1InputDataset(
            dataset=args.dataset,
            observation=args.observation,
            split=split,
            split_strategy=args.split_strategy,
            seed=args.split_seed,
            task=args.task,
            input_root=args.input_root,
            text_feature_path=args.text_feature_path,
        )
        for split in ("train", "val", "test")
    }
    indices = {
        "train": selected_indices(len(datasets["train"]), args.limit_train, args.seed),
        "val": selected_indices(len(datasets["val"]), args.limit_val, args.seed + 1),
        "test": selected_indices(len(datasets["test"]), args.limit_test, args.seed + 2),
    }
    model = build_model(datasets["train"], args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)
    rng = random.Random(args.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_val_mape = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, gradient_norms, loss_components = train_one_epoch(
            model=model,
            dataset=datasets["train"],
            indices=indices["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            args=args,
            device=device,
            rng=rng,
            epoch=epoch,
        )
        val_result = evaluate(
            model, datasets["val"], indices["val"], args, device, split="val"
        )
        val_mape = float(val_result["metrics"]["mape"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_result["metrics"],
                "gradient_norms": gradient_norms,
                "loss_components": loss_components,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        if val_mape < best_val_mape - args.min_delta:
            best_val_mape = val_mape
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    requested_growth_threshold = args.growth_threshold
    threshold_search = []
    if not args.no_threshold_tuning:
        args.growth_threshold, threshold_search = select_growth_threshold(
            model,
            datasets["val"],
            indices["val"],
            args,
            device,
        )
    split_results = {
        split: evaluate(model, datasets[split], indices[split], args, device, split=split)
        for split in ("train", "val", "test")
    }
    robustness_results = {}
    for noise_scale in args.robustness_noise_scales:
        noisy_result = evaluate(
            model,
            datasets["test"],
            indices["test"],
            args,
            device,
            split="test",
            growth_threshold=args.growth_threshold,
            text_noise_scale=noise_scale,
        )
        robustness_results[f"text_noise_{noise_scale:g}"] = noisy_result["metrics"]
    matched_swap_result = evaluate(
        model,
        datasets["test"],
        indices["test"],
        args,
        device,
        split="test",
        growth_threshold=args.growth_threshold,
        matched_text_swap=True,
    )
    robustness_results["matched_text_swap"] = matched_swap_result["metrics"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "dataset": args.dataset,
        "task": "cascade_size_prediction",
        "label_task": args.task,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "model_family": (
            "heterorumor_v2_c1"
            if args.model_version in ("v2_c1_vae", "v2_c1_disentangled")
            else "heterorumor_v1"
        ),
        "model_type": model_name(args),
        "observation_window_minutes": args.observation,
        "device": str(device),
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "relative_loss_weight": args.relative_loss_weight,
        "growth_classification_weight": args.growth_classification_weight,
        "growth_threshold": args.growth_threshold,
        "text_feature_name": args.text_feature_name,
        "text_feature_path": args.text_feature_path,
        "model_version": args.model_version,
        "latent_dim": (
            args.latent_dim
            if args.model_version in ("v2_c1_vae", "v2_c1_disentangled")
            else None
        ),
        "content_latent_dim": (
            args.content_latent_dim if args.model_version == "v2_c1_disentangled" else None
        ),
        "kl_weight": (
            args.kl_weight
            if args.model_version in ("v2_c1_vae", "v2_c1_disentangled")
            else None
        ),
        "reconstruction_weight": (
            args.reconstruction_weight
            if args.model_version in ("v2_c1_vae", "v2_c1_disentangled")
            else None
        ),
        "run_tag": args.run_tag,
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_text_noise": args.counterfactual_text_noise,
        "counterfactual_mode": args.counterfactual_mode,
        "disentangle_weight": args.disentangle_weight,
        "requested_growth_threshold": requested_growth_threshold,
        "threshold_tuned_on": None if args.no_threshold_tuning else "val",
        "threshold_search": threshold_search,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_mape": best_val_mape,
        "parameter_count": parameter_count,
        "training_seconds": time.perf_counter() - start_time,
        "disabled_modalities": [
            name
            for name, disabled in zip(
                MODALITY_NAMES,
                [args.disable_text, args.disable_topology, args.disable_temporal, args.disable_user],
            )
            if disabled
        ],
        "num_samples": {split: len(indices[split]) for split in indices},
        "split_summaries": {
            split: {
                **datasets[split].summary(),
                "loaded_num_samples": len(indices[split]),
            }
            for split in datasets
        },
        "models": {
            model_name(args): {
                split: split_results[split]["metrics"] for split in split_results
            }
        },
        "history": history,
        "robustness": robustness_results,
    }

    ablation_suffix = ""
    if result["disabled_modalities"]:
        ablation_suffix = "_wo_" + "_".join(result["disabled_modalities"])
    prefix = (
        f"{args.dataset}_cascade_size_{args.split_strategy}_{model_name(args)}"
        f"_obs{args.observation}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / f"{prefix}_metrics.json"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    checkpoint_path = output_dir / f"{prefix}_checkpoint.pt"
    write_json(metrics_path, result)
    write_predictions(predictions_path, split_results["test"]["predictions"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "graph_layers": args.graph_layers,
                "dropout": args.dropout,
                "growth_threshold": args.growth_threshold,
                "text_feature_name": args.text_feature_name,
                "text_feature_path": args.text_feature_path,
                "model_version": args.model_version,
                "latent_dim": args.latent_dim,
                "content_latent_dim": args.content_latent_dim,
                "kl_weight": args.kl_weight,
                "reconstruction_weight": args.reconstruction_weight,
                "run_tag": args.run_tag,
                "counterfactual_weight": args.counterfactual_weight,
                "counterfactual_text_noise": args.counterfactual_text_noise,
                "counterfactual_mode": args.counterfactual_mode,
                "disentangle_weight": args.disentangle_weight,
            },
            "dataset": args.dataset,
            "observation": args.observation,
            "split_strategy": args.split_strategy,
            "seed": args.seed,
            "split_seed": args.split_seed,
        },
        checkpoint_path,
    )
    result["outputs"] = {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "checkpoint": str(checkpoint_path),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pheme", choices=SUPPORTED_DATASETS)
    parser.add_argument("--observation", type=int, default=180)
    parser.add_argument("--task", default="rumor_binary")
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--input-root", default="data/processed/v1_inputs")
    parser.add_argument("--text-feature-name", default="hash")
    parser.add_argument("--text-feature-path")
    parser.add_argument(
        "--model-version",
        default="v1",
        choices=["v1", "v2_c1_vae", "v2_c1_disentangled"],
    )
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--content-latent-dim", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--active-factor-std", type=float, default=0.05)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument(
        "--counterfactual-mode", default="mask", choices=["mask", "matched_swap"]
    )
    parser.add_argument("--counterfactual-text-noise", type=float, default=0.2)
    parser.add_argument("--counterfactual-match-top-k", type=int, default=8)
    parser.add_argument("--counterfactual-supervised-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-probability-weight", type=float, default=0.25)
    parser.add_argument("--counterfactual-latent-weight", type=float, default=0.1)
    parser.add_argument("--counterfactual-content-change-weight", type=float, default=0.05)
    parser.add_argument("--disentangle-weight", type=float, default=0.1)
    parser.add_argument("--robustness-noise-scales", default="0.1,0.2,0.3")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--relative-loss-weight", type=float, default=0.5)
    parser.add_argument("--growth-classification-weight", type=float, default=0.25)
    parser.add_argument("--growth-threshold", type=float, default=0.5)
    parser.add_argument("--no-threshold-tuning", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--disable-topology", action="store_true")
    parser.add_argument("--disable-temporal", action="store_true")
    parser.add_argument("--disable-user", action="store_true")
    args = parser.parse_args()
    args.robustness_noise_scales = [
        float(value.strip())
        for value in args.robustness_noise_scales.split(",")
        if value.strip()
    ]
    return args


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    result = run(parse_args())
    model_key = result["model_type"]
    compact = {
        "dataset": result["dataset"],
        "split_strategy": result["split_strategy"],
        "observation": result["observation_window_minutes"],
        "model": model_key,
        "epochs_ran": result["epochs_ran"],
        "best_epoch": result["best_epoch"],
        "best_val_mape": result["best_val_mape"],
        "outputs": result["outputs"],
        "test_metrics": result["models"][model_key]["test"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
