import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import RumorDataset


DEFAULT_RESULTS_DIR = Path("results/paper_baselines/dynamics")
DEFAULT_OBSERVATION_WINDOWS = "60,180,360"
DEFAULT_CAPACITY_MULTIPLIERS = "1,1.5,2,3,5,8,13"
DEFAULT_BETA_GRID = "0.05,0.1,0.2,0.4,0.8,1.2"
DEFAULT_GAMMA_GRID = "0.05,0.1,0.2,0.4,0.8"
DEFAULT_SIGMA_GRID = "0.1,0.3,0.6"
DEFAULT_B_GRID = "0.02,0.05,0.1,0.2,0.4"
DEFAULT_RHO_GRID = "0.05,0.2,0.5"
DEFAULT_EPSILON_GRID = "0.05,0.2,0.5"
DEFAULT_P_GRID = "0.25,0.5,0.75"
DEFAULT_L_GRID = "0.25,0.5,0.75"


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"Expected at least one numeric value, got {raw!r}")
    return values


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"early_count", "sir", "seir", "seiz"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown dynamics models: {unknown}; allowed={sorted(allowed)}")
    return models


def finite_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_event_delays(event_path: Path, sample_ids: set[str], dataset: str) -> dict[str, list[float]]:
    delays_by_sample: dict[str, list[float]] = {sample_id: [] for sample_id in sample_ids}
    with event_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            if row.get("tweet_id") == "ROOT" or row.get("user_id") == "ROOT":
                continue
            raw_delay = row.get("delay_minutes", "")
            if raw_delay == "" and dataset == "weibo":
                raw_delay = row.get("event_order", "")
            if raw_delay == "":
                continue
            value = finite_float(raw_delay, -1.0)
            if value >= 0:
                delays_by_sample[sample_id].append(value)
    for sample_id, delays in delays_by_sample.items():
        if not delays:
            delays.append(0.0)
        delays.sort()
    return delays_by_sample


def load_split_dataset(args: argparse.Namespace, split: str) -> RumorDataset:
    return RumorDataset(
        dataset=args.dataset,
        data_root=args.data_root,
        label_map_path=args.label_map,
        task=args.label_task,
        split=split,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )


def observed_state(
    delays: list[float],
    final_size: int,
    observation_window: float,
    active_window: float,
) -> tuple[int, int, int, float]:
    observed = sum(1 for delay in delays if delay <= observation_window)
    observed = max(1, min(int(final_size), observed))

    active_start = max(0.0, observation_window - active_window)
    active = sum(1 for delay in delays if active_start < delay <= observation_window)
    active = min(observed, active)
    if observation_window <= active_window:
        active = max(active, observed)

    inactive = max(observed - active, 0)
    max_observed_delay = max((delay for delay in delays if delay <= observation_window), default=0.0)
    return observed, active, inactive, max_observed_delay


def make_cases(
    dataset: RumorDataset,
    delays_by_sample: dict[str, list[float]],
    observation_window: float,
    active_window: float,
) -> list[dict[str, Any]]:
    cases = []
    for sample in dataset:
        final_size = max(int(sample.get("num_nodes") or 1), 1)
        delays = delays_by_sample.get(sample["sample_id"], [0.0])
        observed, active, inactive, max_observed_delay = observed_state(
            delays=delays,
            final_size=final_size,
            observation_window=observation_window,
            active_window=active_window,
        )
        max_delay = max(delays) if delays else 0.0
        cases.append(
            {
                "dataset": sample["dataset"],
                "sample_id": sample["sample_id"],
                "raw_label": sample["raw_label"],
                "label_id": int(sample["label_id"]),
                "final_size": final_size,
                "observed_size": observed,
                "active_size": active,
                "inactive_size": inactive,
                "observed_ratio": observed / max(final_size, 1),
                "max_delay_minutes": max_delay,
                "max_observed_delay_minutes": max_observed_delay,
            }
        )
    return cases


def cases_to_arrays(cases: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "final_size": np.asarray([case["final_size"] for case in cases], dtype=np.float64),
        "observed_size": np.asarray([case["observed_size"] for case in cases], dtype=np.float64),
        "active_size": np.asarray([case["active_size"] for case in cases], dtype=np.float64),
        "inactive_size": np.asarray([case["inactive_size"] for case in cases], dtype=np.float64),
    }


def simulate_sir(
    observed: np.ndarray,
    active: np.ndarray,
    capacity_multiplier: float,
    beta: float,
    gamma: float,
    remaining_hours: float,
    step_hours: float,
) -> np.ndarray:
    capacity = np.maximum(np.ceil(observed * capacity_multiplier), observed)
    infected = np.minimum(active, observed)
    removed = np.maximum(observed - infected, 0.0)
    susceptible = np.maximum(capacity - observed, 0.0)

    for _ in range(max(int(math.ceil(remaining_hours / step_hours)), 0)):
        denom = np.maximum(capacity, 1.0)
        new_infections = np.minimum(beta * susceptible * infected / denom * step_hours, susceptible)
        new_recoveries = np.minimum(gamma * infected * step_hours, infected + new_infections)
        susceptible -= new_infections
        infected += new_infections - new_recoveries
        removed += new_recoveries
    return np.clip(infected + removed, observed, capacity)


def simulate_seir(
    observed: np.ndarray,
    active: np.ndarray,
    capacity_multiplier: float,
    beta: float,
    gamma: float,
    sigma: float,
    remaining_hours: float,
    step_hours: float,
) -> np.ndarray:
    capacity = np.maximum(np.ceil(observed * capacity_multiplier), observed)
    infectious = np.minimum(active, observed)
    removed = np.maximum(observed - infectious, 0.0)
    exposed = np.zeros_like(observed, dtype=np.float64)
    susceptible = np.maximum(capacity - observed, 0.0)

    for _ in range(max(int(math.ceil(remaining_hours / step_hours)), 0)):
        denom = np.maximum(capacity, 1.0)
        new_exposed = np.minimum(beta * susceptible * infectious / denom * step_hours, susceptible)
        new_infectious = np.minimum(sigma * exposed * step_hours, exposed + new_exposed)
        new_removed = np.minimum(gamma * infectious * step_hours, infectious + new_infectious)
        susceptible -= new_exposed
        exposed += new_exposed - new_infectious
        infectious += new_infectious - new_removed
        removed += new_removed
    return np.clip(infectious + removed, observed, capacity)


def simulate_seiz(
    observed: np.ndarray,
    active: np.ndarray,
    inactive: np.ndarray,
    capacity_multiplier: float,
    beta: float,
    b: float,
    rho: float,
    epsilon: float,
    p: float,
    l: float,
    remaining_hours: float,
    step_hours: float,
) -> np.ndarray:
    capacity = np.maximum(np.ceil(observed * capacity_multiplier), observed)
    infected = np.minimum(active, observed)
    skeptic = np.minimum(np.maximum(inactive, 0.0), np.maximum(observed - infected, 0.0))
    exposed = np.zeros_like(observed, dtype=np.float64)
    susceptible = np.maximum(capacity - infected - skeptic - exposed, 0.0)

    p = float(np.clip(p, 0.0, 1.0))
    l = float(np.clip(l, 0.0, 1.0))
    for _ in range(max(int(math.ceil(remaining_hours / step_hours)), 0)):
        denom = np.maximum(capacity, 1.0)
        si = beta * susceptible * infected / denom * step_hours
        sz = b * susceptible * skeptic / denom * step_hours
        ei = rho * exposed * infected / denom * step_hours
        spontaneous = epsilon * exposed * step_hours

        si = np.minimum(si, susceptible)
        sz = np.minimum(sz, np.maximum(susceptible - si, 0.0))
        new_exposed = (1.0 - p) * si + (1.0 - l) * sz
        new_infected = p * si + ei + spontaneous
        new_skeptic = l * sz

        new_infected = np.minimum(new_infected, exposed + p * si + spontaneous)
        susceptible = np.maximum(susceptible - si - sz, 0.0)
        exposed = np.maximum(exposed + new_exposed - ei - spontaneous, 0.0)
        infected = np.maximum(infected + new_infected, 0.0)
        skeptic = np.maximum(skeptic + new_skeptic, 0.0)
        total = infected + skeptic + exposed
        overflow = np.maximum(total - capacity, 0.0)
        if np.any(overflow > 0):
            scale = np.divide(capacity, np.maximum(total, 1.0))
            infected *= np.minimum(scale, 1.0)
            skeptic *= np.minimum(scale, 1.0)
            exposed *= np.minimum(scale, 1.0)
    return np.clip(infected + skeptic, observed, capacity)


def predict_dynamics(
    model_type: str,
    arrays: dict[str, np.ndarray],
    params: dict[str, float],
    remaining_hours: float,
    step_hours: float,
) -> np.ndarray:
    if model_type == "early_count":
        return arrays["observed_size"].copy()
    if model_type == "sir":
        return simulate_sir(
            observed=arrays["observed_size"],
            active=arrays["active_size"],
            capacity_multiplier=params["capacity_multiplier"],
            beta=params["beta"],
            gamma=params["gamma"],
            remaining_hours=remaining_hours,
            step_hours=step_hours,
        )
    if model_type == "seir":
        return simulate_seir(
            observed=arrays["observed_size"],
            active=arrays["active_size"],
            capacity_multiplier=params["capacity_multiplier"],
            beta=params["beta"],
            gamma=params["gamma"],
            sigma=params["sigma"],
            remaining_hours=remaining_hours,
            step_hours=step_hours,
        )
    if model_type == "seiz":
        return simulate_seiz(
            observed=arrays["observed_size"],
            active=arrays["active_size"],
            inactive=arrays["inactive_size"],
            capacity_multiplier=params["capacity_multiplier"],
            beta=params["beta"],
            b=params["b"],
            rho=params["rho"],
            epsilon=params["epsilon"],
            p=params["p"],
            l=params["l"],
            remaining_hours=remaining_hours,
            step_hours=step_hours,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "num_samples": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "r2": None,
            "median_ae": None,
            "mean_final_size": None,
            "mean_observed_size": None,
            "mean_observed_ratio": None,
        }
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    denom = np.maximum(np.abs(y_true), 1.0)
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
    total_ss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_ss = float(np.sum(errors**2))
    r2 = None if total_ss <= 1e-12 else 1.0 - residual_ss / total_ss
    return {
        "num_samples": int(len(y_true)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(abs_errors / denom)),
        "smape": float(np.mean(2.0 * abs_errors / smape_denom)),
        "r2": None if r2 is None else float(r2),
        "median_ae": float(np.median(abs_errors)),
        "mean_final_size": float(np.mean(y_true)),
        "mean_observed_size": float(np.mean(observed)),
        "mean_observed_ratio": float(np.mean(observed / denom)),
    }


def score_for_tuning(metrics: dict[str, Any], primary_metric: str) -> float:
    value = metrics.get(primary_metric)
    if value is None:
        return float("inf")
    return float(value)


def tune_grid(
    model_type: str,
    val_arrays: dict[str, np.ndarray],
    remaining_hours: float,
    step_hours: float,
    param_grid: list[dict[str, float]],
    primary_metric: str,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for params in param_grid:
        pred = predict_dynamics(model_type, val_arrays, params, remaining_hours, step_hours)
        metrics = regression_metrics(val_arrays["final_size"], pred, val_arrays["observed_size"])
        score = score_for_tuning(metrics, primary_metric)
        if best is None or score < best["score"]:
            best = {"score": score, "params": params, "val_metrics": metrics}
    if best is None:
        raise RuntimeError(f"{model_type} grid search did not evaluate any candidate")
    return best


def product_grid(**kwargs: list[float]) -> list[dict[str, float]]:
    keys = list(kwargs)
    return [dict(zip(keys, values)) for values in itertools.product(*(kwargs[key] for key in keys))]


def sampled_seiz_grid(args: argparse.Namespace, seed: int) -> list[dict[str, float]]:
    grid = product_grid(
        capacity_multiplier=parse_float_list(args.capacity_multipliers),
        beta=parse_float_list(args.beta_grid),
        b=parse_float_list(args.b_grid),
        rho=parse_float_list(args.rho_grid),
        epsilon=parse_float_list(args.epsilon_grid),
        p=parse_float_list(args.p_grid),
        l=parse_float_list(args.l_grid),
    )
    required = [
        {"capacity_multiplier": 1.5, "beta": 0.2, "b": 0.05, "rho": 0.2, "epsilon": 0.2, "p": 0.5, "l": 0.5},
        {"capacity_multiplier": 3.0, "beta": 0.4, "b": 0.1, "rho": 0.2, "epsilon": 0.2, "p": 0.75, "l": 0.25},
        {"capacity_multiplier": 5.0, "beta": 0.8, "b": 0.2, "rho": 0.5, "epsilon": 0.2, "p": 0.75, "l": 0.25},
    ]
    unique = {tuple(sorted(item.items())): item for item in grid}
    for item in required:
        unique.setdefault(tuple(sorted(item.items())), item)
    grid = list(unique.values())
    if len(grid) <= args.max_seiz_candidates:
        return grid
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(grid), size=args.max_seiz_candidates, replace=False)
    sampled = [grid[int(idx)] for idx in indices]
    for item in required:
        if item not in sampled:
            sampled.append(item)
    return sampled


def split_summary(dataset: RumorDataset) -> dict[str, Any] | None:
    return dataset.split_summary()


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
        "final_size",
        "observed_size",
        "active_size",
        "inactive_size",
        "observed_ratio",
        "max_delay_minutes",
        "pred_final_size",
        "abs_error",
        "squared_error",
        "absolute_percentage_error",
        "capacity_multiplier",
        "beta",
        "gamma",
        "sigma",
        "b",
        "rho",
        "epsilon",
        "p",
        "l",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def prediction_rows(
    split: str,
    model_name: str,
    observation_window: float,
    cases: list[dict[str, Any]],
    y_pred: np.ndarray,
    params: dict[str, float],
    split_strategy: str,
) -> list[dict[str, Any]]:
    rows = []
    for case, pred in zip(cases, y_pred):
        final_size = float(case["final_size"])
        abs_error = abs(float(pred) - final_size)
        rows.append(
            {
                "dataset": case["dataset"],
                "split_strategy": split_strategy,
                "split": split,
                "model": model_name,
                "observation_window_minutes": observation_window,
                "sample_id": case["sample_id"],
                "raw_label": case["raw_label"],
                "label_id": case["label_id"],
                "final_size": case["final_size"],
                "observed_size": case["observed_size"],
                "active_size": case["active_size"],
                "inactive_size": case["inactive_size"],
                "observed_ratio": case["observed_ratio"],
                "max_delay_minutes": case["max_delay_minutes"],
                "pred_final_size": float(pred),
                "abs_error": abs_error,
                "squared_error": abs_error**2,
                "absolute_percentage_error": abs_error / max(final_size, 1.0),
                **params,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = parse_models(args.models)
    observation_windows = parse_float_list(args.observation_windows)
    step_hours = args.step_minutes / 60.0
    forecast_horizon_hours = args.forecast_horizon_hours

    split_datasets = {split: load_split_dataset(args, split) for split in ("train", "val", "test")}
    all_sample_ids = {sample["sample_id"] for dataset in split_datasets.values() for sample in dataset}
    delays_by_sample = parse_event_delays(Path(args.data_root) / args.dataset / "events.csv", all_sample_ids, args.dataset)

    capacity_multipliers = parse_float_list(args.capacity_multipliers)
    beta_grid = parse_float_list(args.beta_grid)
    gamma_grid = parse_float_list(args.gamma_grid)
    sigma_grid = parse_float_list(args.sigma_grid)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    prediction_output_rows: list[dict[str, Any]] = []
    tuned_parameters: dict[str, Any] = {}

    for observation_window in observation_windows:
        remaining_hours = max(forecast_horizon_hours - observation_window / 60.0, 0.0)
        cases_by_split = {
            split: make_cases(dataset, delays_by_sample, observation_window, args.active_window_minutes)
            for split, dataset in split_datasets.items()
        }
        arrays_by_split = {split: cases_to_arrays(cases) for split, cases in cases_by_split.items()}

        model_specs: list[tuple[str, str, dict[str, float]]] = []
        if "early_count" in models:
            model_specs.append((f"early_count_w{int(observation_window)}m", "early_count", {}))
        if "sir" in models:
            sir_best = tune_grid(
                "sir",
                arrays_by_split["val"],
                remaining_hours,
                step_hours,
                product_grid(capacity_multiplier=capacity_multipliers, beta=beta_grid, gamma=gamma_grid),
                args.tune_metric,
            )
            name = f"sir_w{int(observation_window)}m"
            tuned_parameters[name] = sir_best
            model_specs.append((name, "sir", sir_best["params"]))
        if "seir" in models:
            seir_best = tune_grid(
                "seir",
                arrays_by_split["val"],
                remaining_hours,
                step_hours,
                product_grid(capacity_multiplier=capacity_multipliers, beta=beta_grid, gamma=gamma_grid, sigma=sigma_grid),
                args.tune_metric,
            )
            name = f"seir_w{int(observation_window)}m"
            tuned_parameters[name] = seir_best
            model_specs.append((name, "seir", seir_best["params"]))
        if "seiz" in models:
            seiz_best = tune_grid(
                "seiz",
                arrays_by_split["val"],
                remaining_hours,
                step_hours,
                sampled_seiz_grid(args, args.split_seed + int(observation_window)),
                args.tune_metric,
            )
            name = f"seiz_w{int(observation_window)}m"
            tuned_parameters[name] = seiz_best
            model_specs.append((name, "seiz", seiz_best["params"]))

        for model_name, model_type, params in model_specs:
            metrics_by_model[model_name] = {}
            for split in ("train", "val", "test"):
                arrays = arrays_by_split[split]
                predictions = predict_dynamics(model_type, arrays, params, remaining_hours, step_hours)
                metrics = regression_metrics(arrays["final_size"], predictions, arrays["observed_size"])
                metrics["observation_window_minutes"] = observation_window
                metrics["forecast_horizon_hours"] = forecast_horizon_hours
                metrics["active_window_minutes"] = args.active_window_minutes
                metrics["tune_metric"] = args.tune_metric
                metrics_by_model[model_name][split] = metrics
                prediction_output_rows.extend(
                    prediction_rows(
                        split=split,
                        model_name=model_name,
                        observation_window=observation_window,
                        cases=cases_by_split[split],
                        y_pred=predictions,
                        params=params,
                        split_strategy=args.split_strategy,
                    )
                )

    result = {
        "dataset": args.dataset,
        "task": "cascade_size_prediction",
        "label_task": args.label_task,
        "split_strategy": args.split_strategy,
        "split_seed": args.split_seed,
        "model_family": "paper_dynamics_seiz",
        "target": "final_cascade_size",
        "time_unit": "minutes",
        "forecast_horizon_hours": forecast_horizon_hours,
        "observation_windows_minutes": observation_windows,
        "active_window_minutes": args.active_window_minutes,
        "step_minutes": args.step_minutes,
        "models_requested": models,
        "tuned_parameters": tuned_parameters,
        "split_summaries": {split: split_summary(dataset) for split, dataset in split_datasets.items()},
        "models": metrics_by_model,
        "notes": [
            "SEIZ is an SIR-family information diffusion model with susceptible, exposed, infected/adopter, and skeptic/stifler compartments.",
            "Because the local datasets do not contain explicit non-spreading exposure logs, the skeptic/stifler compartment is initialized from observed-but-inactive events within the early window.",
            "Weibo is intentionally excluded from the default paper dynamics run because its current processed version lacks real timestamps.",
        ],
    }

    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset}_cascade_size_{args.split_strategy}_dynamics_seed{args.split_seed}"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    write_json(metrics_path, result)
    write_predictions(predictions_path, prediction_output_rows)
    result["outputs"] = {"metrics": str(metrics_path), "predictions": str(predictions_path)}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pheme", "twitter15", "twitter16", "weibo"])
    parser.add_argument("--label-task", default="rumor_binary", choices=["rumor_binary", "veracity", "raw"])
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--label-map", default="label_map.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--models", default="early_count,sir,seir,seiz")
    parser.add_argument("--observation-windows", default=DEFAULT_OBSERVATION_WINDOWS)
    parser.add_argument("--active-window-minutes", type=float, default=60.0)
    parser.add_argument("--forecast-horizon-hours", type=float, default=168.0)
    parser.add_argument("--step-minutes", type=float, default=15.0)
    parser.add_argument("--capacity-multipliers", default=DEFAULT_CAPACITY_MULTIPLIERS)
    parser.add_argument("--beta-grid", default=DEFAULT_BETA_GRID)
    parser.add_argument("--gamma-grid", default=DEFAULT_GAMMA_GRID)
    parser.add_argument("--sigma-grid", default=DEFAULT_SIGMA_GRID)
    parser.add_argument("--b-grid", default=DEFAULT_B_GRID)
    parser.add_argument("--rho-grid", default=DEFAULT_RHO_GRID)
    parser.add_argument("--epsilon-grid", default=DEFAULT_EPSILON_GRID)
    parser.add_argument("--p-grid", default=DEFAULT_P_GRID)
    parser.add_argument("--l-grid", default=DEFAULT_L_GRID)
    parser.add_argument("--max-seiz-candidates", type=int, default=450)
    parser.add_argument("--tune-metric", default="mape", choices=["mape", "rmse", "mae", "smape"])
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    result = run(parse_args())
    compact = {
        "dataset": result["dataset"],
        "task": result["task"],
        "split_strategy": result["split_strategy"],
        "outputs": result["outputs"],
        "test_metrics": {name: metrics["test"] for name, metrics in result["models"].items()},
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
