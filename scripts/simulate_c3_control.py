import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_C2_DIR = Path("results/c2_breakout")
DEFAULT_OUTPUT_DIR = Path("results/c3_control")


def finite_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite_int(value: Any, default: int = 0) -> int:
    return int(round(finite_float(value, default)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("sample_id", "")].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda item: finite_int(item.get("window_index")))
    return dict(grouped)


def cumulative_final_size(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 1.0
    return max(1.0, finite_float(rows[-1].get("cumulative_nodes"), 1.0))


def c2_prediction_path(dataset: str, split_strategy: str, seed: int, c2_dir: Path) -> Path:
    return c2_dir / f"{dataset}_breakout_{split_strategy}_seed{seed}_predictions.csv"


def c2_metrics_path(dataset: str, split_strategy: str, seed: int, c2_dir: Path) -> Path:
    return c2_dir / f"{dataset}_breakout_{split_strategy}_seed{seed}_metrics.json"


def load_risk_rows(
    dataset: str,
    model: str,
    split_strategy: str,
    seed: int,
    c2_dir: Path,
) -> dict[str, dict[str, Any]]:
    path = c2_prediction_path(dataset, split_strategy, seed, c2_dir)
    rows = read_csv(path)
    selected = {}
    for row in rows:
        if row.get("split") != "test" or row.get("model") != model:
            continue
        selected[row["sample_id"]] = {
            "risk": finite_float(row.get("score_label_1")),
            "label_id": finite_int(row.get("label_id")),
            "pred_label_id": finite_int(row.get("pred_label_id")),
            "first_warning_window": row.get("first_warning_window", ""),
            "first_warning_time": row.get("first_warning_time", ""),
            "lead_time_minutes": row.get("lead_time_minutes", ""),
        }
    return selected


def model_threshold(dataset: str, model: str, split_strategy: str, seed: int, c2_dir: Path) -> float:
    payload = read_json(c2_metrics_path(dataset, split_strategy, seed, c2_dir))
    return finite_float(payload.get("models", {}).get(model, {}).get("test", {}).get("threshold"), 0.5)


def affected_growth(
    rows: list[dict[str, str]],
    effective_window: int,
    strength: float,
    effect_multiplier: float,
) -> tuple[float, float]:
    controlled = 1.0
    suppressed = 0.0
    prev_cumulative = 1.0
    for row in rows:
        window = finite_int(row.get("window_index"))
        current = finite_float(row.get("cumulative_nodes"), prev_cumulative)
        observed_new = max(0.0, current - prev_cumulative)
        if window >= effective_window:
            reduction = min(max(strength * effect_multiplier, 0.0), 0.95)
            kept_new = observed_new * (1.0 - reduction)
            suppressed += observed_new - kept_new
        else:
            kept_new = observed_new
        controlled += kept_new
        prev_cumulative = current
    return max(1.0, controlled), max(0.0, suppressed)


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def ed_id_state_features(
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    threshold: float,
    args: argparse.Namespace,
    *,
    mode: str,
) -> dict[str, float | int]:
    warning_window = risk.get("first_warning_window", "")
    decision_window = (
        finite_int(warning_window, args.fixed_trigger_window)
        if warning_window != ""
        else args.fixed_trigger_window
    )
    observed = [row for row in rows if finite_int(row.get("window_index")) <= decision_window]
    if not observed and rows:
        observed = [rows[0]]
        decision_window = finite_int(rows[0].get("window_index"))

    latest = observed[-1] if observed else {}
    previous = observed[-2] if len(observed) >= 2 else {}
    latest_new = finite_float(latest.get("new_nodes"))
    previous_new = finite_float(previous.get("new_nodes"))
    previous_cumulative = finite_float(previous.get("cumulative_nodes"), 1.0)
    cumulative_nodes = finite_float(latest.get("cumulative_nodes"), 1.0)

    growth_signal = clip01(1.0 - math.exp(-latest_new / max(previous_cumulative, 1.0)))
    trend_signal = clip01(
        0.5 + 0.5 * math.tanh((latest_new - previous_new) / max(previous_new + 1.0, 1.0))
    )
    size_signal = clip01(1.0 - math.exp(-cumulative_nodes / args.edid_size_scale))
    community_signal = clip01(
        1.0
        - math.exp(
            -finite_float(latest.get("active_communities"), 1.0)
            / args.edid_community_scale
        )
    )
    cross_signal = clip01(finite_float(latest.get("cross_edge_ratio")))
    branch_signal = clip01(finite_float(latest.get("branch_community_ratio")))
    internal_coordination = clip01(
        args.edid_internal_community_weight * community_signal
        + args.edid_internal_cross_weight * cross_signal
        + args.edid_internal_branch_weight * branch_signal
    )

    risk_value = clip01(finite_float(risk.get("risk")))
    risk_surplus = clip01((risk_value - threshold) / max(1.0 - threshold, 1e-6))
    external_incentive = clip01(
        args.edid_external_risk_weight * risk_value
        + args.edid_external_surplus_weight * risk_surplus
    )
    temporal_pressure = clip01(
        args.edid_growth_weight * growth_signal
        + args.edid_trend_weight * trend_signal
        + args.edid_size_weight * size_signal
    )

    if mode == "internal_only":
        external_incentive = 0.0
        diffusion_pressure = clip01(
            args.edid_internal_pressure_growth_weight * temporal_pressure
            + args.edid_internal_pressure_coord_weight * internal_coordination
        )
    elif mode == "external_only":
        internal_coordination = 0.0
        diffusion_pressure = clip01(
            args.edid_external_pressure_risk_weight * risk_value
            + args.edid_external_pressure_growth_weight * temporal_pressure
        )
    else:
        diffusion_pressure = clip01(
            args.edid_pressure_risk_weight * risk_value
            + args.edid_pressure_growth_weight * temporal_pressure
            + args.edid_pressure_coord_weight * internal_coordination
        )

    return {
        "decision_window": decision_window,
        "risk_value": risk_value,
        "growth_signal": growth_signal,
        "trend_signal": trend_signal,
        "size_signal": size_signal,
        "internal_coordination": internal_coordination,
        "external_incentive": external_incentive,
        "diffusion_pressure": diffusion_pressure,
        "active_communities": max(
            1.0, finite_float(latest.get("active_communities"), 1.0)
        ),
    }


def ed_id_evolution_decision(
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    threshold: float,
    args: argparse.Namespace,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    signals = ed_id_state_features(rows, risk, threshold, args, mode=mode)
    coordination = finite_float(signals["internal_coordination"])
    incentive = finite_float(signals["external_incentive"])
    pressure = finite_float(signals["diffusion_pressure"])

    intervention_share = clip01(
        args.edid_initial_share
        + args.edid_initial_external_gain * incentive
        + args.edid_initial_internal_gain * coordination
    )
    payoff_advantage = 0.0
    spread_response = 0.0
    for _ in range(args.edid_evolution_steps):
        spread_response = sigmoid(
            args.edid_selection_intensity
            * (
                pressure
                + args.edid_spread_coord_gain * coordination
                - args.edid_deterrence_gain * intervention_share
                - args.edid_spread_midpoint
            )
        )
        intervention_benefit = (
            args.edid_benefit_gain
            * pressure
            * spread_response
            * (1.0 + coordination)
        )
        external_reward = args.edid_external_reward_gain * incentive
        cost_signal = args.edid_cost_base + args.edid_cost_share_gain * intervention_share * (
            1.0 + args.edid_cost_coord_gain * coordination
        )
        intervene_payoff = (
            intervention_benefit
            + external_reward
            - args.edid_cost_penalty * cost_signal
        )
        wait_payoff = -args.edid_wait_harm_gain * pressure * spread_response
        payoff_advantage = intervene_payoff - wait_payoff
        intervention_share = clip01(
            intervention_share
            + args.edid_evolution_rate
            * intervention_share
            * (1.0 - intervention_share)
            * payoff_advantage
        )

    triggered = intervention_share >= args.edid_trigger_threshold
    raw_strength = (
        args.edid_pulse_base
        + args.edid_pulse_share_gain * intervention_share
        + args.edid_pulse_pressure_gain * pressure * spread_response
    )
    strength = (
        min(args.max_pulse, max(args.min_pulse, raw_strength)) if triggered else 0.0
    )
    return {
        **signals,
        "mode": mode,
        "triggered": int(triggered),
        "intervention_share": intervention_share,
        "spread_response": spread_response,
        "payoff_advantage": payoff_advantage,
        "raw_strength": strength,
    }


def ed_id_cost(strength: float, decision: dict[str, Any], args: argparse.Namespace) -> float:
    if strength <= 0:
        return 0.0
    community_load = math.sqrt(max(1.0, finite_float(decision.get("active_communities"), 1.0)))
    return (
        strength
        * (1.0 + args.edid_community_cost_gain * community_load)
        * (1.0 + args.delay_cost * args.delay_windows)
    )


def ed_id_budget_scale(
    decisions: dict[str, dict[str, Any]],
    target_total_cost: float,
    args: argparse.Namespace,
) -> float:
    triggered = [decision for decision in decisions.values() if finite_int(decision.get("triggered")) == 1]
    if not triggered or target_total_cost <= 0:
        return 0.0

    def total_cost(scale: float) -> float:
        return sum(
            ed_id_cost(
                min(args.max_pulse, finite_float(decision.get("raw_strength")) * scale),
                decision,
                args,
            )
            for decision in triggered
        )

    if total_cost(1.0) <= target_total_cost:
        low, high = 1.0, 2.0
        while high < 128.0 and total_cost(high) < target_total_cost:
            high *= 2.0
    else:
        low, high = 0.0, 1.0

    for _ in range(60):
        middle = (low + high) / 2.0
        if total_cost(middle) < target_total_cost:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def strategy_ed_id(
    sample_id: str,
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    decision: dict[str, Any],
    args: argparse.Namespace,
    *,
    strategy_name: str,
    strength_scale: float = 1.0,
) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    triggered = finite_int(decision.get("triggered")) == 1 and strength_scale > 0
    if triggered:
        trigger_window = finite_int(decision.get("decision_window"), args.fixed_trigger_window)
        effective_window = trigger_window + args.delay_windows
        strength = min(
            args.max_pulse,
            max(0.0, finite_float(decision.get("raw_strength")) * strength_scale),
        )
        controlled, suppressed = affected_growth(
            rows, effective_window, strength, args.effect_multiplier
        )
        cost = ed_id_cost(strength, decision, args)
    else:
        trigger_window = ""
        effective_window = ""
        strength = 0.0
        controlled = baseline
        suppressed = 0.0
        cost = 0.0

    return {
        "sample_id": sample_id,
        "strategy": strategy_name,
        "triggered": int(triggered),
        "trigger_window": trigger_window,
        "effective_window": effective_window,
        "pulse_strength": strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
        "edid_mode": decision.get("mode", ""),
        "edid_internal_coordination": decision.get("internal_coordination", 0.0),
        "edid_external_incentive": decision.get("external_incentive", 0.0),
        "edid_diffusion_pressure": decision.get("diffusion_pressure", 0.0),
        "edid_intervention_share": decision.get("intervention_share", 0.0),
        "edid_spread_response": decision.get("spread_response", 0.0),
        "edid_payoff_advantage": decision.get("payoff_advantage", 0.0),
    }


def strategy_none(sample_id: str, rows: list[dict[str, str]], risk: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_size = cumulative_final_size(rows)
    return {
        "sample_id": sample_id,
        "strategy": "no_intervention",
        "triggered": 0,
        "trigger_window": "",
        "effective_window": "",
        "pulse_strength": 0.0,
        "baseline_size": final_size,
        "controlled_size": final_size,
        "suppressed_nodes": 0.0,
        "suppression_rate": 0.0,
        "cost": 0.0,
        "benefit_cost_ratio": 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
    }


def strategy_fixed(sample_id: str, rows: list[dict[str, str]], risk: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    trigger_window = args.fixed_trigger_window
    effective_window = trigger_window + args.delay_windows
    controlled, suppressed = affected_growth(rows, effective_window, args.fixed_strength, args.effect_multiplier)
    cost = args.fixed_strength * max(args.fixed_duration_windows, 1)
    return {
        "sample_id": sample_id,
        "strategy": "fixed_intervention",
        "triggered": 1,
        "trigger_window": trigger_window,
        "effective_window": effective_window,
        "pulse_strength": args.fixed_strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
    }


def strategy_influence_blocking(
    sample_id: str,
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    first = rows[0] if rows else {}
    early_communities = max(finite_float(first.get("active_communities"), 1.0), 1.0)
    strength = min(args.max_pulse, args.blocking_base_strength + args.blocking_community_gain * math.log1p(early_communities))
    trigger_window = args.fixed_trigger_window
    effective_window = trigger_window + args.delay_windows
    controlled, suppressed = affected_growth(rows, effective_window, strength, args.effect_multiplier * 0.88)
    cost = strength * math.sqrt(early_communities)
    return {
        "sample_id": sample_id,
        "strategy": "influence_blocking",
        "triggered": 1,
        "trigger_window": trigger_window,
        "effective_window": effective_window,
        "pulse_strength": strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
    }


def c3_pulse_strength(risk_value: float, threshold: float, args: argparse.Namespace, use_game: bool = True) -> float:
    if not use_game:
        return args.fixed_strength
    if risk_value <= threshold:
        return 0.0
    normalized = (risk_value - threshold) / max(1.0 - threshold, 1e-6)
    follower_evasion = args.evasion_base + args.evasion_gain * normalized
    leader_response = args.pulse_base + args.pulse_gain * normalized
    strength = leader_response * (1.0 + args.game_response_gain * follower_evasion)
    return min(args.max_pulse, max(args.min_pulse, strength))


def strategy_event_pulse(
    sample_id: str,
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    threshold: float,
    args: argparse.Namespace,
    *,
    use_game: bool,
) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    risk_value = finite_float(risk.get("risk"))
    triggered = risk_value >= threshold
    if triggered:
        trigger_window = finite_int(risk.get("first_warning_window"), args.fixed_trigger_window)
        if risk.get("first_warning_window", "") == "":
            trigger_window = args.fixed_trigger_window
        strength = c3_pulse_strength(risk_value, threshold, args, use_game=use_game)
        effective_window = trigger_window + args.delay_windows
        controlled, suppressed = affected_growth(rows, effective_window, strength, args.effect_multiplier)
        cost = strength * (1.0 + args.delay_cost * args.delay_windows)
    else:
        trigger_window = ""
        effective_window = ""
        strength = 0.0
        controlled = baseline
        suppressed = 0.0
        cost = 0.0

    return {
        "sample_id": sample_id,
        "strategy": "heterorumor_c3_event_pulse" if use_game else "heterorumor_c3_no_game",
        "triggered": int(triggered),
        "trigger_window": trigger_window,
        "effective_window": effective_window,
        "pulse_strength": strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk_value,
        "label_id": risk.get("label_id", 0),
    }


def strategy_no_event_trigger(
    sample_id: str,
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    trigger_window = args.periodic_trigger_window
    strength = args.fixed_strength
    effective_window = trigger_window + args.delay_windows
    controlled, suppressed = affected_growth(rows, effective_window, strength, args.effect_multiplier)
    cost = strength * max(args.periodic_duration_windows, 1)
    return {
        "sample_id": sample_id,
        "strategy": "heterorumor_c3_no_event_trigger",
        "triggered": 1,
        "trigger_window": trigger_window,
        "effective_window": effective_window,
        "pulse_strength": strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
    }


def strategy_random_intervention(
    sample_id: str,
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    args: argparse.Namespace,
    *,
    selected_ids: set[str],
    cost_per_trigger: float,
) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    triggered = sample_id in selected_ids and cost_per_trigger > 0
    if triggered:
        trigger_window = args.fixed_trigger_window
        effective_window = trigger_window + args.delay_windows
        strength = min(args.max_pulse, max(0.0, cost_per_trigger / (1.0 + args.delay_cost * args.delay_windows)))
        controlled, suppressed = affected_growth(rows, effective_window, strength, args.effect_multiplier)
        cost = cost_per_trigger
    else:
        trigger_window = ""
        effective_window = ""
        strength = 0.0
        controlled = baseline
        suppressed = 0.0
        cost = 0.0

    return {
        "sample_id": sample_id,
        "strategy": "random_same_budget",
        "triggered": int(triggered),
        "trigger_window": trigger_window,
        "effective_window": effective_window,
        "pulse_strength": strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
    }


def strategy_same_budget_fixed(
    sample_id: str,
    rows: list[dict[str, str]],
    risk: dict[str, Any],
    args: argparse.Namespace,
    *,
    mean_budget: float,
) -> dict[str, Any]:
    baseline = cumulative_final_size(rows)
    trigger_window = args.fixed_trigger_window
    effective_window = trigger_window + args.delay_windows
    strength = min(args.max_pulse, max(0.0, mean_budget / max(args.fixed_duration_windows, 1)))
    controlled, suppressed = affected_growth(rows, effective_window, strength, args.effect_multiplier)
    cost = strength * max(args.fixed_duration_windows, 1)
    return {
        "sample_id": sample_id,
        "strategy": "fixed_same_budget",
        "triggered": 1 if cost > 0 else 0,
        "trigger_window": trigger_window if cost > 0 else "",
        "effective_window": effective_window if cost > 0 else "",
        "pulse_strength": strength,
        "baseline_size": baseline,
        "controlled_size": controlled,
        "suppressed_nodes": suppressed,
        "suppression_rate": (baseline - controlled) / baseline if baseline else 0.0,
        "cost": cost,
        "benefit_cost_ratio": suppressed / cost if cost > 0 else 0.0,
        "risk": risk.get("risk", 0.0),
        "label_id": risk.get("label_id", 0),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    summary = {}
    for strategy, items in grouped.items():
        suppression = [finite_float(row["suppression_rate"]) for row in items]
        costs = [finite_float(row["cost"]) for row in items]
        ratios = [finite_float(row["benefit_cost_ratio"]) for row in items if finite_float(row["cost"]) > 0]
        summary[strategy] = {
            "num_samples": len(items),
            "trigger_rate": float(np.mean([finite_float(row["triggered"]) for row in items])) if items else 0.0,
            "mean_baseline_size": float(np.mean([finite_float(row["baseline_size"]) for row in items])) if items else 0.0,
            "mean_controlled_size": float(np.mean([finite_float(row["controlled_size"]) for row in items])) if items else 0.0,
            "mean_suppression_rate": float(np.mean(suppression)) if suppression else 0.0,
            "median_suppression_rate": float(np.median(suppression)) if suppression else 0.0,
            "mean_cost": float(np.mean(costs)) if costs else 0.0,
            "median_cost": float(np.median(costs)) if costs else 0.0,
            "mean_benefit_cost_ratio": float(np.mean(ratios)) if ratios else 0.0,
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.data_root) / args.dataset
    snapshots = group_rows(read_csv(dataset_dir / "dynamic_snapshots" / "snapshots.csv"))
    risks = load_risk_rows(args.dataset, args.c2_model, args.split_strategy, args.seed, Path(args.c2_dir))
    threshold = model_threshold(args.dataset, args.c2_model, args.split_strategy, args.seed, Path(args.c2_dir))

    full_event_rows: dict[str, dict[str, Any]] = {}
    for sample_id, risk in sorted(risks.items()):
        sample_snapshots = snapshots.get(sample_id, [])
        if not sample_snapshots:
            continue
        full_event_rows[sample_id] = strategy_event_pulse(sample_id, sample_snapshots, risk, threshold, args, use_game=True)

    event_rows = list(full_event_rows.values())
    total_event_cost = float(sum(finite_float(row.get("cost")) for row in event_rows))
    event_triggered = [row["sample_id"] for row in event_rows if finite_int(row.get("triggered")) == 1]
    trigger_count = len(event_triggered)
    mean_budget = total_event_cost / max(len(event_rows), 1)
    cost_per_random_trigger = total_event_cost / trigger_count if trigger_count else 0.0
    rng_seed = args.seed * 1009 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(args.dataset))
    rng = np.random.default_rng(rng_seed)
    candidate_ids = sorted(full_event_rows)
    if trigger_count and candidate_ids:
        selected_random_ids = set(rng.choice(candidate_ids, size=min(trigger_count, len(candidate_ids)), replace=False).tolist())
    else:
        selected_random_ids = set()

    ed_id_decisions: dict[str, dict[str, dict[str, Any]]] = {
        "full": {},
        "internal_only": {},
        "external_only": {},
    }
    for sample_id, risk in sorted(risks.items()):
        sample_snapshots = snapshots.get(sample_id, [])
        if not sample_snapshots:
            continue
        for mode in ed_id_decisions:
            ed_id_decisions[mode][sample_id] = ed_id_evolution_decision(
                sample_snapshots,
                risk,
                threshold,
                args,
                mode=mode,
            )
    ed_id_same_budget_scale = ed_id_budget_scale(
        ed_id_decisions["full"],
        total_event_cost,
        args,
    )

    rows: list[dict[str, Any]] = []
    for sample_id, risk in sorted(risks.items()):
        sample_snapshots = snapshots.get(sample_id, [])
        if not sample_snapshots:
            continue
        rows.append(strategy_none(sample_id, sample_snapshots, risk, args))
        rows.append(strategy_fixed(sample_id, sample_snapshots, risk, args))
        rows.append(strategy_influence_blocking(sample_id, sample_snapshots, risk, args))
        rows.append(strategy_event_pulse(sample_id, sample_snapshots, risk, threshold, args, use_game=True))
        rows.append(strategy_event_pulse(sample_id, sample_snapshots, risk, threshold, args, use_game=False))
        rows.append(strategy_no_event_trigger(sample_id, sample_snapshots, risk, args))
        rows.append(
            strategy_random_intervention(
                sample_id,
                sample_snapshots,
                risk,
                args,
                selected_ids=selected_random_ids,
                cost_per_trigger=cost_per_random_trigger,
            )
        )
        rows.append(strategy_same_budget_fixed(sample_id, sample_snapshots, risk, args, mean_budget=mean_budget))
        rows.append(
            strategy_ed_id(
                sample_id,
                sample_snapshots,
                risk,
                ed_id_decisions["full"][sample_id],
                args,
                strategy_name="ed_id_adapted",
            )
        )
        rows.append(
            strategy_ed_id(
                sample_id,
                sample_snapshots,
                risk,
                ed_id_decisions["full"][sample_id],
                args,
                strategy_name="ed_id_adapted_same_budget",
                strength_scale=ed_id_same_budget_scale,
            )
        )
        rows.append(
            strategy_ed_id(
                sample_id,
                sample_snapshots,
                risk,
                ed_id_decisions["internal_only"][sample_id],
                args,
                strategy_name="ed_id_internal_only",
            )
        )
        rows.append(
            strategy_ed_id(
                sample_id,
                sample_snapshots,
                risk,
                ed_id_decisions["external_only"][sample_id],
                args,
                strategy_name="ed_id_external_only",
            )
        )

    payload = {
        "dataset": args.dataset,
        "task": "closed_loop_control",
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "model_family": "heterorumor_c3",
        "c2_model": args.c2_model,
        "c2_threshold": threshold,
        "delay_windows": args.delay_windows,
        "effect_multiplier": args.effect_multiplier,
        "same_budget_reference": {
            "reference_strategy": "heterorumor_c3_event_pulse",
            "reference_total_cost": total_event_cost,
            "reference_mean_cost": mean_budget,
            "reference_trigger_count": trigger_count,
            "random_cost_per_trigger": cost_per_random_trigger,
        },
        "ed_id_adapted": {
            "description": (
                "Evolutionary-game adaptation using internal community coordination, "
                "external risk incentive, diffusion pressure, intervention cost, "
                "and replicator dynamics."
            ),
            "same_budget_scale": ed_id_same_budget_scale,
            "same_budget_reference_total_cost": total_event_cost,
        },
        "strategies": summarize(rows),
    }

    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset}_control_{args.split_strategy}_{args.c2_model}_seed{args.seed}"
    write_json(output_dir / f"{prefix}_metrics.json", payload)
    fields = [
        "sample_id",
        "strategy",
        "triggered",
        "trigger_window",
        "effective_window",
        "pulse_strength",
        "baseline_size",
        "controlled_size",
        "suppressed_nodes",
        "suppression_rate",
        "cost",
        "benefit_cost_ratio",
        "risk",
        "label_id",
        "edid_mode",
        "edid_internal_coordination",
        "edid_external_incentive",
        "edid_diffusion_pressure",
        "edid_intervention_share",
        "edid_spread_response",
        "edid_payoff_advantage",
    ]
    write_csv(output_dir / f"{prefix}_simulations.csv", rows, fields)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pheme", "twitter15", "twitter16", "weibo"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--c2-dir", default=str(DEFAULT_C2_DIR))
    parser.add_argument("--c2-model", default="heterorumor_c2")
    parser.add_argument("--split-strategy", default="stratified", choices=["stratified", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--effect-multiplier", type=float, default=0.85)
    parser.add_argument("--fixed-trigger-window", type=int, default=1)
    parser.add_argument("--fixed-strength", type=float, default=0.30)
    parser.add_argument("--fixed-duration-windows", type=int, default=3)
    parser.add_argument("--blocking-base-strength", type=float, default=0.18)
    parser.add_argument("--blocking-community-gain", type=float, default=0.035)
    parser.add_argument("--periodic-trigger-window", type=int, default=1)
    parser.add_argument("--periodic-duration-windows", type=int, default=4)
    parser.add_argument("--min-pulse", type=float, default=0.10)
    parser.add_argument("--max-pulse", type=float, default=0.65)
    parser.add_argument("--pulse-base", type=float, default=0.18)
    parser.add_argument("--pulse-gain", type=float, default=0.32)
    parser.add_argument("--evasion-base", type=float, default=0.12)
    parser.add_argument("--evasion-gain", type=float, default=0.35)
    parser.add_argument("--game-response-gain", type=float, default=0.45)
    parser.add_argument("--delay-cost", type=float, default=0.08)
    parser.add_argument("--edid-size-scale", type=float, default=25.0)
    parser.add_argument("--edid-community-scale", type=float, default=6.0)
    parser.add_argument("--edid-internal-community-weight", type=float, default=0.45)
    parser.add_argument("--edid-internal-cross-weight", type=float, default=0.35)
    parser.add_argument("--edid-internal-branch-weight", type=float, default=0.20)
    parser.add_argument("--edid-external-risk-weight", type=float, default=0.65)
    parser.add_argument("--edid-external-surplus-weight", type=float, default=0.35)
    parser.add_argument("--edid-growth-weight", type=float, default=0.50)
    parser.add_argument("--edid-trend-weight", type=float, default=0.30)
    parser.add_argument("--edid-size-weight", type=float, default=0.20)
    parser.add_argument("--edid-pressure-risk-weight", type=float, default=0.48)
    parser.add_argument("--edid-pressure-growth-weight", type=float, default=0.32)
    parser.add_argument("--edid-pressure-coord-weight", type=float, default=0.20)
    parser.add_argument("--edid-internal-pressure-growth-weight", type=float, default=0.55)
    parser.add_argument("--edid-internal-pressure-coord-weight", type=float, default=0.45)
    parser.add_argument("--edid-external-pressure-risk-weight", type=float, default=0.60)
    parser.add_argument("--edid-external-pressure-growth-weight", type=float, default=0.40)
    parser.add_argument("--edid-initial-share", type=float, default=0.08)
    parser.add_argument("--edid-initial-external-gain", type=float, default=0.42)
    parser.add_argument("--edid-initial-internal-gain", type=float, default=0.20)
    parser.add_argument("--edid-evolution-steps", type=int, default=8)
    parser.add_argument("--edid-evolution-rate", type=float, default=0.75)
    parser.add_argument("--edid-selection-intensity", type=float, default=4.0)
    parser.add_argument("--edid-spread-coord-gain", type=float, default=0.45)
    parser.add_argument("--edid-deterrence-gain", type=float, default=0.80)
    parser.add_argument("--edid-spread-midpoint", type=float, default=0.50)
    parser.add_argument("--edid-benefit-gain", type=float, default=0.90)
    parser.add_argument("--edid-external-reward-gain", type=float, default=0.48)
    parser.add_argument("--edid-cost-base", type=float, default=0.12)
    parser.add_argument("--edid-cost-share-gain", type=float, default=0.30)
    parser.add_argument("--edid-cost-coord-gain", type=float, default=0.35)
    parser.add_argument("--edid-cost-penalty", type=float, default=0.25)
    parser.add_argument("--edid-wait-harm-gain", type=float, default=0.60)
    parser.add_argument("--edid-trigger-threshold", type=float, default=0.62)
    parser.add_argument("--edid-pulse-base", type=float, default=0.08)
    parser.add_argument("--edid-pulse-share-gain", type=float, default=0.38)
    parser.add_argument("--edid-pulse-pressure-gain", type=float, default=0.22)
    parser.add_argument("--edid-community-cost-gain", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    payload = run(parse_args())
    print(json.dumps({"dataset": payload["dataset"], "strategies": payload["strategies"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
