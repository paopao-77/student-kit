import csv
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"
SEEDS = [7, 21, 42, 84, 2024]
BOOTSTRAP_ROUNDS = 10000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 8) -> float | str:
    return round(value, digits) if math.isfinite(value) else ""


def model_metric(payload: dict[str, Any], metric: str) -> float:
    model_type = str(payload["model_type"])
    return float(payload["models"][model_type]["test"][metric])


def load_v_metric(directory: Path, model_prefix: str, metric: str) -> dict[int, float]:
    values = {}
    for path in sorted(directory.glob("*_metrics.json")):
        payload = read_json(path)
        model_type = str(payload.get("model_type", ""))
        if not model_type.startswith(model_prefix):
            continue
        seed = int(payload["seed"])
        values[seed] = model_metric(payload, metric)
    missing = [seed for seed in SEEDS if seed not in values]
    if missing:
        raise FileNotFoundError(f"Missing {model_prefix} seeds under {directory}: {missing}")
    return values


def load_group_metric(path: Path, group_key: str, group_value: str, metric: str) -> dict[int, float]:
    rows = read_csv(path)
    values = {}
    for row in rows:
        if row[group_key] == group_value:
            values[int(row["seed"])] = float(row[metric])
    missing = [seed for seed in SEEDS if seed not in values]
    if missing:
        raise FileNotFoundError(f"Missing {group_value} seeds in {path}: {missing}")
    return values


def exact_sign_flip_p(improvements: list[float]) -> float:
    observed = abs(mean(improvements))
    if observed <= 1e-15:
        return 1.0
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(improvements)):
        total += 1
        value = abs(mean([sign * diff for sign, diff in zip(signs, improvements)]))
        if value >= observed - 1e-15:
            count += 1
    return count / total


def bootstrap_ci(values: list[float], rounds: int = BOOTSTRAP_ROUNDS) -> tuple[float, float]:
    rng = random.Random(20260701 + len(values))
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    lower = means[int(0.025 * rounds)]
    upper = means[min(rounds - 1, int(0.975 * rounds))]
    return lower, upper


def paired_comparison(
    family: str,
    metric: str,
    better: str,
    reference_name: str,
    candidate_name: str,
    reference: dict[int, float],
    candidate: dict[int, float],
) -> dict[str, Any]:
    if better not in {"lower", "higher"}:
        raise ValueError(f"Unknown better direction: {better}")
    paired = []
    for seed in SEEDS:
        ref = float(reference[seed])
        cand = float(candidate[seed])
        improvement = ref - cand if better == "lower" else cand - ref
        paired.append({"seed": seed, "reference": ref, "candidate": cand, "improvement": improvement})
    improvements = [row["improvement"] for row in paired]
    ci_low, ci_high = bootstrap_ci(improvements)
    sd = stdev(improvements) if len(improvements) > 1 else 0.0
    return {
        "family": family,
        "metric": metric,
        "better": better,
        "reference": reference_name,
        "candidate": candidate_name,
        "n_pairs": len(improvements),
        "seeds": " ".join(map(str, SEEDS)),
        "reference_mean": fmt(mean([row["reference"] for row in paired])),
        "candidate_mean": fmt(mean([row["candidate"] for row in paired])),
        "mean_improvement": fmt(mean(improvements)),
        "improvement_std": fmt(sd),
        "bootstrap_ci95_low": fmt(ci_low),
        "bootstrap_ci95_high": fmt(ci_high),
        "exact_sign_flip_p_two_sided": fmt(exact_sign_flip_p(improvements), 6),
        "positive_pairs": sum(value > 0 for value in improvements),
        "negative_pairs": sum(value < 0 for value in improvements),
        "zero_pairs": sum(abs(value) <= 1e-15 for value in improvements),
        "significant_p_lt_0_05": exact_sign_flip_p(improvements) < 0.05,
        "ci_excludes_zero": ci_low > 0 or ci_high < 0,
        "paired_improvements": ",".join(f"{value:.10f}" for value in improvements),
    }


def build_comparisons() -> list[dict[str, Any]]:
    rows = []
    v1_mape = load_v_metric(
        ROOT / "results" / "heterorumor_v1_weibo_multiseed",
        "heterorumor_v1_hurdle",
        "mape",
    )
    v2_mape = load_v_metric(
        ROOT / "results" / "heterorumor_v2_c1_weibo_selected_multiseed",
        "heterorumor_v2_c1_vae_k4_weibo_selected",
        "mape",
    )
    rows.append(
        paired_comparison(
            family="V1_vs_V2C1",
            metric="mape",
            better="lower",
            reference_name="heterorumor_v1_hurdle",
            candidate_name="heterorumor_v2_c1_vae_k4_weibo_selected",
            reference=v1_mape,
            candidate=v2_mape,
        )
    )

    c2_path = SUMMARY / "c2_breakout_weibo_raw_preferred_all_runs.csv"
    c2_full_auc = load_group_metric(c2_path, "model", "heterorumor_c2", "auc")
    c2_full_f1 = load_group_metric(c2_path, "model", "heterorumor_c2", "f1")
    for ablation in [
        "heterorumor_c2_no_lowfreq",
        "heterorumor_c2_no_cross",
        "heterorumor_c2_no_temporal_trend",
        "heterorumor_c2_dynamic_only",
        "heterorumor_c2_community_only",
        "dynamic_random_forest",
        "static_logistic",
    ]:
        rows.append(
            paired_comparison(
                family="C2_breakout",
                metric="auc",
                better="higher",
                reference_name=ablation,
                candidate_name="heterorumor_c2",
                reference=load_group_metric(c2_path, "model", ablation, "auc"),
                candidate=c2_full_auc,
            )
        )
        rows.append(
            paired_comparison(
                family="C2_breakout",
                metric="f1",
                better="higher",
                reference_name=ablation,
                candidate_name="heterorumor_c2",
                reference=load_group_metric(c2_path, "model", ablation, "f1"),
                candidate=c2_full_f1,
            )
        )

    c3_path = SUMMARY / "c3_control_weibo_raw_preferred_all_runs.csv"
    c3_full = load_group_metric(
        c3_path, "strategy", "heterorumor_c3_event_pulse", "mean_suppression_rate"
    )
    for strategy in [
        "random_same_budget",
        "fixed_same_budget",
        "heterorumor_c3_no_game",
        "heterorumor_c3_no_event_trigger",
        "ed_id_adapted_same_budget",
        "ed_id_adapted",
    ]:
        rows.append(
            paired_comparison(
                family="C3_control",
                metric="mean_suppression_rate",
                better="higher",
                reference_name=strategy,
                candidate_name="heterorumor_c3_event_pulse",
                reference=load_group_metric(c3_path, "strategy", strategy, "mean_suppression_rate"),
                candidate=c3_full,
            )
        )
    return rows


def write_note(rows: list[dict[str, Any]]) -> None:
    highlights = [
        row
        for row in rows
        if row["family"] in {"V1_vs_V2C1", "C3_control"}
        or (row["family"] == "C2_breakout" and row["metric"] == "auc")
    ]
    lines = [
        "# Raw Weibo E4 Significance Tests",
        "",
        "## Method",
        "",
        "- Unit of pairing: identical random seed across compared raw-Weibo runs.",
        "- Effect direction is `mean_improvement`: positive means the candidate is better than the reference.",
        "- p-value: exact two-sided sign-flip test over the five paired seed differences.",
        "- CI: deterministic bootstrap 95% CI over paired seed improvements.",
        "- With only five seeds, exact sign-flip p-values are coarse; CI and pair counts should be read together.",
        "",
        "## Highlighted Comparisons",
        "",
        "| family | metric | reference | candidate | mean improvement | CI95 | p | pairs +/-/0 |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in highlights:
        lines.append(
            f"| {row['family']} | {row['metric']} | {row['reference']} | {row['candidate']} | "
            f"{row['mean_improvement']} | [{row['bootstrap_ci95_low']}, {row['bootstrap_ci95_high']}] | "
            f"{row['exact_sign_flip_p_two_sided']} | "
            f"{row['positive_pairs']}/{row['negative_pairs']}/{row['zero_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- results/summary/weibo_raw_e4_significance_tests.csv",
            "- results/drafts/weibo_raw_e4_significance_tests.md",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_e4_significance_tests.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    rows = build_comparisons()
    write_csv(SUMMARY / "weibo_raw_e4_significance_tests.csv", rows)
    write_note(rows)
    print(
        json.dumps(
            {
                "comparisons": len(rows),
                "summary": str(SUMMARY / "weibo_raw_e4_significance_tests.csv"),
                "note": str(DRAFTS / "weibo_raw_e4_significance_tests.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
