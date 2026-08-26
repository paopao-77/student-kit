import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"

ARTIFACTS = {
    "v1_summary": SUMMARY / "v1_weibo_multiseed_summary.csv",
    "v2_summary": SUMMARY / "v2_c1_weibo_selected_multiseed_summary.csv",
    "c2_summary": SUMMARY / "c2_breakout_weibo_raw_preferred_summary.csv",
    "c3_summary": SUMMARY / "c3_control_weibo_raw_preferred_summary.csv",
    "v1_runs": ROOT / "results/heterorumor_v1_weibo_multiseed",
    "v2_runs": ROOT / "results/heterorumor_v2_c1_weibo_selected_multiseed",
    "c2_runs": ROOT / "results/c2_breakout_weibo_raw_ow50",
    "c3_runs": ROOT / "results/c3_control_weibo_raw_ow50",
    "v1_input_metadata": ROOT / "data/processed/v1_inputs/weibo/obs_180events_metadata.json",
    "c2_foundation_stats": ROOT / "data/processed/weibo/c2_foundation_stats.json",
    "preferred_artifact_validation": SUMMARY / "weibo_raw_preferred_artifact_validation.csv",
    "threshold_audit": DRAFTS / "weibo_raw_c2_threshold_audit.md",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_json(path: Path) -> int:
    return len(list(path.glob("*_metrics.json")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows[0]


def find_row(path: Path, key: str, value: str) -> dict[str, str]:
    for row in read_csv(path):
        if row.get(key) == value:
            return row
    raise ValueError(f"No row where {key}={value} in {path}")


def build_entrypoints() -> list[dict[str, Any]]:
    v1 = first_row(ARTIFACTS["v1_summary"])
    v2 = first_row(ARTIFACTS["v2_summary"])
    c2 = find_row(ARTIFACTS["c2_summary"], "model", "heterorumor_c2")
    c3 = find_row(ARTIFACTS["c3_summary"], "strategy", "heterorumor_c3_event_pulse")
    metadata = read_json(ARTIFACTS["v1_input_metadata"])
    c2_stats = read_json(ARTIFACTS["c2_foundation_stats"])
    coverage = metadata.get("modality_coverage", {})

    return [
        {
            "module": "V1",
            "task": "raw_weibo_cascade_size",
            "preferred_summary": str(ARTIFACTS["v1_summary"]),
            "preferred_runs": str(ARTIFACTS["v1_runs"]),
            "n_seed_files": count_json(ARTIFACTS["v1_runs"]),
            "seeds": v1.get("seeds", ""),
            "primary_metric": "MAPE",
            "primary_value": v1.get("mape_mean", ""),
            "secondary_metric": "MAE",
            "secondary_value": v1.get("mae_mean", ""),
            "setting": "obs_180events; raw dataset input",
            "status": "preferred",
            "caveat": f"text_coverage={coverage.get('text', '')}; user_profile_coverage={coverage.get('user_profile', '')}",
        },
        {
            "module": "V2/C1",
            "task": "raw_weibo_selected_vae_k4",
            "preferred_summary": str(ARTIFACTS["v2_summary"]),
            "preferred_runs": str(ARTIFACTS["v2_runs"]),
            "n_seed_files": count_json(ARTIFACTS["v2_runs"]),
            "seeds": v2.get("seeds", ""),
            "primary_metric": "MAPE",
            "primary_value": v2.get("mape_mean", ""),
            "secondary_metric": "active_latent_factors",
            "secondary_value": v2.get("active_latent_factors_mean", ""),
            "setting": "obs_180events; K=4 selected VAE",
            "status": "preferred",
            "caveat": "Supersedes old BiGCN-adapter Weibo C1/V2 runs.",
        },
        {
            "module": "C2",
            "task": "raw_weibo_breakout_prediction",
            "preferred_summary": str(ARTIFACTS["c2_summary"]),
            "preferred_runs": str(ARTIFACTS["c2_runs"]),
            "n_seed_files": count_json(ARTIFACTS["c2_runs"]),
            "seeds": c2.get("seeds", ""),
            "primary_metric": "AUC",
            "primary_value": c2.get("auc_mean", ""),
            "secondary_metric": "F1",
            "secondary_value": c2.get("f1_mean", ""),
            "setting": f"order_window_size={c2_stats.get('order_window_size')}; theta_cross={c2_stats.get('theta_cross')}; theta_branch_ratio={c2_stats.get('theta_branch_ratio')}",
            "status": "preferred",
            "caveat": "Raw Weibo star-edge proxy; threshold audit required when interpreting community features.",
        },
        {
            "module": "C3",
            "task": "raw_weibo_event_pulse_control",
            "preferred_summary": str(ARTIFACTS["c3_summary"]),
            "preferred_runs": str(ARTIFACTS["c3_runs"]),
            "n_seed_files": count_json(ARTIFACTS["c3_runs"]),
            "seeds": c3.get("seeds", ""),
            "primary_metric": "mean_suppression_rate",
            "primary_value": c3.get("mean_suppression_rate_mean", ""),
            "secondary_metric": "mean_cost",
            "secondary_value": c3.get("mean_cost_mean", ""),
            "setting": f"order_window_size={c2_stats.get('order_window_size')}; c2_model=heterorumor_c2",
            "status": "preferred",
            "caveat": "Use preferred C2 ow50 outputs; ow100 is historical comparison only.",
        },
    ]


def build_validation(entrypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entrypoints:
        summary_exists = Path(entry["preferred_summary"]).exists()
        runs_exist = Path(entry["preferred_runs"]).exists()
        seed_count_ok = int(entry["n_seed_files"]) == 5
        rows.append(
            {
                "module": entry["module"],
                "summary_exists": summary_exists,
                "runs_exist": runs_exist,
                "n_seed_files": entry["n_seed_files"],
                "seed_count_ok": seed_count_ok,
                "status": "PASS" if summary_exists and runs_exist and seed_count_ok else "FAIL",
            }
        )
    rows.append(
        {
            "module": "C2/C3 preferred artifact validation",
            "summary_exists": ARTIFACTS["preferred_artifact_validation"].exists(),
            "runs_exist": ARTIFACTS["threshold_audit"].exists(),
            "n_seed_files": "",
            "seed_count_ok": True,
            "status": "PASS"
            if ARTIFACTS["preferred_artifact_validation"].exists()
            and ARTIFACTS["threshold_audit"].exists()
            else "FAIL",
        }
    )
    return rows


def write_note(entrypoints: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    status = "PASS" if all(row["status"] == "PASS" for row in validation) else "FAIL"
    lines = [
        "# Raw Weibo Reporting Entry Points",
        "",
        f"Validation status: **{status}**.",
        "",
        "Use this file as the routing table for the new raw Weibo dataset. The C2/C3 preferred setting is `order_window_size=50`; the older `ow100` outputs are historical comparisons only.",
        "",
        "## Preferred Summaries",
        "",
        "| module | task | preferred summary | primary metric | value | setting |",
        "|---|---|---|---|---:|---|",
    ]
    for row in entrypoints:
        lines.append(
            f"| {row['module']} | {row['task']} | `{row['preferred_summary']}` | "
            f"{row['primary_metric']} | {row['primary_value']} | {row['setting']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for row in entrypoints:
        lines.append(f"- {row['module']}: {row['caveat']}")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| module | n_seed_files | status |",
            "|---|---:|---|",
        ]
    )
    for row in validation:
        lines.append(f"| {row['module']} | {row['n_seed_files']} | {row['status']} |")
    lines.append("")
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_reporting_entrypoints.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    entrypoints = build_entrypoints()
    validation = build_validation(entrypoints)
    write_csv(SUMMARY / "weibo_raw_reporting_entrypoints.csv", entrypoints)
    write_csv(SUMMARY / "weibo_raw_reporting_entrypoints_validation.csv", validation)
    write_note(entrypoints, validation)
    print(
        json.dumps(
            {
                "passed": all(row["status"] == "PASS" for row in validation),
                "entrypoints": str(SUMMARY / "weibo_raw_reporting_entrypoints.csv"),
                "validation": str(SUMMARY / "weibo_raw_reporting_entrypoints_validation.csv"),
                "note": str(DRAFTS / "weibo_raw_reporting_entrypoints.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
