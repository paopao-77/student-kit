import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"
OUTPUT_JSON = SUMMARY / "weibo_raw_e14_reproducibility_manifest.json"
OUTPUT_FILES = SUMMARY / "weibo_raw_e14_reproducibility_files.csv"
OUTPUT_CHECKLIST = SUMMARY / "weibo_raw_e14_reproducibility_checklist.csv"
OUTPUT_NOTE = DRAFTS / "weibo_raw_e14_reproducibility_audit.md"

REQUIRED_FILES = [
    "data/processed/weibo/stats.json",
    "data/processed/v1_inputs/weibo/obs_180events_metadata.json",
    "data/processed/weibo/c2_foundation_stats.json",
    "data/processed/splits/weibo_rumor_binary_stratified_seed42_split.json",
    "data/processed/splits/weibo_rumor_binary_seed42_split.json",
    "results/summary/weibo_raw_reporting_entrypoints.csv",
    "results/summary/v1_weibo_multiseed_summary.csv",
    "results/summary/v2_c1_weibo_selected_multiseed_summary.csv",
    "results/summary/c2_breakout_weibo_raw_preferred_summary.csv",
    "results/summary/c3_control_weibo_raw_preferred_summary.csv",
    "results/summary/weibo_raw_external_holdout_comparison.csv",
    "results/summary/weibo_raw_efficiency_summary.csv",
    "results/summary/weibo_raw_e4_significance_tests.csv",
    "results/summary/weibo_raw_e12_early_warning_summary.csv",
    "results/case_studies/weibo_raw_e10_cases.csv",
    "results/figures/fig_weibo_raw_e9_diagnostics.png",
    "results/figures/fig_weibo_raw_e10_case_studies.png",
    "results/figures/fig_weibo_raw_e12_early_warning.png",
]

CORE_SCRIPTS = [
    "scripts/prepare_rumor_datasets.py",
    "scripts/build_v1_inputs.py",
    "scripts/train_heterorumor_v1.py",
    "scripts/build_c2_foundation.py",
    "scripts/train_c2_breakout.py",
    "scripts/simulate_c3_control.py",
    "scripts/promote_weibo_raw_c2_c3_preferred.py",
    "scripts/validate_weibo_raw_preferred_artifact.py",
    "scripts/build_weibo_raw_reporting_entrypoints.py",
    "scripts/benchmark_weibo_raw_efficiency.py",
    "scripts/summarize_weibo_raw_external_holdout.py",
    "scripts/plot_weibo_raw_experiment_diagnostics.py",
    "scripts/build_weibo_raw_e10_case_studies.py",
    "scripts/build_weibo_raw_e12_early_warning.py",
    "scripts/build_weibo_raw_e4_significance.py",
    "scripts/build_weibo_raw_e14_reproducibility.py",
    "scripts/workflow_status.py",
]


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for raw_path in paths:
        path = ROOT / raw_path
        exists = path.exists()
        rows.append(
            {
                "path": raw_path,
                "exists": exists,
                "bytes": path.stat().st_size if exists and path.is_file() else "",
                "sha256": sha256(path) if exists and path.is_file() else "",
            }
        )
    return rows


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for package in ["torch", "numpy", "sklearn", "scipy", "matplotlib"]:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "available"))
        except Exception:
            versions[package] = "not_available"
    return versions


def git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return "not_available"
    if completed.returncode != 0:
        return "not_a_git_repository"
    return completed.stdout.strip()


def split_summary(path: str) -> dict[str, Any]:
    payload = read_json(path)
    summary = payload.get("summary", {})
    return {
        "path": path,
        "strategy": payload.get("strategy"),
        "seed": payload.get("seed"),
        "ratios": payload.get("ratios"),
        "num_available_labeled_samples": payload.get("num_available_labeled_samples"),
        "train": summary.get("train", {}).get("num_samples"),
        "val": summary.get("val", {}).get("num_samples"),
        "test": summary.get("test", {}).get("num_samples"),
        "notes": payload.get("notes", []),
    }


def first_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def build_manifest() -> dict[str, Any]:
    stats = read_json("data/processed/weibo/stats.json")
    v1_meta = read_json("data/processed/v1_inputs/weibo/obs_180events_metadata.json")
    c2_stats = read_json("data/processed/weibo/c2_foundation_stats.json")
    entrypoints = read_csv("results/summary/weibo_raw_reporting_entrypoints.csv")
    external = read_csv("results/summary/weibo_raw_external_holdout_comparison.csv")
    e12_summary = read_csv("results/summary/weibo_raw_e12_early_warning_summary.csv")
    significance = read_csv("results/summary/weibo_raw_e4_significance_tests.csv")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "environment": package_versions(),
        "dataset": {
            "name": "weibo",
            "source": "数据集\\微博",
            "num_samples": stats.get("num_samples"),
            "num_users": stats.get("num_users"),
            "num_users_with_profile": stats.get("num_users_with_profile"),
            "num_edges_raw": stats.get("num_edges"),
            "avg_cascade_nodes": stats.get("avg_cascade_nodes"),
            "samples_with_source_text": stats.get("samples_with_source_text"),
            "notes": stats.get("notes", []),
        },
        "v1_artifact": {
            "observation": v1_meta.get("observation"),
            "time_mode": v1_meta.get("time_mode"),
            "num_samples": v1_meta.get("num_samples"),
            "num_nodes": v1_meta.get("num_nodes"),
            "num_edges": v1_meta.get("num_edges"),
            "modality_coverage": v1_meta.get("modality_coverage"),
            "leakage_policy": v1_meta.get("leakage_policy"),
        },
        "splits": {
            "preferred": split_summary("data/processed/splits/weibo_rumor_binary_stratified_seed42_split.json"),
            "external_holdout": split_summary("data/processed/splits/weibo_rumor_binary_seed42_split.json"),
        },
        "random_seeds": [7, 21, 42, 84, 2024],
        "c2_foundation": c2_stats,
        "preferred_results": entrypoints,
        "external_holdout": external,
        "early_warning": e12_summary,
        "significance_highlights": [
            row
            for row in significance
            if row.get("family") in {"V1_vs_V2C1", "C3_control"}
            or (row.get("family") == "C2_breakout" and row.get("metric") == "auc")
        ],
        "caveats": [
            "Raw Weibo diffusion data is represented as source-to-retweet star edges because retweet parent IDs are not available.",
            "Raw Weibo C2/C3 time is event-order based; lead times are event-order units, not wall-clock minutes.",
            "The raw artifact has cascade-size targets but not true rumor/non-rumor labels; split task labels are compatibility placeholders.",
            "With five seeds, exact sign-flip p-values are coarse; minimum two-sided p for 5/5 direction is 0.0625.",
        ],
    }
    return manifest


def checklist_rows(manifest: dict[str, Any], files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage = manifest["v1_artifact"]["modality_coverage"]
    return [
        {
            "item": "raw_weibo_source_documented",
            "status": bool(manifest["dataset"]["source"]),
            "detail": manifest["dataset"]["source"],
        },
        {
            "item": "text_coverage_nonzero",
            "status": float(coverage.get("text", 0.0)) > 0.0,
            "detail": coverage.get("text"),
        },
        {
            "item": "user_profile_coverage_nonzero",
            "status": float(coverage.get("user_profile", 0.0)) > 0.0,
            "detail": coverage.get("user_profile"),
        },
        {
            "item": "preferred_split_documented",
            "status": manifest["splits"]["preferred"]["train"] == 3500,
            "detail": manifest["splits"]["preferred"],
        },
        {
            "item": "external_holdout_split_documented",
            "status": manifest["splits"]["external_holdout"]["train"] == 3500,
            "detail": manifest["splits"]["external_holdout"],
        },
        {
            "item": "preferred_c2_order_window_50",
            "status": int(manifest["c2_foundation"].get("order_window_size", -1)) == 50,
            "detail": manifest["c2_foundation"].get("order_window_size"),
        },
        {
            "item": "all_required_files_exist",
            "status": all(bool(row["exists"]) for row in files),
            "detail": sum(bool(row["exists"]) for row in files),
        },
        {
            "item": "environment_recorded",
            "status": bool(manifest["environment"].get("python")),
            "detail": manifest["environment"].get("python"),
        },
    ]


def write_note(manifest: dict[str, Any], checklist: list[dict[str, Any]]) -> None:
    def md_cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    coverage = manifest["v1_artifact"]["modality_coverage"]
    entry = {row["module"]: row for row in manifest["preferred_results"]}
    lines = [
        "# Raw Weibo E14 Reproducibility Audit",
        "",
        "## Dataset And Inputs",
        "",
        f"- Source: `{manifest['dataset']['source']}`.",
        f"- Samples: {manifest['dataset']['num_samples']}; raw users: {manifest['dataset']['num_users']}; users with profile: {manifest['dataset']['num_users_with_profile']}.",
        f"- V1 artifact: observation `{manifest['v1_artifact']['observation']}`, time mode `{manifest['v1_artifact']['time_mode']}`.",
        f"- Modality coverage: text {coverage.get('text')}, topology {coverage.get('topology')}, temporal {coverage.get('temporal')}, user profile {coverage.get('user_profile')}.",
        "",
        "## Preferred Results",
        "",
        f"- V1 MAPE: {entry['V1']['primary_value']} from `{entry['V1']['preferred_summary']}`.",
        f"- V2/C1 MAPE: {entry['V2/C1']['primary_value']} from `{entry['V2/C1']['preferred_summary']}`.",
        f"- C2 AUC: {entry['C2']['primary_value']} from `{entry['C2']['preferred_summary']}`.",
        f"- C3 suppression: {entry['C3']['primary_value']} from `{entry['C3']['preferred_summary']}`.",
        "",
        "## Checklist",
        "",
        "| item | status | detail |",
        "|---|---:|---|",
    ]
    for row in checklist:
        lines.append(f"| {row['item']} | {row['status']} | {md_cell(row['detail'])} |")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in manifest["caveats"])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- results/summary/weibo_raw_e14_reproducibility_manifest.json",
            "- results/summary/weibo_raw_e14_reproducibility_files.csv",
            "- results/summary/weibo_raw_e14_reproducibility_checklist.csv",
            "- results/drafts/weibo_raw_e14_reproducibility_audit.md",
            "",
        ]
    )
    OUTPUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    files = file_rows(REQUIRED_FILES + CORE_SCRIPTS)
    manifest = build_manifest()
    checklist = checklist_rows(manifest, files)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUTPUT_FILES, files, ["path", "exists", "bytes", "sha256"])
    write_csv(OUTPUT_CHECKLIST, checklist, ["item", "status", "detail"])
    write_note(manifest, checklist)
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT_JSON),
                "files": str(OUTPUT_FILES),
                "checklist": str(OUTPUT_CHECKLIST),
                "note": str(OUTPUT_NOTE),
                "all_required_files_exist": all(bool(row["exists"]) for row in files),
                "checklist_pass": all(bool(row["status"]) for row in checklist),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
