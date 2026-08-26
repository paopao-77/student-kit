import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"

FINAL_INDEX = SUMMARY / "weibo_raw_final_experiment_index.csv"
FINAL_AUDIT = SUMMARY / "weibo_raw_final_integrity_audit.csv"
FINAL_MD = DRAFTS / "weibo_raw_final_experiment_index.md"

REPORTING_ENTRYPOINTS = SUMMARY / "weibo_raw_reporting_entrypoints.csv"
ARTIFACT_MAP = SUMMARY / "weibo_raw_preferred_artifact_map.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def file_count(path: str, pattern: str) -> int:
    directory = ROOT / path
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def add_entry(
    rows: list[dict[str, Any]],
    module: str,
    artifact_type: str,
    role: str,
    path: str,
    status: str,
    notes: str = "",
    expected_files: int | str = "",
    observed_files: int | str = "",
) -> None:
    rows.append(
        {
            "module": module,
            "artifact_type": artifact_type,
            "role": role,
            "path": path,
            "status": status,
            "exists": exists(path),
            "expected_files": expected_files,
            "observed_files": observed_files,
            "notes": notes,
        }
    )


def build_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entrypoints = read_csv(REPORTING_ENTRYPOINTS)
    for entry in entrypoints:
        module = entry["module"]
        add_entry(
            rows,
            module,
            "preferred_summary",
            "use_for_reporting",
            entry["preferred_summary"],
            "preferred",
            f"{entry['primary_metric']}={entry['primary_value']}; {entry['secondary_metric']}={entry['secondary_value']}; {entry['caveat']}",
        )
        run_path = entry["preferred_runs"]
        expected = int(entry["n_seed_files"])
        pattern = "*_metrics.json"
        observed = file_count(run_path, pattern)
        add_entry(
            rows,
            module,
            "preferred_seed_runs",
            "use_for_reporting",
            run_path,
            "preferred",
            entry["setting"],
            expected,
            observed,
        )

    supporting = [
        ("V1", "metadata", "use_for_reporting", "data/processed/v1_inputs/weibo/obs_180events_metadata.json", "raw V1 artifact metadata"),
        ("C2/C3", "artifact_map", "use_for_reporting", "results/summary/weibo_raw_preferred_artifact_map.csv", "preferred vs historical artifact map"),
        ("C2/C3", "order_window_sensitivity", "supporting_analysis", "results/summary/weibo_raw_c2_c3_order_window_sensitivity.csv", "ow50/ow100/ow200 sensitivity"),
        ("C2/C3", "threshold_sensitivity", "supporting_analysis", "results/summary/weibo_raw_c2_c3_threshold_sensitivity.csv", "theta sensitivity"),
        ("C2", "threshold_audit", "supporting_analysis", "results/drafts/weibo_raw_c2_threshold_audit.md", "why threshold perturbations are invariant"),
        ("Raw-Weibo", "external_holdout", "supporting_analysis", "results/summary/weibo_raw_external_holdout_comparison.csv", "non-stratified fixed-seed holdout"),
        ("Raw-Weibo", "efficiency", "supporting_analysis", "results/summary/weibo_raw_efficiency_summary.csv", "runtime benchmark"),
        ("Raw-Weibo", "significance", "supporting_analysis", "results/summary/weibo_raw_e4_significance_tests.csv", "paired seed-level tests"),
        ("Raw-Weibo", "visual_diagnostics", "supporting_figure", "results/figures/fig_weibo_raw_e9_diagnostics.png", "E9 diagnostic figure"),
        ("Raw-Weibo", "case_studies", "supporting_figure", "results/figures/fig_weibo_raw_e10_case_studies.png", "E10 case figure"),
        ("Raw-Weibo", "early_warning", "supporting_figure", "results/figures/fig_weibo_raw_e12_early_warning.png", "E12 early-warning figure"),
        ("Raw-Weibo", "reproducibility", "supporting_audit", "results/summary/weibo_raw_e14_reproducibility_manifest.json", "E14 manifest"),
    ]
    for module, artifact_type, role, path, notes in supporting:
        add_entry(rows, module, artifact_type, role, path, "supporting", notes)

    historical = [
        ("C2", "legacy_summary", "historical_only", "results/summary/c2_breakout_weibo_raw_summary.csv", "ow100 former raw-Weibo result; do not report as preferred"),
        ("C3", "legacy_summary", "historical_only", "results/summary/c3_control_weibo_raw_summary.csv", "ow100 former raw-Weibo result; do not report as preferred"),
        ("C2", "legacy_runs", "historical_only", "results/c2_breakout_weibo_raw", "ow100 former run directory"),
        ("C3", "legacy_runs", "historical_only", "results/c3_control_weibo_raw", "ow100 former run directory"),
        ("C2", "sensitivity_runs", "historical_only", "results/c2_breakout_weibo_raw_ow200", "order-window sensitivity only"),
        ("C3", "sensitivity_runs", "historical_only", "results/c3_control_weibo_raw_ow200", "order-window sensitivity only"),
    ]
    for module, artifact_type, role, path, notes in historical:
        add_entry(rows, module, artifact_type, role, path, "historical_only", notes)
    return rows


def audit_rows(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entrypoints = read_csv(REPORTING_ENTRYPOINTS)
    artifact_map = read_csv(ARTIFACT_MAP)
    c2_stats = read_json(ROOT / "data/processed/weibo/c2_foundation_stats.json")
    manifest = read_json(SUMMARY / "weibo_raw_e14_reproducibility_manifest.json")

    preferred_paths = {row["path"] for row in index_rows if row["role"] == "use_for_reporting"}
    historical_paths = {row["path"] for row in index_rows if row["role"] == "historical_only"}

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check(
        "all_index_paths_exist",
        all(bool(row["exists"]) for row in index_rows),
        f"{sum(bool(row['exists']) for row in index_rows)}/{len(index_rows)} paths exist",
    )
    check(
        "all_preferred_entrypoints_status_preferred",
        all(row["status"] == "preferred" for row in entrypoints),
        "weibo_raw_reporting_entrypoints.csv status column",
    )
    check(
        "c2_preferred_runs_are_ow50",
        any(row["module"] == "C2" and row["preferred_runs"].endswith("c2_breakout_weibo_raw_ow50") for row in entrypoints),
        "C2 preferred_runs should point to ow50",
    )
    check(
        "c3_preferred_runs_are_ow50",
        any(row["module"] == "C3" and row["preferred_runs"].endswith("c3_control_weibo_raw_ow50") for row in entrypoints),
        "C3 preferred_runs should point to ow50",
    )
    check(
        "c2_foundation_order_window_50",
        int(c2_stats.get("order_window_size", -1)) == 50,
        f"order_window_size={c2_stats.get('order_window_size')}",
    )
    legacy_marked = [
        row for row in artifact_map if row["path"] in {"results\\c2_breakout_weibo_raw", "results\\c3_control_weibo_raw"}
    ]
    check(
        "legacy_ow100_marked_historical_only",
        all(row["status"] == "historical_comparison_only" for row in legacy_marked),
        "; ".join(f"{row['path']}={row['status']}" for row in legacy_marked),
    )
    check(
        "legacy_paths_not_in_reporting_preferred",
        not any(path in preferred_paths for path in historical_paths),
        "historical_only paths are disjoint from use_for_reporting paths",
    )
    check(
        "no_raw_summary_used_as_preferred_for_c2_c3",
        not any(
            row["preferred_summary"].endswith("c2_breakout_weibo_raw_summary.csv")
            or row["preferred_summary"].endswith("c3_control_weibo_raw_summary.csv")
            for row in entrypoints
        ),
        "preferred summaries use *_preferred_summary.csv for C2/C3",
    )
    coverage = manifest.get("v1_artifact", {}).get("modality_coverage", {})
    check(
        "modality_coverage_nonzero",
        float(coverage.get("text", 0.0)) > 0.0 and float(coverage.get("user_profile", 0.0)) > 0.0,
        f"text={coverage.get('text')}; user_profile={coverage.get('user_profile')}",
    )
    check(
        "final_index_has_use_support_historical_roles",
        {"use_for_reporting", "supporting_analysis", "supporting_figure", "supporting_audit", "historical_only"}.issubset(
            {row["role"] for row in index_rows}
        ),
        "role coverage in final index",
    )
    return checks


def write_markdown(index_rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    entrypoints = read_csv(REPORTING_ENTRYPOINTS)
    lines = [
        "# Raw Weibo Final Experiment Index",
        "",
        "## Reporting Rule",
        "",
        "- Use only rows marked `use_for_reporting` for final raw-Weibo reporting.",
        "- `results/c2_breakout_weibo_raw` and `results/c3_control_weibo_raw` are historical ow100 outputs; preferred C2/C3 uses `*_ow50`.",
        "- Old BiGCN-adapter Weibo outputs are superseded by the raw artifact results listed here.",
        "",
        "## Preferred Entry Points",
        "",
        "| module | summary | runs | metric | value | status |",
        "|---|---|---|---|---:|---|",
    ]
    for row in entrypoints:
        lines.append(
            f"| {row['module']} | {row['preferred_summary']} | {row['preferred_runs']} | "
            f"{row['primary_metric']} | {row['primary_value']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity Audit",
            "",
            "| check | passed | detail |",
            "|---|---:|---|",
        ]
    )
    for row in checks:
        lines.append(f"| {row['check']} | {row['passed']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Final Index Files",
            "",
            "- results/summary/weibo_raw_final_experiment_index.csv",
            "- results/summary/weibo_raw_final_integrity_audit.csv",
            "- results/drafts/weibo_raw_final_experiment_index.md",
            "",
        ]
    )
    FINAL_MD.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    index_rows = build_index()
    checks = audit_rows(index_rows)
    write_csv(
        FINAL_INDEX,
        index_rows,
        ["module", "artifact_type", "role", "path", "status", "exists", "expected_files", "observed_files", "notes"],
    )
    write_csv(FINAL_AUDIT, checks, ["check", "passed", "detail"])
    write_markdown(index_rows, checks)
    print(
        json.dumps(
            {
                "index": str(FINAL_INDEX),
                "audit": str(FINAL_AUDIT),
                "note": str(FINAL_MD),
                "all_checks_passed": all(bool(row["passed"]) for row in checks),
                "num_index_rows": len(index_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
