import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_ORDER_WINDOW_SIZE = 50
ROOT = Path(".")
SUMMARY = ROOT / "results" / "summary"
DRAFTS = ROOT / "results" / "drafts"

FILES = {
    "foundation_stats": ROOT / "data/processed/weibo/c2_foundation_stats.json",
    "c2_preferred_summary": SUMMARY / "c2_breakout_weibo_raw_preferred_summary.csv",
    "c3_preferred_summary": SUMMARY / "c3_control_weibo_raw_preferred_summary.csv",
    "c2_ow50_dir": ROOT / "results/c2_breakout_weibo_raw_ow50",
    "c3_ow50_dir": ROOT / "results/c3_control_weibo_raw_ow50",
    "c2_ow100_dir": ROOT / "results/c2_breakout_weibo_raw",
    "c3_ow100_dir": ROOT / "results/c3_control_weibo_raw",
    "preferred_note": DRAFTS / "weibo_raw_c2_c3_preferred_setting.md",
    "threshold_audit": DRAFTS / "weibo_raw_c2_threshold_audit.md",
    "order_sensitivity": SUMMARY / "weibo_raw_c2_c3_order_window_sensitivity.csv",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_json(path: Path) -> int:
    return len(list(path.glob("*_metrics.json")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_artifact() -> list[dict[str, Any]]:
    stats = read_json(FILES["foundation_stats"])
    c2_rows = read_csv(FILES["c2_preferred_summary"])
    c3_rows = read_csv(FILES["c3_preferred_summary"])
    c2_main = next(row for row in c2_rows if row["model"] == "heterorumor_c2")
    c3_main = next(row for row in c3_rows if row["strategy"] == "heterorumor_c3_event_pulse")

    checks = [
        {
            "check": "foundation_order_window_size",
            "expected": PREFERRED_ORDER_WINDOW_SIZE,
            "actual": int(stats.get("order_window_size", 0)),
            "passed": int(stats.get("order_window_size", 0)) == PREFERRED_ORDER_WINDOW_SIZE,
        },
        {
            "check": "foundation_theta_cross",
            "expected": 0.2,
            "actual": float(stats.get("theta_cross", -1.0)),
            "passed": float(stats.get("theta_cross", -1.0)) == 0.2,
        },
        {
            "check": "foundation_theta_branch_ratio",
            "expected": 0.2,
            "actual": float(stats.get("theta_branch_ratio", -1.0)),
            "passed": float(stats.get("theta_branch_ratio", -1.0)) == 0.2,
        },
        {
            "check": "c2_ow50_seed_files",
            "expected": 5,
            "actual": count_json(FILES["c2_ow50_dir"]),
            "passed": count_json(FILES["c2_ow50_dir"]) == 5,
        },
        {
            "check": "c3_ow50_seed_files",
            "expected": 5,
            "actual": count_json(FILES["c3_ow50_dir"]),
            "passed": count_json(FILES["c3_ow50_dir"]) == 5,
        },
        {
            "check": "c2_preferred_summary_order_window",
            "expected": PREFERRED_ORDER_WINDOW_SIZE,
            "actual": int(c2_main["preferred_order_window_size"]),
            "passed": int(c2_main["preferred_order_window_size"]) == PREFERRED_ORDER_WINDOW_SIZE,
        },
        {
            "check": "c3_preferred_summary_order_window",
            "expected": PREFERRED_ORDER_WINDOW_SIZE,
            "actual": int(c3_main["preferred_order_window_size"]),
            "passed": int(c3_main["preferred_order_window_size"]) == PREFERRED_ORDER_WINDOW_SIZE,
        },
        {
            "check": "c2_preferred_n_seeds",
            "expected": 5,
            "actual": int(c2_main["n_seeds"]),
            "passed": int(c2_main["n_seeds"]) == 5,
        },
        {
            "check": "c3_preferred_n_seeds",
            "expected": 5,
            "actual": int(c3_main["n_seeds"]),
            "passed": int(c3_main["n_seeds"]) == 5,
        },
        {
            "check": "legacy_ow100_retained_for_comparison",
            "expected": "present",
            "actual": "present" if FILES["c2_ow100_dir"].exists() and FILES["c3_ow100_dir"].exists() else "missing",
            "passed": FILES["c2_ow100_dir"].exists() and FILES["c3_ow100_dir"].exists(),
        },
    ]
    return checks


def write_mapping() -> list[dict[str, Any]]:
    rows = [
        {
            "artifact_role": "preferred_c2_summary",
            "path": str(FILES["c2_preferred_summary"]),
            "status": "use_for_future_raw_weibo_c2_reporting",
            "notes": "Derived from results/c2_breakout_weibo_raw_ow50.",
        },
        {
            "artifact_role": "preferred_c3_summary",
            "path": str(FILES["c3_preferred_summary"]),
            "status": "use_for_future_raw_weibo_c3_reporting",
            "notes": "Derived from results/c3_control_weibo_raw_ow50.",
        },
        {
            "artifact_role": "preferred_c2_runs",
            "path": str(FILES["c2_ow50_dir"]),
            "status": "preferred_seed_outputs",
            "notes": "order_window_size=50.",
        },
        {
            "artifact_role": "preferred_c3_runs",
            "path": str(FILES["c3_ow50_dir"]),
            "status": "preferred_seed_outputs",
            "notes": "order_window_size=50.",
        },
        {
            "artifact_role": "legacy_c2_runs",
            "path": str(FILES["c2_ow100_dir"]),
            "status": "historical_comparison_only",
            "notes": "Former raw-Weibo main setting, order_window_size=100.",
        },
        {
            "artifact_role": "legacy_c3_runs",
            "path": str(FILES["c3_ow100_dir"]),
            "status": "historical_comparison_only",
            "notes": "Former raw-Weibo main setting, order_window_size=100.",
        },
        {
            "artifact_role": "preferred_decision_note",
            "path": str(FILES["preferred_note"]),
            "status": "read_before_reporting",
            "notes": "Explains why ow50 was promoted and keeps the star-edge proxy caveat.",
        },
    ]
    write_csv(SUMMARY / "weibo_raw_preferred_artifact_map.csv", rows)
    return rows


def write_note(checks: list[dict[str, Any]], mapping: list[dict[str, Any]]) -> None:
    status = "PASS" if all(row["passed"] for row in checks) else "FAIL"
    lines = [
        "# Raw Weibo Preferred Artifact Validation",
        "",
        f"Validation status: **{status}**.",
        "",
        "## Checks",
        "",
        "| check | expected | actual | passed |",
        "|---|---:|---:|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['expected']} | {row['actual']} | {row['passed']} |")
    lines.extend(
        [
            "",
            "## Reporting Entry Points",
            "",
            "| role | path | status |",
            "|---|---|---|",
        ]
    )
    for row in mapping:
        lines.append(f"| {row['artifact_role']} | `{row['path']}` | {row['status']} |")
    lines.extend(
        [
            "",
            "Future raw-Weibo C2/C3 reporting should use the preferred summaries. The old `ow100` directories are retained only for historical comparison.",
            "",
        ]
    )
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "weibo_raw_preferred_artifact_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    checks = check_artifact()
    mapping = write_mapping()
    write_csv(SUMMARY / "weibo_raw_preferred_artifact_validation.csv", checks)
    write_note(checks, mapping)
    print(
        json.dumps(
            {
                "passed": all(row["passed"] for row in checks),
                "validation": str(SUMMARY / "weibo_raw_preferred_artifact_validation.csv"),
                "mapping": str(SUMMARY / "weibo_raw_preferred_artifact_map.csv"),
                "note": str(DRAFTS / "weibo_raw_preferred_artifact_validation.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
