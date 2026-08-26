import csv
import math
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path("results") / "summary"

INPUT_TABLES = [
    {
        "family": "SEIZ/CD-SEIZ dynamics",
        "role": "SIR-family diffusion dynamics",
        "path": SUMMARY_DIR / "paper_dynamics_baseline_best.csv",
    },
    {
        "family": "MIDPMS-adapted",
        "role": "Multi-scale diffusion prediction",
        "path": SUMMARY_DIR / "paper_midpms_adapted_best.csv",
    },
    {
        "family": "DSHCL-adapted",
        "role": "High-order interaction representation",
        "path": SUMMARY_DIR / "paper_dshcl_adapted_best.csv",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    lines = [
        "\\begin{tabular}{" + "l" * len(fieldnames) + "}",
        "\\toprule",
        " & ".join(fieldnames) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("_", "\\_") for field in fieldnames]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fnum(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return math.nan
    return output if math.isfinite(output) else math.nan


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def load_best_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in INPUT_TABLES:
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing baseline summary: {spec['path']}")
        for row in read_csv(spec["path"]):
            rows.append(
                {
                    "Dataset": row["Dataset"],
                    "Split": row["Split"],
                    "Baseline family": spec["family"],
                    "Role": spec["role"],
                    "Best method": row["Best method"],
                    "Window": row["Window"],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE": row["MAPE"],
                    "SMAPE": row["SMAPE"],
                    "R2": row["R2"],
                }
            )
    return sorted(rows, key=lambda item: (item["Dataset"], item["Split"], item["Baseline family"]))


def winner_rows(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in best_rows:
        grouped.setdefault((row["Dataset"], row["Split"]), []).append(row)

    output = []
    for (dataset, split), rows in sorted(grouped.items()):
        winner = min(rows, key=lambda row: fnum(row["MAPE"]))
        output.append(
            {
                "Dataset": dataset,
                "Split": split,
                "Winning baseline family": winner["Baseline family"],
                "Winning method": winner["Best method"],
                "Window": winner["Window"],
                "MAPE": winner["MAPE"],
                "MAE": winner["MAE"],
                "RMSE": winner["RMSE"],
                "R2": winner["R2"],
            }
        )
    return output


def family_mean_rows(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in best_rows:
        grouped.setdefault((row["Baseline family"], row["Split"]), []).append(row)

    output = []
    for (family, split), rows in sorted(grouped.items()):
        mape_values = [fnum(row["MAPE"]) for row in rows]
        mae_values = [fnum(row["MAE"]) for row in rows]
        rmse_values = [fnum(row["RMSE"]) for row in rows]
        r2_values = [fnum(row["R2"]) for row in rows]
        output.append(
            {
                "Baseline family": family,
                "Split": split,
                "Datasets": len(rows),
                "Mean MAE": fmt(sum(mae_values) / len(mae_values)),
                "Mean RMSE": fmt(sum(rmse_values) / len(rmse_values)),
                "Mean MAPE": fmt(sum(mape_values) / len(mape_values)),
                "Mean R2": fmt(sum(r2_values) / len(r2_values)),
            }
        )
    return output


def emit_table(stem: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(SUMMARY_DIR / f"{stem}.csv", rows, fields)
    write_markdown(SUMMARY_DIR / f"{stem}.md", rows, fields)
    write_latex(SUMMARY_DIR / f"{stem}.tex", rows, fields)


def main() -> None:
    best = load_best_rows()
    winners = winner_rows(best)
    family_means = family_mean_rows(best)

    best_fields = [
        "Dataset",
        "Split",
        "Baseline family",
        "Role",
        "Best method",
        "Window",
        "MAE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "R2",
    ]
    winner_fields = [
        "Dataset",
        "Split",
        "Winning baseline family",
        "Winning method",
        "Window",
        "MAPE",
        "MAE",
        "RMSE",
        "R2",
    ]
    family_fields = [
        "Baseline family",
        "Split",
        "Datasets",
        "Mean MAE",
        "Mean RMSE",
        "Mean MAPE",
        "Mean R2",
    ]

    emit_table("paper_main_baselines_best", best, best_fields)
    emit_table("paper_main_baselines_winners", winners, winner_fields)
    emit_table("paper_main_baselines_family_mean", family_means, family_fields)

    print("Generated paper main baseline summary tables:")
    print(SUMMARY_DIR / "paper_main_baselines_best.csv")
    print(SUMMARY_DIR / "paper_main_baselines_winners.csv")
    print(SUMMARY_DIR / "paper_main_baselines_family_mean.csv")


if __name__ == "__main__":
    main()
