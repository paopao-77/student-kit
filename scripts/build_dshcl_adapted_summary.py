import csv
import json
import math
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("results")
DSHCL_DIR = RESULTS_ROOT / "paper_baselines" / "dshcl_adapted"
SUMMARY_DIR = RESULTS_ROOT / "summary"

MODEL_NAMES = {
    "dshcl_diffusion": "DSHCL-diffusion",
    "dshcl_interaction": "DSHCL-interaction",
    "dshcl_adapted": "DSHCL-adapted",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def model_base_name(model_name: str) -> str:
    return model_name.split("_w", 1)[0]


def model_display_name(model_name: str) -> str:
    return MODEL_NAMES.get(model_base_name(model_name), model_base_name(model_name))


def model_window(model_name: str) -> int:
    raw = model_name.rsplit("_w", 1)[-1].replace("m", "")
    try:
        return int(float(raw))
    except ValueError:
        return -1


def load_long_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(DSHCL_DIR.glob("*_metrics.json")):
        payload = read_json(path)
        for model_name, split_payload in payload.get("models", {}).items():
            for split, metrics in split_payload.items():
                rows.append(
                    {
                        "dataset": payload.get("dataset", ""),
                        "split_strategy": payload.get("split_strategy", ""),
                        "split": split,
                        "model": model_name,
                        "method": model_display_name(model_name),
                        "feature_role": metrics.get("feature_role", ""),
                        "contrastive_mode": metrics.get("contrastive_mode", ""),
                        "contrastive_components": metrics.get("contrastive_components", ""),
                        "window_minutes": metrics.get("observation_window_minutes", model_window(model_name)),
                        "num_samples": metrics.get("num_samples"),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "mape": metrics.get("mape"),
                        "smape": metrics.get("smape"),
                        "r2": metrics.get("r2"),
                        "median_ae": metrics.get("median_ae"),
                        "source_file": str(path),
                    }
                )
    return rows


def paper_rows(long_rows: list[dict[str, Any]], split_strategy: str = "stratified") -> list[dict[str, Any]]:
    test_rows = [row for row in long_rows if row["split"] == "test" and row["split_strategy"] == split_strategy]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in test_rows:
        grouped.setdefault((row["dataset"], int(float(row["window_minutes"]))), []).append(row)
    output = []
    for (dataset, window), rows in sorted(grouped.items()):
        best_mape = min(float(row["mape"]) for row in rows if row.get("mape") not in (None, ""))
        for row in sorted(rows, key=lambda item: (item["method"], item["model"])):
            output.append(
                {
                    "Dataset": dataset,
                    "Window": f"{window}m",
                    "Method": row["method"],
                    "MAE": fmt(row["mae"]),
                    "RMSE": fmt(row["rmse"]),
                    "MAPE": fmt(row["mape"]),
                    "SMAPE": fmt(row["smape"]),
                    "R2": fmt(row["r2"]),
                    "Best MAPE": "yes" if abs(float(row["mape"]) - best_mape) < 1e-12 else "",
                }
            )
    return output


def best_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split_strategy in ["stratified", "temporal"]:
        test_rows = [row for row in long_rows if row["split"] == "test" and row["split_strategy"] == split_strategy]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in test_rows:
            grouped.setdefault(row["dataset"], []).append(row)
        for dataset, rows in sorted(grouped.items()):
            best = min(rows, key=lambda row: float(row["mape"]))
            output.append(
                {
                    "Dataset": dataset,
                    "Split": split_strategy,
                    "Best method": best["method"],
                    "Window": f"{int(float(best['window_minutes']))}m",
                    "MAE": fmt(best["mae"]),
                    "RMSE": fmt(best["rmse"]),
                    "MAPE": fmt(best["mape"]),
                    "SMAPE": fmt(best["smape"]),
                    "R2": fmt(best["r2"]),
                }
            )
    return output


def main() -> None:
    long_rows = load_long_rows()
    if not long_rows:
        raise FileNotFoundError(f"No DSHCL-adapted metrics found under {DSHCL_DIR}")
    long_fields = list(long_rows[0].keys())
    paper_fields = ["Dataset", "Window", "Method", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "Best MAPE"]
    best_fields = ["Dataset", "Split", "Best method", "Window", "MAE", "RMSE", "MAPE", "SMAPE", "R2"]

    write_csv(SUMMARY_DIR / "paper_dshcl_adapted_all_runs.csv", long_rows, long_fields)
    for basename, rows, fields in [
        ("paper_dshcl_adapted_table", paper_rows(long_rows), paper_fields),
        ("paper_dshcl_adapted_best", best_rows(long_rows), best_fields),
    ]:
        write_csv(SUMMARY_DIR / f"{basename}.csv", rows, fields)
        write_markdown(SUMMARY_DIR / f"{basename}.md", rows, fields)
        write_latex(SUMMARY_DIR / f"{basename}.tex", rows, fields)

    print(
        json.dumps(
            {
                "long": str(SUMMARY_DIR / "paper_dshcl_adapted_all_runs.csv"),
                "paper_table": str(SUMMARY_DIR / "paper_dshcl_adapted_table.csv"),
                "best_table": str(SUMMARY_DIR / "paper_dshcl_adapted_best.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
