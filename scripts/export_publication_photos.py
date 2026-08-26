from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree


FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
PHOTO_DIR = PROJECT_ROOT / "results" / "photos"
RAW_DIR = PHOTO_DIR / "_raw_regenerated"

CAPTIONS = {
    "fig1_macro_f1_family_comparison": "Figure 1. Macro-F1 Comparison by Baseline Family",
    "fig2_graph_gain": "Figure 2. Propagation Graph Feature Gain",
    "fig3_split_robustness": "Figure 3. Split Robustness Analysis",
    "fig4_pheme_seir_size_prediction": "Figure 4. PHEME Size Prediction with SEIR Baselines",
    "fig5_v1_window_and_ablation": "Figure 5. V1 Observation Window and Modality Ablation",
    "fig6_v2_c1_latent_factors": "Figure 6. V2/C1 Latent Propagation Factors",
    "fig7_v2_c1_sensitivity_and_counterfactual": "Figure 7. V2/C1 Sensitivity and Counterfactual Stress Test",
    "fig8_v2_disentangled_multiseed": "Figure 8. Disentangled V2 Multi-Seed Results",
    "fig9_rumdetect2017_text_fairness": "Figure 9. RumDetect2017 Text-Feature Fairness Audit",
    "fig10_c2_breakout_multiseed": "Figure 10. C2 Breakout Warning Multi-Seed Results",
    "fig11_c3_control_multiseed": "Figure 11. C3 Closed-Loop Control Multi-Seed Results",
    "fig12_c2_c3_case_studies": "Figure 12. C2-C3 Warning and Control Case Studies",
    "fig_weibo_raw_e9_diagnostics": "Figure 13. Raw-Weibo External Validation and Diagnostic Results",
    "fig_weibo_raw_e10_case_studies": "Figure 14. Raw-Weibo Warning and Control Case Studies",
    "fig_weibo_raw_e12_early_warning": "Figure 15. Raw-Weibo Early-Warning Recall and Lead Time",
}

FONT_STACK = "'Times New Roman', Times, serif"
TEXT_FONT_RE = re.compile(r"font-family:\s*[^;\"]+")
WIDTH_RE = re.compile(r'width="([0-9.]+)pt"')
HEIGHT_RE = re.compile(r'height="([0-9.]+)pt"')
VIEWBOX_RE = re.compile(r'viewBox="([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)"')
SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def regenerate_path_text_figures() -> None:
    """Regenerate early figures whose original SVGs converted labels to paths."""
    import scripts.plot_results as plot_results
    import scripts.plot_seir_results as plot_seir_results

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    publication_style()

    df = plot_results.read_metrics(PROJECT_ROOT / "results" / "summary")
    family_df = plot_results.best_by_family(df)
    overall_df = plot_results.best_overall(df)
    plot_results.plot_family_comparison(family_df, RAW_DIR)
    publication_style()
    plot_results.plot_graph_gain(family_df, RAW_DIR)
    publication_style()
    plot_results.plot_split_robustness(overall_df, RAW_DIR)

    publication_style()
    seir_df = plot_seir_results.read_seir_rows(PROJECT_ROOT / "results" / "summary")
    plot_seir_results.plot_pheme_size_prediction(seir_df, RAW_DIR)


def normalize_svg(svg_text: str, caption: str) -> str:
    svg_text = TEXT_FONT_RE.sub(f"font-family: {FONT_STACK}", svg_text)

    width_match = WIDTH_RE.search(svg_text)
    height_match = HEIGHT_RE.search(svg_text)
    viewbox_match = VIEWBOX_RE.search(svg_text)
    if not width_match or not height_match or not viewbox_match:
        raise ValueError("Could not parse SVG size/viewBox.")

    width = float(width_match.group(1))
    height = float(height_match.group(1))
    vb_x, vb_y, vb_w, vb_h = [float(x) for x in viewbox_match.groups()]
    extra_h = max(28.0, min(46.0, vb_h * 0.075))
    new_height = height + extra_h
    new_vb_h = vb_h + extra_h
    caption_size = max(10.0, min(15.0, vb_w * 0.018))
    caption_y = vb_y + vb_h + extra_h * 0.64

    svg_text = WIDTH_RE.sub(f'width="{width:.6f}pt"', svg_text, count=1)
    svg_text = HEIGHT_RE.sub(f'height="{new_height:.6f}pt"', svg_text, count=1)
    svg_text = VIEWBOX_RE.sub(
        f'viewBox="{vb_x:g} {vb_y:g} {vb_w:g} {new_vb_h:.6f}"',
        svg_text,
        count=1,
    )

    caption_element = (
        f'\n  <g id="bottom_figure_caption">\n'
        f'    <text x="{vb_x + vb_w / 2:.6f}" y="{caption_y:.6f}" '
        f'style="font-family: {FONT_STACK}; font-size: {caption_size:.2f}px; '
        f'font-weight: 700; text-anchor: middle; fill: #111111">{caption}</text>\n'
        f'  </g>\n'
    )
    svg_text = svg_text.replace("</svg>", caption_element + "</svg>")
    return move_panel_titles_to_bottom(svg_text)


def parse_svg_numbers(path_d: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", path_d)]


def text_content(text_el: etree._Element) -> str:
    return "".join(text_el.itertext()).strip()


def text_position(text_el: etree._Element) -> tuple[float | None, float | None]:
    x = text_el.get("x")
    y = text_el.get("y")
    if x is not None and y is not None:
        try:
            return float(x), float(y)
        except ValueError:
            pass
    transform = text_el.get("transform", "")
    rotate_match = re.search(
        r"rotate\(-?\d+(?:\.\d+)?\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)",
        transform,
    )
    if rotate_match:
        return float(rotate_match.group(1)), float(rotate_match.group(2))
    translate_match = re.search(r"translate\((-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)", transform)
    if translate_match:
        return float(translate_match.group(1)), float(translate_match.group(2))
    return None, None


def set_style_property(style: str, key: str, value: str) -> str:
    parts = [part.strip() for part in style.split(";") if part.strip()]
    out: list[str] = []
    seen = False
    for part in parts:
        if part.startswith(key + ":"):
            out.append(f"{key}: {value}")
            seen = True
        else:
            out.append(part)
    if not seen:
        out.append(f"{key}: {value}")
    return "; ".join(out)


def set_text_position(text_el: etree._Element, x: float, y: float, anchor: str = "middle") -> None:
    text_el.set("x", f"{x:.6f}")
    text_el.set("y", f"{y:.6f}")
    text_el.set("transform", f"rotate(-0 {x:.6f} {y:.6f})")
    style = text_el.get("style", "")
    style = set_style_property(style, "text-anchor", anchor)
    style = set_style_property(style, "font-family", FONT_STACK)
    text_el.set("style", style)


def is_panel_title(label: str) -> bool:
    if not label or label.startswith("Figure "):
        return False
    if re.match(r"^[a-f]\s{1,2}\S", label):
        return True
    if re.match(r"^[A-F]\.\s+\S", label):
        return True
    if re.match(r"^[A-F]\s{1,2}[A-Za-z]", label):
        return True
    if re.match(r"^W\d+\s+\S", label):
        return True
    if label.startswith("score="):
        return True
    if label in {
        "Stratified split",
        "Temporal split",
        "Graph baseline gain by dataset and split",
        "Best model robustness from stratified to temporal split",
    }:
        return True
    return False


def axes_rect(axes_group: etree._Element) -> tuple[float, float, float, float] | None:
    for path in axes_group.xpath(".//svg:path", namespaces=NS):
        style = path.get("style", "")
        vals = parse_svg_numbers(path.get("d", ""))
        if len(vals) < 8 or "fill: #ffffff" not in style:
            continue
        xs = vals[0::2]
        ys = vals[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def move_panel_titles_to_bottom(svg_text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    caption_texts = root.xpath('.//svg:g[@id="bottom_figure_caption"]//svg:text', namespaces=NS)
    bottom_caption_y = None
    if caption_texts:
        _, bottom_caption_y = text_position(caption_texts[0])

    max_moved_title_y = 0.0
    for axes_group in root.xpath('.//svg:g[starts-with(@id, "axes_")]', namespaces=NS):
        rect = axes_rect(axes_group)
        if rect is None:
            continue
        left, top, right, bottom = rect
        center_x = (left + right) / 2.0
        text_elements = axes_group.xpath(".//svg:text", namespaces=NS)
        title_elements: list[tuple[float, etree._Element]] = []
        non_title_y: list[float] = []

        for text_el in text_elements:
            label = text_content(text_el)
            x, y = text_position(text_el)
            if y is None:
                continue
            if is_panel_title(label):
                title_elements.append((y, text_el))
            else:
                non_title_y.append(y)

        if not title_elements:
            continue

        title_elements.sort(key=lambda item: item[0])
        base_y = max([bottom + 14.0, *(non_title_y or [bottom])]) + 13.0
        # When only a plot title is present and tick labels already sit below the axes,
        # this puts the title below the tick labels. Multi-line case titles keep order.
        for line_index, (_, text_el) in enumerate(title_elements):
            y = base_y + line_index * 12.5
            set_text_position(text_el, center_x, y)
            max_moved_title_y = max(max_moved_title_y, y)

    if bottom_caption_y is not None and max_moved_title_y > bottom_caption_y - 20.0:
        extra = max_moved_title_y - bottom_caption_y + 28.0
        width_match = WIDTH_RE.search(svg_text)
        height_match = HEIGHT_RE.search(svg_text)
        viewbox_match = VIEWBOX_RE.search(svg_text)
        if width_match and height_match and viewbox_match:
            old_height = float(root.get("height", "0pt").replace("pt", ""))
            root.set("height", f"{old_height + extra:.6f}pt")
            vb_x, vb_y, vb_w, vb_h = [float(x) for x in viewbox_match.groups()]
            root.set("viewBox", f"{vb_x:g} {vb_y:g} {vb_w:g} {vb_h + extra:.6f}")
            if caption_texts:
                caption_x, _ = text_position(caption_texts[0])
                if caption_x is not None:
                    set_text_position(caption_texts[0], caption_x, bottom_caption_y + extra)

    xml = etree.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + xml


def source_svg_for(stem: str) -> Path:
    regenerated = RAW_DIR / f"{stem}.svg"
    if regenerated.exists():
        return regenerated
    return FIGURE_DIR / f"{stem}.svg"


def export_svgs() -> list[dict[str, str]]:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for stem, caption in CAPTIONS.items():
        src = source_svg_for(stem)
        if not src.exists():
            rows.append({"stem": stem, "status": "missing", "source": str(src), "output_svg": ""})
            continue
        svg_text = src.read_text(encoding="utf-8", errors="ignore")
        out = PHOTO_DIR / f"{stem}.svg"
        out.write_text(normalize_svg(svg_text, caption), encoding="utf-8")
        source = "regenerated from plotting script" if src.parent == RAW_DIR else str(src)
        rows.append({"stem": stem, "status": "ok", "source": source, "output_svg": str(out)})
    return rows


def copy_regenerated_pngs(rows: list[dict[str, str]]) -> None:
    for row in rows:
        stem = row["stem"]
        regenerated_png = RAW_DIR / f"{stem}.png"
        if regenerated_png.exists():
            shutil.copy2(regenerated_png, PHOTO_DIR / f"{stem}_preview.png")


def audit_outputs(rows: list[dict[str, str]]) -> None:
    chinese_re = re.compile(r"[\u4e00-\u9fff]")
    for row in rows:
        if row["status"] != "ok":
            continue
        path = Path(row["output_svg"])
        text = path.read_text(encoding="utf-8", errors="ignore")
        row["text_elements"] = str(text.count("<text"))
        row["has_chinese"] = str(bool(chinese_re.search(text)))
        row["times_new_roman_mentions"] = str(text.count("Times New Roman"))
        row["has_bottom_caption"] = str("bottom_figure_caption" in text)

    manifest = PHOTO_DIR / "publication_photo_manifest.csv"
    fieldnames = [
        "stem",
        "status",
        "source",
        "output_svg",
        "text_elements",
        "has_chinese",
        "times_new_roman_mentions",
        "has_bottom_caption",
    ]
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    regenerate_path_text_figures()
    rows = export_svgs()
    copy_regenerated_pngs(rows)
    audit_outputs(rows)
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    ok = sum(1 for row in rows if row["status"] == "ok")
    missing = [row["stem"] for row in rows if row["status"] != "ok"]
    print(f"Exported {ok} SVG figures to {PHOTO_DIR}")
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
