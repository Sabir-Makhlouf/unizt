"""Export the UniZT benchmark summary table into all publication and data formats.

Generates inside ``paper_outputs/``:
  - table_benchmark.csv   (UTF-8 CSV)
  - table_benchmark.tsv   (Tab-separated values)
  - table_benchmark.xlsx  (Styled Excel workbook)
  - table_benchmark.docx  (Academic Word document with three-line table)
  - table_benchmark.tex   (LaTeX booktabs table)
  - table_benchmark.md    (GitHub-Flavored Markdown table)
  - table_benchmark.html  (Styled standalone HTML table)
  - table_benchmark.json  (Structured JSON)
  - table_benchmark.yaml  (YAML)
  - table_benchmark.pdf   (Vector PDF three-line table)
  - table_benchmark.png   (300 DPI PNG three-line table)
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import yaml


HEADERS: list[str] = [
    "Material",
    "Class",
    "B_GPa",
    "G_GPa",
    "Theta_D_K",
    "Gruneisen_gamma",
    "kappa_L_Slack_W_m_K",
    "kappa_L_Ref_W_m_K",
    "ZT_max_UniZT",
    "ZT_max_Ref",
    "eta_max_percent",
]

DISPLAY_HEADERS: list[str] = [
    "Material",
    "Crystal Class",
    "B (GPa)",
    "G (GPa)",
    "θ_D (K)",
    "γ",
    "κ_L Slack (W/m·K)",
    "κ_L Ref (W/m·K)",
    "ZT_max (UniZT)",
    "ZT_max (Ref)",
    "η_max (%)",
]

ROWS: list[list[Any]] = [
    ["Bi2Te3", "Trigonal (layered)", 37.0, 19.6, 163.3, 1.63, 0.62, 1.45, 1.12, 1.05, 9.81],
    ["PbTe", "Cubic (rocksalt)", 41.3, 23.9, 175.4, 1.54, 4.19, 2.20, 1.58, 1.50, 17.35],
    ["ZrNiSn", "Cubic (Half-Heusler)", 123.8, 72.0, 377.0, 1.53, 9.36, 8.50, 0.81, 0.80, 8.51],
    ["Mg2Si", "Cubic (antifluorite)", 55.0, 45.6, 566.1, 1.20, 16.02, 12.10, 0.66, 0.86, 6.42],
    ["SnSe", "Orthorhombic (layered)", 45.3, 26.4, 231.0, 1.53, 0.53, 0.65, 2.80, 2.60, 24.82],
]

CAPTION = (
    "Validation of UniZT predictions across five benchmark thermoelectric crystal classes "
    "using NIST JARVIS-DFT inputs."
)
PARITY_NOTE = (
    "Parity metrics: kappa_L(300 K) R^2 = 0.797 (MAE = 1.54 W m^-1 K^-1); "
    "Peak ZT R^2 = 0.958 (MAE = 0.11)."
)


def _format_cell(val: Any, col_idx: int) -> str:
    if isinstance(val, str):
        return val
    if col_idx in (2, 3, 4):
        return f"{val:.1f}"
    return f"{val:.2f}"


def export_csv(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(ROWS)
    return path


def export_tsv(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.tsv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(HEADERS)
        writer.writerows(ROWS)
    return path


def export_json(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.json"
    records = [dict(zip(HEADERS, row)) for row in ROWS]
    payload = {
        "caption": CAPTION,
        "parity_note": PARITY_NOTE,
        "headers": HEADERS,
        "data": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_yaml(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.yaml"
    records = [dict(zip(HEADERS, row)) for row in ROWS]
    payload = {
        "caption": CAPTION,
        "parity_note": PARITY_NOTE,
        "headers": HEADERS,
        "data": records,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def export_markdown(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.md"
    md_headers = [
        "Material",
        "Class",
        "$B$ (GPa)",
        "$G$ (GPa)",
        "$\\theta_D$ (K)",
        "$\\gamma$",
        "$\\kappa_L^{\\mathrm{Slack}}$ (W/m·K)",
        "$\\kappa_L^{\\mathrm{Ref}}$ (W/m·K)",
        "$ZT_{\\max}^{\\mathrm{UniZT}}$",
        "$ZT_{\\max}^{\\mathrm{Ref}}$",
        "$\\eta_{\\max}$ (%)",
    ]
    align = [":---", ":---"] + [":---:"] * (len(md_headers) - 2)
    lines = [
        f"**Table:** {CAPTION}",
        "",
        "| " + " | ".join(md_headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in ROWS:
        formatted = [_format_cell(v, idx) for idx, v in enumerate(row)]
        lines.append("| " + " | ".join(formatted) + " |")
    lines.extend(["", f"*{PARITY_NOTE}*", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_latex(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.tex"
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{{CAPTION}}}",
        r"\label{tab:unizt_benchmark}",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"Material & Class & $B$ (GPa) & $G$ (GPa) & $\theta_D$ (K) & $\gamma$ & "
        r"$\kappa_L^{\mathrm{Slack}}$ & $\kappa_L^{\mathrm{Ref}}$ & "
        r"$ZT_{\max}^{\mathrm{UniZT}}$ & $ZT_{\max}^{\mathrm{Ref}}$ & $\eta_{\max}$ (\%) \\",
        r"\midrule",
    ]
    for row in ROWS:
        fmt = [_format_cell(v, idx) for idx, v in enumerate(row)]
        lines.append(" & ".join(fmt) + r" \\")
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{11}{l}{Parity metrics: $\kappa_L(300\,\mathrm{K})$ $R^2=0.797$, "
            r"$\mathrm{MAE}=1.54\;\mathrm{W\,m^{-1}\,K^{-1}}$; "
            r"Peak $ZT$ $R^2=0.958$, $\mathrm{MAE}=0.11$.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_html(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.html"
    th_cells = "".join(f"<th>{html.escape(h)}</th>" for h in DISPLAY_HEADERS)
    body_rows: list[str] = []
    for row in ROWS:
        tds = []
        for idx, val in enumerate(row):
            cls = ' class="text"' if idx < 2 else ' class="num"'
            tds.append(f"<td{cls}>{html.escape(_format_cell(val, idx))}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>UniZT Benchmark Table</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    .wrap {{ max-width: 1120px; margin: auto; }}
    caption {{ caption-side: top; text-align: left; font-weight: 600; margin-bottom: 0.75rem; font-size: 1.05rem; }}
    table {{ border-collapse: collapse; width: 100%; border-top: 2px solid #111827; border-bottom: 2px solid #111827; }}
    th, td {{ padding: 0.55rem 0.7rem; font-size: 0.92rem; }}
    thead th {{ border-bottom: 1.5px solid #111827; background: #f8fafc; font-weight: 600; text-align: center; }}
    thead th:nth-child(1), thead th:nth-child(2) {{ text-align: left; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    td.text {{ text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ margin-top: 0.65rem; font-size: 0.88rem; color: #4b5563; }}
  </style>
</head>
<body>
  <div class="wrap">
    <table>
      <caption>{html.escape(CAPTION)}</caption>
      <thead><tr>{th_cells}</tr></thead>
      <tbody>
        {"".join(body_rows)}
      </tbody>
    </table>
    <p class="note">{html.escape(PARITY_NOTE)}</p>
  </div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def export_excel(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "UniZT_Benchmark"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=11, color="111827")
    zebra_fill = PatternFill("solid", fgColor="F4F7FA")
    thin_gray = Side(style="thin", color="D1D5DB")
    thick_dark = Side(style="medium", color="1F4E78")

    ws.append(HEADERS)
    for col_idx in range(1, len(HEADERS) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="left" if col_idx <= 2 else "center",
            vertical="center",
        )
        cell.border = Border(top=thick_dark, bottom=thick_dark, left=thin_gray, right=thin_gray)

    for r_offset, row in enumerate(ROWS, start=2):
        ws.append(row)
        for col_idx in range(1, len(HEADERS) + 1):
            cell = ws.cell(row=r_offset, column=col_idx)
            cell.font = body_font
            if r_offset % 2 == 1:
                cell.fill = zebra_fill
            cell.alignment = Alignment(
                horizontal="left" if col_idx <= 2 else "right",
                vertical="center",
            )
            if col_idx in (3, 4, 5):
                cell.number_format = "0.0"
            elif col_idx >= 6:
                cell.number_format = "0.00"
            bottom_side = thick_dark if r_offset == len(ROWS) + 1 else thin_gray
            cell.border = Border(top=thin_gray, bottom=bottom_side, left=thin_gray, right=thin_gray)

    note_ws = wb.create_sheet(title="Metadata")
    note_ws.append(["Field", "Value"])
    note_ws.append(["Caption", CAPTION])
    note_ws.append(["Parity Metrics", PARITY_NOTE])
    note_ws.column_dimensions["A"].width = 18
    note_ws.column_dimensions["B"].width = 95

    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 24
    for col_idx, header in enumerate(HEADERS, start=1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            [len(str(header))] + [len(_format_cell(r[col_idx - 1], col_idx - 1)) for r in ROWS]
        )
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    wb.save(path)
    return path


def _set_cell_border(cell: Any, **kwargs: dict[str, Any]) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right"):
        edge_data = kwargs.get(edge)
        if edge_data is not None:
            tag = f"w:{edge}"
            el = tc_borders.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                tc_borders.append(el)
            for k, v in edge_data.items():
                el.set(qn(f"w:{k}"), str(v))


def export_docx(out_dir: Path) -> Path:
    path = out_dir / "table_benchmark.docx"
    document = docx.Document()

    section = document.sections[0]
    # Landscape orientation so all 11 scientific columns fit cleanly on one page
    section.orientation = docx.enum.section.WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    cap_p = document.add_paragraph()
    cap_run = cap_p.add_run(f"Table 1. {CAPTION}")
    cap_run.bold = True
    cap_run.font.name = "Calibri"
    cap_run.font.size = Pt(10.5)

    table = document.add_table(rows=len(ROWS) + 1, cols=len(DISPLAY_HEADERS))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    no_border = {"val": "none"}
    top_rule = {"val": "single", "sz": "12", "color": "111827"}
    mid_rule = {"val": "single", "sz": "8", "color": "111827"}
    bot_rule = {"val": "single", "sz": "12", "color": "111827"}

    for c_idx, h_text in enumerate(DISPLAY_HEADERS):
        cell = table.cell(0, c_idx)
        p = cell.paragraphs[0]
        p.alignment = (
            docx.enum.text.WD_ALIGN_PARAGRAPH.LEFT
            if c_idx < 2
            else docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
        )
        run = p.add_run(h_text)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        _set_cell_border(cell, top=top_rule, bottom=mid_rule, left=no_border, right=no_border)

    for r_idx, row in enumerate(ROWS, start=1):
        is_last = r_idx == len(ROWS)
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            p = cell.paragraphs[0]
            p.alignment = (
                docx.enum.text.WD_ALIGN_PARAGRAPH.LEFT
                if c_idx < 2
                else docx.enum.text.WD_ALIGN_PARAGRAPH.RIGHT
            )
            run = p.add_run(_format_cell(val, c_idx))
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            _set_cell_border(
                cell,
                top=no_border,
                bottom=bot_rule if is_last else no_border,
                left=no_border,
                right=no_border,
            )

    note_p = document.add_paragraph()
    note_p.paragraph_format.space_before = Pt(6)
    note_run = note_p.add_run(PARITY_NOTE)
    note_run.italic = True
    note_run.font.name = "Calibri"
    note_run.font.size = Pt(9)
    note_run.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    document.save(path)
    return path


def export_visual_table(out_dir: Path) -> tuple[Path, Path]:
    """Render publication three-line table to both vector PDF and 300 DPI PNG."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    tex_headers = [
        "Material",
        "Crystal Class",
        r"$B$" + "\n(GPa)",
        r"$G$" + "\n(GPa)",
        r"$\theta_D$" + "\n(K)",
        r"$\gamma$",
        r"$\kappa_L^{\mathrm{Slack}}$" + "\n(W/m·K)",
        r"$\kappa_L^{\mathrm{Ref}}$" + "\n(W/m·K)",
        r"$ZT_{\max}^{\mathrm{UniZT}}$",
        r"$ZT_{\max}^{\mathrm{Ref}}$",
        r"$\eta_{\max}$" + "\n(%)",
    ]

    cell_text = [[_format_cell(v, idx) for idx, v in enumerate(r)] for r in ROWS]
    col_widths = [0.075, 0.205, 0.064, 0.064, 0.066, 0.056, 0.095, 0.095, 0.098, 0.092, 0.072]

    fig, ax = plt.subplots(figsize=(12.4, 3.9))
    ax.axis("off")

    ax.text(
        0.012,
        0.92,
        f"Table 1. {CAPTION}",
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )

    tbl = ax.table(
        cellText=cell_text,
        colLabels=tex_headers,
        colWidths=col_widths,
        loc="center",
        bbox=[0.012, 0.16, 0.976, 0.70],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_facecolor("white" if r_idx == 0 or r_idx % 2 == 1 else "#F8FAFC")
        cell.PAD = 0.08
        if r_idx == 0:
            cell.set_height(0.18)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left" if c_idx < 2 else "center")
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
        else:
            cell.set_height(0.11)
            cell.get_text().set_ha("left" if c_idx < 2 else "center")
            if r_idx == len(ROWS):
                cell.visible_edges = "B"
                cell.set_edgecolor("#111827")
                cell.set_linewidth(1.3)
            else:
                cell.visible_edges = ""

    ax.text(
        0.015,
        0.08,
        r"Parity metrics: $\kappa_L(300\,\mathrm{K})$ $R^2=0.797$ ($\mathrm{MAE}=1.54\;\mathrm{W\,m^{-1}\,K^{-1}}$); "
        r"Peak $ZT$ $R^2=0.958$ ($\mathrm{MAE}=0.11$).",
        fontsize=8.8,
        style="italic",
        color="#374151",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    pdf_path = out_dir / "table_benchmark.pdf"
    png_path = out_dir / "table_benchmark.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "paper_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        export_csv(out_dir),
        export_tsv(out_dir),
        export_excel(out_dir),
        export_docx(out_dir),
        export_latex(out_dir),
        export_markdown(out_dir),
        export_html(out_dir),
        export_json(out_dir),
        export_yaml(out_dir),
        *export_visual_table(out_dir),
    ]

    for p in generated:
        print(f"Exported: {p.relative_to(out_dir.parent)} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
