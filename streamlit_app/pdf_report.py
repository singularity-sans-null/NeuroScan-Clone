"""
pdf_report.py
-------------

Generates the downloadable NeuroScan PDF report.

Layout follows a plain clinical-report style: a compact letterhead,
key/value blocks separated by hairline rules, a single accent colour,
and a running footer on every page.
"""

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Palette & type
# ---------------------------------------------------------------------------

INK = colors.HexColor("#1a1d21")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d4d7dc")
ACCENT = colors.HexColor("#1f3a5f")
PANEL = colors.HexColor("#f4f5f7")

MARGIN = 18 * mm
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 2 * MARGIN

BASE = "Helvetica"
BOLD = "Helvetica-Bold"


def _style(name, **kw):
    defaults = dict(fontName=BASE, fontSize=9.5, leading=13, textColor=INK)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


S = {
    "brand": _style("brand", fontName=BOLD, fontSize=15, leading=18, textColor=ACCENT),
    "doc_title": _style("doc_title", fontSize=9.5, textColor=MUTED),
    "meta": _style("meta", fontSize=8.5, leading=12, textColor=MUTED, alignment=TA_RIGHT),
    "section": _style("section", fontName=BOLD, fontSize=8, leading=10,
                      textColor=ACCENT, spaceBefore=0, spaceAfter=0),
    "label": _style("label", fontSize=8, leading=10, textColor=MUTED),
    "value": _style("value", fontName=BOLD, fontSize=10, leading=13),
    "result": _style("result", fontName=BOLD, fontSize=16, leading=20),
    "body": _style("body", fontSize=9, leading=13),
    "caption": _style("caption", fontSize=8, leading=10, textColor=MUTED),
    "small": _style("small", fontSize=7.5, leading=10.5, textColor=MUTED),
}


def _section(title):
    """Small-caps style section label with a hairline under it."""
    t = Table([[Paragraph(title.upper(), S["section"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.6, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _kv_grid(pairs, cols=4):
    """Label-over-value cells laid out in a grid, no borders."""
    cells = [
        [Paragraph(label, S["label"]), Paragraph(value, S["value"])]
        for label, value in pairs
    ]
    rows = []
    for i in range(0, len(cells), cols):
        chunk = cells[i:i + cols]
        chunk += [""] * (cols - len(chunk))
        rows.append(chunk)

    inner = []
    for row in rows:
        inner.append([
            Table([[c[0]], [c[1]]], style=[
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]) if c else ""
            for c in row
        ])

    t = Table(inner, colWidths=[CONTENT_W / cols] * cols)
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _image_or_blank(path, size):
    try:
        img = Image(path, width=size, height=size)
        img.hAlign = "CENTER"
        return img
    except Exception:
        return Paragraph("Image unavailable", S["caption"])


def _footer(report_ref):
    def draw(canvas, doc):
        canvas.saveState()
        y = MARGIN - 8 * mm
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, y + 4 * mm, PAGE_W - MARGIN, y + 4 * mm)
        canvas.setFont(BASE, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            MARGIN, y,
            f"NeuroScan  |  {report_ref}  |  For research and educational use only",
        )
        canvas.drawRightString(PAGE_W - MARGIN, y, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def create_pdf_report(
    patient_id,
    patient_age,
    patient_sex,
    report_date,
    prediction_name,
    confidence,
    tumor_percentage,
    tumor_pixels,
    total_pixels,
    original_image_path,
    gradcam_path,
    mask_path,
    overlay_path,
    logo_path,
):
    buffer = BytesIO()
    generated = datetime.now()
    report_ref = f"NS-{generated:%Y%m%d-%H%M%S}"

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 4 * mm,
        title=f"NeuroScan Report - {patient_id}",
        author="NeuroScan",
    )

    story = []

    # ------------------------------------------------------------------
    # Letterhead
    # ------------------------------------------------------------------
    try:
        logo = Image(logo_path, width=13 * mm, height=13 * mm)
    except Exception:
        logo = ""

    brand = [
        Paragraph("NeuroScan", S["brand"]),
        Paragraph("Brain MRI Analysis Report", S["doc_title"]),
    ]
    meta = Paragraph(
        f"Report ref. {report_ref}<br/>"
        f"Generated {generated:%d %b %Y, %H:%M}",
        S["meta"],
    )

    head = Table(
        [[logo, brand, meta]],
        colWidths=[17 * mm, CONTENT_W * 0.5, CONTENT_W * 0.5 - 17 * mm],
    )
    head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, INK),
    ]))
    story += [head, Spacer(1, 5 * mm)]

    # ------------------------------------------------------------------
    # Patient
    # ------------------------------------------------------------------
    scan_date = (
        report_date.strftime("%d %b %Y")
        if hasattr(report_date, "strftime") else str(report_date)
    )
    story += [
        _section("Patient"),
        _kv_grid([
            ("Patient ID", str(patient_id) or "-"),
            ("Age", str(patient_age)),
            ("Sex", str(patient_sex)),
            ("Scan date", scan_date),
        ]),
        Spacer(1, 6 * mm),
    ]

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------
    label = prediction_name.replace("notumor", "no tumor").replace("_", " ")
    label = label.capitalize()
    no_tumor = "no tumor" in label.lower()

    result_panel = Table(
        [[
            [
                Paragraph("Model classification", S["label"]),
                Paragraph(label, S["result"]),
            ],
            [
                Paragraph("Confidence", S["label"]),
                Paragraph(f"{confidence * 100:.1f}%", S["result"]),
            ],
        ]],
        colWidths=[CONTENT_W * 0.65, CONTENT_W * 0.35],
    )
    result_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    seg_rows = [
        ["Segmented tumour area", f"{tumor_percentage:.2f}% of slice"],
        ["Tumour pixels", f"{tumor_pixels:,}"],
        ["Total pixels", f"{total_pixels:,}"],
    ]
    seg = Table(
        [[Paragraph(a, S["body"]), Paragraph(b, S["body"])] for a, b in seg_rows],
        colWidths=[CONTENT_W * 0.45, CONTENT_W * 0.55],
    )
    seg.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
    ]))

    story += [_section("Findings"), Spacer(1, 3 * mm), result_panel, Spacer(1, 3 * mm), seg]
    if no_tumor and tumor_pixels:
        story += [
            Spacer(1, 2 * mm),
            Paragraph(
                "Note: the segmentation model flagged some pixels although the "
                "classifier predicts no tumour. Review the mask before drawing "
                "conclusions.",
                S["small"],
            ),
        ]
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    size = 52 * mm
    col = CONTENT_W / 2

    def fig(path, n, caption):
        return [
            _image_or_blank(path, size),
            Paragraph(f"<b>Fig. {n}</b>  {caption}", S["caption"]),
        ]

    figs = Table(
        [
            [fig(original_image_path, 1, "Input MRI slice"),
             fig(gradcam_path, 2, "Grad-CAM attention map")],
            [fig(mask_path, 3, "Predicted segmentation mask"),
             fig(overlay_path, 4, "Mask overlaid on input")],
        ],
        colWidths=[col, col],
    )
    figs.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [KeepTogether([_section("Imaging"), Spacer(1, 3 * mm), figs])]

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    story += [
        Spacer(1, 5 * mm),
        Paragraph(
            "<b>Disclaimer.</b> This report was produced automatically by the "
            "NeuroScan software (University of Ibadan) for research and "
            "educational purposes. It is not a diagnosis and must not replace "
            "review by a qualified radiologist or physician.",
            S["small"],
        ),
    ]

    footer = _footer(report_ref)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer
