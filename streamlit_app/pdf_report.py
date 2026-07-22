"""
pdf_report.py
-------------

Generates a professional PDF report
for Neuroscan
"""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

def create_pdf_report(
    prediction_name,
    confidence,
    tumor_percentage,
    tumor_pixels,
    total_pixels,
    original_image_path,
    gradcam_path,
    mask_path,
    overlay_path,
    logo_path
):
    """
    Creates a NeuroScan PDF report
    
    Returns
    -------
    BytesIO
        PDF stored in memory.
    """
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize = (8.27 * inch, 11.69 * inch)
    )
    styles = getSampleStyleSheet()
    story = []
    logo = Image(
        logo_path,
        width = 1.1*inch,
        height= 1.1*inch
    )
    logo.hAlign = "CENTER"
    story.append(logo)
    story.append(
        Spacer(
            1,
            0.15*inch
        )
    )
    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER
    title_style.textColor = colors.darkblue
    
    story.append(
        Paragraph(
            "<font color = '#1565c0'><b>NeuroScan AI Report</b></font>",
            title_style
        )
    )
    
    story.append(
        Paragraph(
            "Brain MRI Classification • Tumor Segmentation • Explainable AI",
            styles["Heading3"]
        )
    )
    
    story.append(
        Spacer(
            1,
            0.3 * inch
        )
    )
    data = [
        ["Prediction", prediction_name.replace("notumor", "no tumor").title()],
        ["Confidence", f"{confidence*100:.2f}%"],
        ["Tumor Area", f"{tumor_percentage:2f}%"],
        ["Tumor Pixels", f"{tumor_pixels:,}"],
        ["Total Pixels", f"{total_pixels:,}"],
        
    ]
    table = Table(
        data,
        colWidths=[2.4 * inch, 3.4 * inch]
    )
    
    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightblue      
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.grey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica-Bold"
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.whitesmoke
            )
        ])
    )
    story.append(table)
    
    story.append(
        Spacer(
            1,
            0.35 * inch
        )
    )
    story.append(
        Paragraph(
            "<b> MRI Analysis Visualizations </b>",
            styles["Heading2"]
        )
    )
    story.append(
        Spacer(
            1,
            0.2 * inch
        )
    )
    image_width = 2.8 * inch
    image_height = 2.8 * inch
    
    original_image = Image(
        original_image_path,
        width = image_width,
        height=image_height
    )
    gradcam_img = Image(
        gradcam_path,
        width = image_width,
        height = image_height
    )
    mask_img = Image(
        mask_path,
        width = image_width,
        height= image_height
    )
    overlay_img = Image(
        overlay_path,
        width=image_width,
        height=image_height
    )
    image_table = [
        
        [
            Paragraph("<b>Original MRI </b>", styles["BodyText"]),
            Paragraph("<b>Grad-CAM</b>", styles["BodyText"])
        ],
        [
            original_image,
            gradcam_img
        ],
        [
            Paragraph("<b>Segmentation Mask</b>", styles["BodyText"]),
            Paragraph("<b>Tumor Overlay</b>", styles["BodyText"])
        ],
        [
            mask_img,
            overlay_img
        ]
        
    ]
    images = Table(
        image_table,
        colWidths=[3.1 * inch, 3.1 * inch]
    )
    images.setStyle(
        
        TableStyle([
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                8.5,
                colors.lightgrey
            )
        ])
    )
    story.append(images)
    story.append(
        Spacer(
            1, 
            0.35 * inch
        )
    )
    story.append(
        Paragraph(
            "<b>Medical Disclaimer</b>",
            styles["Heading2"]
        )
    )
    
    story.append(
        Paragraph(
            """
This report was automatically generated by the NeuroScan AI system.
It is intended solely for educational and research purposes and
must not be used as a substitute for professional medical diagnosis
or treatment.
Always consult a qualified radiologist or physician before making
clinical decisions.
            """,
            styles["BodyText"]  
        )
    )
    story.append(
        Spacer(
            1,
            0.3 * inch
        )
    )
    story.append(
        Paragraph(
            "Generated by NeuroScan • University of Ibadan • 2026",
            styles["Italic"]
        )
    )
    
    doc.build(story)
    buffer.seek(0)
    return buffer