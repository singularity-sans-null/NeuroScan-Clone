"""
app.py
--

NeuroScan
AI-Powered Brain Tumor Classification System

Features:
- Single Patient Analysis
- Batch MRI Processing
- Brain Tumor Classification
- Tumor Segmentation
- Grad-CAM Explainability
- Tumor Overlay
- Individual PDF Reports
- Batch ZIP Report Download
"""

from pathlib import Path
import sys
import base64
import html
import tempfile
import zipfile

from io import BytesIO
from datetime import date

import cv2
import numpy as np
import re
import torch
import streamlit as st

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pdf_report import create_pdf_report
import ui
import volume3d
from model_store import resolve_model_path

import classifier.config
from classifier.dataset import get_transforms
from classifier.gradcam import GradCAMGenerator
from classifier.model import create_model
from classifier.utils import load_checkpoint

import segmentation.config as seg_config
from segmentation.model import create_model as create_unet
from segmentation.train import load_model as load_unet_model

# PAGE CONFIG


st.set_page_config(
    page_title="NeuroScan",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)



# TEMPORARY DIRECTORY


TEMP_DIR = Path(tempfile.gettempdir()) / "neuroscan"
TEMP_DIR.mkdir(exist_ok=True)

# Helper functions

def get_base64(image_path):
    """
    Convert an image into Base64 format for HTML display.
    """

    with open(image_path, "rb") as file:
        return base64.b64encode(
            file.read()
        ).decode()


def parse_patient_filename(filename):
    """
    Extract patient information from filename.

    Expected format:

        PATIENTID_AGE_SEX.png

    Examples:

        NS001_24_Male.png
        NS002_45_Female.jpg

    Returns:

        patient_id
        patient_age
        patient_sex
    """

    stem = Path(filename).stem

    parts = stem.split("_")

    if len(parts) != 3:

        raise ValueError(
            "Invalid filename format. "
            "Use: PATIENTID_AGE_SEX"
        )

    patient_id = parts[0]

    try:

        patient_age = int(parts[1])

    except ValueError:

        raise ValueError(
            "Age in filename must be a number."
        )

    if patient_age < 0 or patient_age > 120:

        raise ValueError(
            "Age must be between 0 and 120."
        )

    patient_sex = parts[2].capitalize()
    sexes = ["Male", "Female"]
    if patient_sex in sexes:
            pass
    elif f"{patient_sex[0]+sexes[0].lstrip("M")}" in sexes:
        patient_sex = sexes[0]
    elif f"{patient_sex[0]+sexes[1].lstrip("MF")}" in sexes:
        patient_sex = sexes[1]
    else:
        raise ValueError(
            "Sex must be Male or Female"
    )

    # if patient_sex not in ["Male", "Female"]:

    #     raise ValueError(
    #         "Sex must be Male or Female."
    #     )

    return (
        patient_id,
        patient_age,
        patient_sex
    )


def save_analysis_images(
    image,
    heatmap,
    predicted_mask,
    overlay,
    patient_id
):
    """
    Save MRI analysis images into a unique temporary
    directory for each patient.

    This prevents batch patients from overwriting
    each other's images.
    """

    # Create patient-specific folder

    patient_temp_dir = (
        TEMP_DIR / str(patient_id)
    )

    patient_temp_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    # ORIGINAL MRI
  
    original_path = (
        patient_temp_dir / "original.png"
    )

    image.save(original_path)

    # GRAD-CAM
  

    gradcam_path = (
        patient_temp_dir / "gradcam.png"
    )

    cv2.imwrite(
        str(gradcam_path),

        cv2.cvtColor(
            heatmap,
            cv2.COLOR_RGB2BGR
        )
    )


    # SEGMENTATION MASK

    mask_path = (
        patient_temp_dir / "mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        predicted_mask
    )
    # TUMOR OVERLAY

    overlay_uint8 = (
        overlay * 255
    ).astype(np.uint8)

    overlay_path = (
        patient_temp_dir / "overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),

        cv2.cvtColor(
            overlay_uint8,
            cv2.COLOR_RGB2BGR
        )
    )

    return {
        "original_path": original_path,
        "gradcam_path": gradcam_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path
    }

# Natural Sorting Function
# Haven't made the test file drectory yet. Keep the code modular. 
def natural_sort_key(filename):
    """
    Extracts numbers from strings for natural sequential sorting.
    Example: 'slice_2.png' comes before 'slice_10.png'
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


# LOAD CLASSIFICATION MODEL


@st.cache_resource
def load_classification_model():
    """
    Load the trained EfficientNet classification model.
    """

    model = create_model()

    model = load_checkpoint(
        model,

        resolve_model_path(
            classifier.config.MODEL_SAVE_DIR,
            classifier.config.BEST_MODEL_NAME,
        )
    )

    model.eval()

    return model

# LOAD SEGMENTATION MODEL


@st.cache_resource
def load_segmentation_model():
    """
    Load the trained U-Net segmentation model.
    """

    model = create_unet()

    model = load_unet_model(
        model,

        resolve_model_path(
            seg_config.MODEL_SAVE_DIR,
            seg_config.BEST_MODEL_NAME,
        )
    )

    model.eval()

    return model

# LOAD MODELS


classification_model = (
    load_classification_model()
)

segmentation_model = (
    load_segmentation_model()
)


# Enable gradients for Grad-CAM target layer

for parameter in classification_model.features[-1].parameters():

    parameter.requires_grad = True

# IMAGE TRANSFORMS


classification_transform = get_transforms()


segmentation_transform = A.Compose([

    A.Resize(
        seg_config.IMAGE_SIZE,
        seg_config.IMAGE_SIZE
    ),

    A.Normalize(
        mean=seg_config.MEAN,
        std=seg_config.STD
    ),

    ToTensorV2()

])


# GRAD-CAM


gradcam = GradCAMGenerator(
    classification_model
)

# LOAD LOGO


LOGO_PATH = "assets/logo.png"

logo = get_base64(
    LOGO_PATH
)



# MAIN MRI ANALYSIS FUNCTION


def analyze_mri(
    image,
    patient_id,
    alpha=0.5
):
    """
    Complete NeuroScan MRI analysis pipeline.

    Performs:

    1. Image preprocessing
    2. Tumor classification
    3. Probability calculation
    4. Tumor segmentation
    5. Tumor area calculation
    6. Grad-CAM generation
    7. Tumor overlay generation
    8. Saving visualization images

    Returns a dictionary containing all results.
    """

    
    # CLASSIFICATION PREPROCESSING
    

    input_tensor = classification_transform(
        image
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(
        classifier.config.DEVICE
    )


    
    # IMAGE FOR GRAD-CAM
    

    image_np = np.array(

        image.resize(
            (224, 224)
        )

    ).astype(np.float32) / 255.0


    
    # SEGMENTATION PREPROCESSING
    

    segmented = segmentation_transform(

        image=np.array(image)

    )

    segmentation_input = (

        segmented["image"]
        .unsqueeze(0)
        .to(seg_config.DEVICE)

    )


    
    # CLASSIFICATION
    

    with torch.no_grad():

        outputs = classification_model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    prediction_index = prediction.item()

    prediction_name = (
        classifier.config.CLASS_NAMES[
            prediction_index
        ]
    )

    confidence_value = confidence.item()


    
    # SEGMENTATION
    

    with torch.no_grad():

        segmentation_output = (
            segmentation_model(
                segmentation_input
            )
        )

        segmentation_output = torch.sigmoid(
            segmentation_output
        )

        predicted_mask = (

            segmentation_output > 0.5

        ).float()


    predicted_mask = (

        predicted_mask
        .squeeze()
        .cpu()
        .numpy()

    )


    predicted_mask = (

        predicted_mask * 255

    ).astype(np.uint8)


    
    # CREATE TUMOR OVERLAY
    

    overlay_image = np.array(

        image.resize(

            (
                seg_config.IMAGE_SIZE,
                seg_config.IMAGE_SIZE
            )

        )

    ).astype(np.float32) / 255.0


    mask_bool = predicted_mask > 0


    overlay = overlay_image.copy()


    red_mask = np.zeros_like(
        overlay
    )


    red_mask[:, :, 0] = 1.0


    overlay[mask_bool] = (

        (1 - alpha)
        * overlay[mask_bool]

        +

        alpha
        * red_mask[mask_bool]

    )


    overlay = np.clip(
        overlay,
        0,
        1
    )


    
    # SEGMENTATION STATISTICS
    

    tumor_pixels = np.count_nonzero(
        predicted_mask
    )


    total_pixels = predicted_mask.size


    tumor_percentage = (

        tumor_pixels /
        total_pixels

    ) * 100


    
    # GRAD-CAM
    

    heatmap = gradcam.generate(

        input_tensor=input_tensor,

        original_image=image_np,

        target_class=prediction_index,

        alpha=alpha

    )


    
    # SAVE ANALYSIS IMAGES
    

    image_paths = save_analysis_images(

        image=image,

        heatmap=heatmap,

        predicted_mask=predicted_mask,

        overlay=overlay,

        patient_id=patient_id

    )


    
    # RETURN RESULTS
    

    return {

        "prediction_name": prediction_name,

        "prediction_index": prediction_index,

        "confidence": confidence_value,

        "probabilities": (
            probabilities
            .squeeze()
            .cpu()
            .numpy()
        ),

        "predicted_mask": predicted_mask,

        "tumor_pixels": tumor_pixels,

        "total_pixels": total_pixels,

        "tumor_percentage": tumor_percentage,

        "heatmap": heatmap,

        "overlay": overlay,

        "image_paths": image_paths

    }



# FAST BATCHED INFERENCE (3D MODE)


@torch.inference_mode()
def predict_slices_batch(images, batch_size=8):
    """
    Classification and segmentation for many slices at once.

    Unlike analyze_mri, this skips Grad-CAM (which needs a backward pass)
    and does not write images to disk. That makes it much faster for
    volumetric reconstruction, where neither is used.

    Returns a list of dicts with prediction_name, confidence,
    probabilities, tumor_prob (sigmoid map), tumor_pixels, total_pixels.
    """
    results = []

    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]

        cls_batch = torch.stack(
            [classification_transform(img) for img in chunk]
        ).to(classifier.config.DEVICE)

        seg_batch = torch.stack(
            [segmentation_transform(image=np.array(img))["image"] for img in chunk]
        ).to(seg_config.DEVICE)

        probabilities = torch.softmax(classification_model(cls_batch), dim=1)
        seg_probs = torch.sigmoid(segmentation_model(seg_batch))[:, 0]

        probabilities = probabilities.cpu().numpy()
        seg_probs = seg_probs.float().cpu().numpy()

        for probs, tumor_prob in zip(probabilities, seg_probs):
            index = int(np.argmax(probs))
            tumor_pixels = int(np.count_nonzero(tumor_prob > 0.5))
            results.append({
                "prediction_name": classifier.config.CLASS_NAMES[index],
                "prediction_index": index,
                "confidence": float(probs[index]),
                "probabilities": probs,
                "tumor_prob": tumor_prob,
                "tumor_pixels": tumor_pixels,
                "total_pixels": int(tumor_prob.size),
            })

    return results


@st.cache_data(show_spinner=False, max_entries=8)
def build_3d_meshes(brain_stack, tumor_stack, slice_spacing, quality, smoothness):
    """Cached mesh reconstruction so viewer tweaks don't recompute meshes."""
    return volume3d.reconstruct(
        brain_stack, tumor_stack,
        slice_spacing=slice_spacing, quality=quality, smoothness=smoothness,
    )


# UI SETUP


ui.inject_css()

CLASS_NAMES = classifier.config.CLASS_NAMES

DEVICE_LABEL = (
    "GPU" if classifier.config.DEVICE.type == "cuda" else "CPU"
)


def is_tumor_class(name):
    """
    True for any tumor class. Handles 'no tumor', 'notumor', 'no_tumor'.
    """
    normalized = str(name).lower().replace(" ", "").replace("_", "")
    return normalized != "notumor"


def display_class(name):
    return name.title() if is_tumor_class(name) else "No Tumor"


def class_note(name):
    notes = {
        "glioma": "Glial-cell tumor pattern identified by the classifier.",
        "meningioma": "Meningeal tumor pattern identified by the classifier.",
        "pituitary": "Pituitary-region tumor pattern identified by the classifier.",
    }
    return notes.get(
        str(name).lower(),
        "Tumor pattern identified by the classifier."
    )


# SIDEBAR


with st.sidebar:

    ui.sidebar_brand(logo)

    ui.sidebar_label("Analysis settings")

    alpha = st.slider(
        "Grad-CAM intensity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Opacity of the Grad-CAM heatmap blended over the MRI.",
    )

    ui.sidebar_label("Model stack")

    ui.badges([
        ("EfficientNet-B0", "classifier"),
        ("U-Net", "segmentation"),
        ("Grad-CAM", "explainability"),
        ("PyTorch", ""),
    ])

    ui.sidebar_label("Detectable classes")

    ui.badges([(display_class(name), "") for name in CLASS_NAMES])

    ui.sidebar_label("Team")

    ui.sidebar_team(
        ["Adetunji Victor", "Ayoola Marvelous"],
        "NACOS UI/DATICAN Competition 2026",
    )

    ui.sidebar_label("Appearance")

    ui.sidebar_theme_hint()

    ui.sidebar_disclaimer()


# PAGE HEADER


ui.top_bar(logo, DEVICE_LABEL)

ui.hero(
    "",
    "Brain MRI Analysis",
    "Tumor classification, segmentation and Grad-CAM explainability "
    "for single scans, batches and 3D volumes.",
)


# PROCESSING MODE


MODE_SINGLE = "Single Patient"
MODE_BATCH = "Batch Processing"
MODE_VOLUME = "3D Volumetric"

MODE_ICONS = {
    MODE_SINGLE: ":material/person:",
    MODE_BATCH: ":material/stacks:",
    MODE_VOLUME: ":material/view_in_ar:",
}

processing_mode = st.segmented_control(
    "Processing mode",
    options=[MODE_SINGLE, MODE_BATCH, MODE_VOLUME],
    default=MODE_SINGLE,
    required=True,
    format_func=lambda mode: f"{MODE_ICONS[mode]} {mode}",
    label_visibility="collapsed",
    key="processing_mode",
    width="stretch",
)


# SINGLE PATIENT MODE


def render_single_report(data):
    """
    Render a stored single-patient analysis.
    """

    results = data["results"]
    patient = data["patient"]
    prediction = results["prediction_name"]
    tumor = is_tumor_class(prediction)

    ui.section_header(
        "02",
        "Analysis report",
        f"Patient {patient['id']} · {patient['age']} y · "
        f"{patient['sex']} · Scan date {patient['date']}",
    )

    if tumor:
        ui.diagnosis_banner(
            title=display_class(prediction),
            kicker="Classification result",
            note=class_note(prediction),
            confidence=results["confidence"],
            kind="danger",
        )
    else:
        ui.diagnosis_banner(
            title="No tumor detected",
            kicker="Classification result",
            note="The classifier found no tumor pattern in this MRI slice.",
            confidence=results["confidence"],
            kind="success",
        )

    mask_shape = results["predicted_mask"].shape

    ui.stat_cards([
        {
            "label": "Confidence",
            "value": f"{results['confidence'] * 100:.2f}%",
            "sub": "Top-class probability",
        },
        {
            "label": "Tumor area",
            "value": f"{results['tumor_percentage']:.2f}%",
            "sub": "Share of the image",
            "accent": ui.COLORS["danger"] if tumor else ui.COLORS["success"],
        },
        {
            "label": "Tumor pixels",
            "value": f"{results['tumor_pixels']:,}",
            "sub": f"of {results['total_pixels']:,} pixels",
            "accent": ui.COLORS["teal"],
        },
        {
            "label": "Mask resolution",
            "value": f"{mask_shape[0]}×{mask_shape[1]}",
            "sub": "U-Net output",
            "accent": ui.COLORS["muted"],
        },
    ])

    ui.section_header(
        "03",
        "Imaging",
        "Source scan, model attention, predicted tumor region and overlay.",
    )

    ui.image_grid([
        (data["image"], "Input", "Original MRI", "Uploaded scan"),
        (results["heatmap"], "Grad-CAM", "Model attention", "Regions influencing the classifier"),
        (results["predicted_mask"], "U-Net", "Segmentation mask", "Predicted tumor region"),
        (results["overlay"], "Overlay", "Tumor overlay", "Predicted region over the MRI"),
    ])

    ui.section_header(
        "04",
        "Classification detail & report",
    )

    prob_col, report_col = st.columns([1.5, 1], gap="large")

    with prob_col:
        with st.container(border=True):
            ui.card_title("Class probability distribution", "Softmax")
            ui.probability_bars(
                [display_class(name) for name in CLASS_NAMES],
                results["probabilities"],
            )

    with report_col:
        with st.container(border=True):
            ui.card_title("Clinical report", "PDF")
            st.markdown(
                "A formatted PDF with patient details, the AI findings and all "
                "four images, ready to share or archive."
            )
            st.download_button(
                label="Download NeuroScan report",
                icon=":material/download:",
                data=data["pdf"],
                file_name=f"NeuroScan_Report_{patient['id']}.pdf",
                mime="application/pdf",
                type="primary",
                on_click="ignore",
                width="stretch",
            )


def reset_scan_state(result_keys, version_key):
    """
    Drop stored results and bump the form version. Widgets keyed with the
    version (uploaders, patient fields) remount empty on the next run,
    since Streamlit can't clear a file_uploader directly.
    """

    for key in result_keys:
        st.session_state.pop(key, None)

    st.session_state[version_key] = st.session_state.get(version_key, 0) + 1


def new_scan_button(label, result_keys, version_key, key):
    """
    Full-width button that clears the current results and inputs.
    """

    st.write("")
    st.button(
        label,
        icon=":material/restart_alt:",
        on_click=reset_scan_state,
        args=(result_keys, version_key),
        key=key,
        width="stretch",
    )


if processing_mode == MODE_SINGLE:

    st.session_state.setdefault("single_form_version", 0)
    single_version = st.session_state["single_form_version"]

    ui.section_header(
        "01",
        "Patient intake",
        "Enter the patient's details and upload a single axial brain MRI slice.",
    )

    intake_left, intake_right = st.columns([1.15, 1], gap="large")

    with intake_left:
        with st.container(border=True):

            ui.card_title("Patient details", "Required")

            extract_from_filename = st.toggle(
                "Extract details from filename",
                key="single_extract_from_filename",
                help=(
                    "Read Patient ID, Age and Sex from the uploaded file's "
                    "name, using the format PATIENTID_AGE_SEX "
                    "(e.g. NS001_24_Male.png)."
                ),
            )

            # The uploader is rendered later in the script, but its current
            # value is already available in session state via its key.
            pending_upload = st.session_state.get(f"single_upload_{single_version}")

            parsed_id = parsed_age = parsed_sex = None
            filename_error = None

            if extract_from_filename and pending_upload is not None:
                try:
                    parsed_id, parsed_age, parsed_sex = parse_patient_filename(
                        pending_upload.name
                    )
                except ValueError as error:
                    filename_error = str(error)

            patient_col1, patient_col2 = st.columns(2)

            with patient_col1:

                if extract_from_filename:
                    patient_id = st.text_input(
                        "Patient ID",
                        value=parsed_id or "",
                        placeholder="From filename",
                        disabled=True,
                    )

                    patient_age = st.number_input(
                        "Age",
                        min_value=0,
                        max_value=120,
                        value=parsed_age,
                        placeholder="From filename",
                        disabled=True,
                    )
                else:
                    patient_id = st.text_input(
                        "Patient ID",
                        placeholder="e.g. NS-00142",
                        key=f"single_patient_id_{single_version}",
                    )

                    patient_age = st.number_input(
                        "Age",
                        min_value=0,
                        max_value=120,
                        value=None,
                        placeholder="Years",
                        key=f"single_patient_age_{single_version}",
                    )

            with patient_col2:

                sex_options = ["Male", "Female"]

                if extract_from_filename:
                    patient_sex = st.selectbox(
                        "Sex",
                        options=sex_options,
                        index=(
                            sex_options.index(parsed_sex)
                            if parsed_sex in sex_options
                            else None
                        ),
                        placeholder="From filename",
                        disabled=True,
                    )
                else:
                    patient_sex = st.selectbox(
                        "Sex",
                        options=sex_options,
                        index=None,
                        placeholder="Select",
                        key=f"single_patient_sex_{single_version}",
                    )

                report_date = st.date_input(
                    "Scan date",
                    key=f"single_scan_date_{single_version}",
                )

            if extract_from_filename:
                if filename_error:
                    ui.callout(
                        "Couldn't read the filename",
                        f"{filename_error} Rename the file to "
                        "<b>PATIENTID_AGE_SEX</b> (e.g. NS001_24_Male.png) "
                        "or turn off extraction to enter details manually.",
                        kind="danger",
                        icon="!",
                    )
                elif pending_upload is None:
                    st.caption(
                        "Upload a scan named PATIENTID_AGE_SEX "
                        "(e.g. NS001_24_Male.png) to fill these fields."
                    )

    with intake_right:
        with st.container(border=True):

            ui.card_title("MRI scan", "JPG or PNG")

            uploaded_file = st.file_uploader(
                "Brain MRI image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"single_upload_{single_version}",
            )

            preview_image = None

            if uploaded_file is not None:
                try:
                    preview_image = Image.open(uploaded_file).convert("RGB")
                    ui.image_preview(
                        preview_image,
                        uploaded_file.name,
                        f"{preview_image.width}×{preview_image.height} px",
                    )
                except Exception:
                    ui.callout(
                        "Unreadable image",
                        "This file could not be opened as an image. "
                        "Please upload a valid JPG or PNG MRI slice.",
                        kind="danger",
                        icon="!",
                    )

    analyze_button = st.button(
        "Run NeuroScan analysis",
        icon=":material/neurology:",
        type="primary",
        width="stretch",
    )

    if analyze_button:

        missing = []

        if not patient_id:
            missing.append("Patient ID")

        if patient_age is None:
            missing.append("Age")

        if patient_sex is None:
            missing.append("Sex")

        if uploaded_file is None:
            missing.append("MRI image")

        if missing:

            ui.callout(
                "Some details are missing",
                "Please provide: <b>" + ", ".join(missing) + "</b>.",
                kind="warning",
                icon="!",
            )

        elif preview_image is None:

            ui.callout(
                "Unreadable image",
                "The uploaded file could not be opened as an image.",
                kind="danger",
                icon="!",
            )

        else:

            with st.spinner("Running analysis..."):

                results = analyze_mri(
                    image=preview_image,
                    patient_id=patient_id,
                    alpha=alpha,
                )

                pdf_buffer = create_pdf_report(
                    patient_id=patient_id,
                    patient_age=patient_age,
                    patient_sex=patient_sex,
                    report_date=report_date,
                    prediction_name=results["prediction_name"],
                    confidence=results["confidence"],
                    tumor_percentage=results["tumor_percentage"],
                    tumor_pixels=results["tumor_pixels"],
                    total_pixels=results["total_pixels"],
                    original_image_path=str(results["image_paths"]["original_path"]),
                    gradcam_path=str(results["image_paths"]["gradcam_path"]),
                    mask_path=str(results["image_paths"]["mask_path"]),
                    overlay_path=str(results["image_paths"]["overlay_path"]),
                    logo_path=LOGO_PATH,
                )

            # Persist so results survive reruns (e.g. downloading the PDF)
            st.session_state["single_report"] = {
                "results": results,
                "image": preview_image,
                "pdf": pdf_buffer.getvalue(),
                "patient": {
                    "id": patient_id,
                    "age": patient_age,
                    "sex": patient_sex,
                    "date": report_date.strftime("%d %b %Y"),
                },
            }

    if "single_report" in st.session_state:

        render_single_report(st.session_state["single_report"])

        new_scan_button(
            "Clear results & start a new scan",
            ["single_report"],
            "single_form_version",
            key="single_new_scan",
        )

    else:

        st.write("")
        ui.empty_state(
            "No analysis yet",
            "Complete the intake form and run the analysis to see the report here.",
        )


# BATCH PROCESSING MODE


elif processing_mode == MODE_BATCH:

    ui.section_header(
        "01",
        "Batch intake",
        "Process multiple patients. Each scan produces its own PDF report, "
        "and all reports are bundled into one ZIP file.",
    )

    guide_col, upload_col = st.columns([1, 1.3], gap="large")

    with guide_col:

        ui.callout(
            "Filename convention",
            "Patient details are read from each filename using "
            "<code>PATIENTID_AGE_SEX</code>.<br><br>"
            "<code>NS001_24_Male.png</code><br>"
            "<code>NS002_45_Female.jpg</code><br>"
            "<code>NS003_31_M.png</code>",
            kind="info",
            icon="i",
        )

        report_date = st.date_input(
            "Report / scan date",
            value=date.today(),
            key="batch_date",
        )

    with upload_col:
        with st.container(border=True):

            ui.card_title("MRI scans", "Multiple files")

            st.session_state.setdefault("batch_upload_version", 0)

            uploaded_files = st.file_uploader(
                "MRI images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"batch_upload_{st.session_state['batch_upload_version']}",
                label_visibility="collapsed",
            )

    if uploaded_files:

        preview_data = []
        valid_files = []

        for uploaded_file in uploaded_files:

            try:

                patient_id, age, sex = parse_patient_filename(
                    uploaded_file.name
                )

                preview_data.append({
                    "Status": "Ready",
                    "Filename": uploaded_file.name,
                    "Patient ID": patient_id,
                    "Age": str(age),
                    "Sex": sex,
                    "Note": "",
                })

                valid_files.append(uploaded_file)

            except Exception as error:

                preview_data.append({
                    "Status": "Invalid",
                    "Filename": uploaded_file.name,
                    "Patient ID": "",
                    "Age": "",
                    "Sex": "",
                    "Note": str(error),
                })

        invalid_count = len(uploaded_files) - len(valid_files)

        ui.section_header(
            "02",
            "Review queue",
            "Check that patient details were read correctly before processing.",
        )

        ui.stat_cards([
            {"label": "Files selected", "value": f"{len(uploaded_files)}"},
            {
                "label": "Ready",
                "value": f"{len(valid_files)}",
                "accent": ui.COLORS["success"],
            },
            {
                "label": "Invalid names",
                "value": f"{invalid_count}",
                "accent": ui.COLORS["danger"] if invalid_count else ui.COLORS["muted"],
            },
        ])

        st.dataframe(
            preview_data,
            width="stretch",
            hide_index=True,
        )

        process_batch_button = st.button(
            f"Process {len(valid_files)} MRI scan(s)",
            icon=":material/neurology:",
            type="primary",
            width="stretch",
            disabled=not valid_files,
        )

        if process_batch_button:

            zip_buffer = BytesIO()
            successful_reports = []
            failed_reports = []

            with st.status(
                f"Processing {len(valid_files)} scan(s)...",
                expanded=True,
            ) as status:

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

                    for index, uploaded_file in enumerate(valid_files):

                        try:

                            patient_id, patient_age, patient_sex = (
                                parse_patient_filename(uploaded_file.name)
                            )

                            status_text.caption(
                                f"Scan {index + 1} of {len(valid_files)} · {patient_id}"
                            )

                            image = Image.open(uploaded_file).convert("RGB")

                            results = analyze_mri(
                                image=image,
                                patient_id=patient_id,
                                alpha=alpha,
                            )

                            pdf_buffer = create_pdf_report(
                                patient_id=patient_id,
                                patient_age=patient_age,
                                patient_sex=patient_sex,
                                report_date=report_date,
                                prediction_name=results["prediction_name"],
                                confidence=results["confidence"],
                                tumor_percentage=results["tumor_percentage"],
                                tumor_pixels=results["tumor_pixels"],
                                total_pixels=results["total_pixels"],
                                original_image_path=str(results["image_paths"]["original_path"]),
                                gradcam_path=str(results["image_paths"]["gradcam_path"]),
                                mask_path=str(results["image_paths"]["mask_path"]),
                                overlay_path=str(results["image_paths"]["overlay_path"]),
                                logo_path=LOGO_PATH,
                            )

                            pdf_filename = f"NeuroScan_Report_{patient_id}.pdf"

                            zip_file.writestr(pdf_filename, pdf_buffer.getvalue())

                            successful_reports.append({
                                "Patient ID": patient_id,
                                "Age": str(patient_age),
                                "Sex": patient_sex,
                                "Finding": display_class(results["prediction_name"]),
                                "Confidence": float(results["confidence"]),
                                "Tumor area %": round(float(results["tumor_percentage"]), 2),
                                "Report": pdf_filename,
                            })

                        except Exception as error:

                            failed_reports.append({
                                "Filename": uploaded_file.name,
                                "Error": str(error),
                            })

                        progress_bar.progress((index + 1) / len(valid_files))

                status_text.empty()

                status.update(
                    label=(
                        f"Batch complete · {len(successful_reports)} report(s) generated"
                        + (f", {len(failed_reports)} failed" if failed_reports else "")
                    ),
                    state="error" if failed_reports and not successful_reports else "complete",
                    expanded=False,
                )

            zip_buffer.seek(0)

            st.session_state["batch_zip"] = zip_buffer.getvalue()
            st.session_state["batch_successful_reports"] = successful_reports
            st.session_state["batch_failed_reports"] = failed_reports

    elif "batch_successful_reports" not in st.session_state:

        st.write("")
        ui.empty_state(
            "No scans queued",
            "Upload one or more MRI images named PATIENTID_AGE_SEX to begin.",
        )

    # BATCH RESULTS

    if "batch_successful_reports" in st.session_state:

        successful_reports = st.session_state["batch_successful_reports"]
        failed_reports = st.session_state.get("batch_failed_reports", [])

        tumor_count = sum(
            1 for report in successful_reports
            if report["Finding"] != "No Tumor"
        )

        ui.section_header(
            "03",
            "Batch results",
            "Summary of every processed scan. Reports are bundled in the ZIP below.",
        )

        ui.stat_cards([
            {"label": "Reports generated", "value": f"{len(successful_reports)}"},
            {
                "label": "Tumor detected",
                "value": f"{tumor_count}",
                "accent": ui.COLORS["danger"],
            },
            {
                "label": "No tumor",
                "value": f"{len(successful_reports) - tumor_count}",
                "accent": ui.COLORS["success"],
            },
            {
                "label": "Failed",
                "value": f"{len(failed_reports)}",
                "accent": ui.COLORS["warning"] if failed_reports else ui.COLORS["muted"],
            },
        ])

        if successful_reports:

            st.dataframe(
                successful_reports,
                width="stretch",
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                    "Tumor area %": st.column_config.NumberColumn(
                        "Tumor area",
                        format="%.2f%%",
                    ),
                },
            )

        if failed_reports:

            with st.expander(f"View {len(failed_reports)} failed file(s)"):
                st.dataframe(
                    failed_reports,
                    width="stretch",
                    hide_index=True,
                )

        if successful_reports and "batch_zip" in st.session_state:

            st.download_button(
                label=f"Download all {len(successful_reports)} reports (ZIP)",
                icon=":material/folder_zip:",
                data=st.session_state["batch_zip"],
                file_name="NeuroScan_Batch_Reports.zip",
                mime="application/zip",
                type="primary",
                on_click="ignore",
                width="stretch",
            )

        new_scan_button(
            "Clear results & start a new batch",
            ["batch_zip", "batch_successful_reports", "batch_failed_reports"],
            "batch_upload_version",
            key="batch_new_scan",
        )


# 3D VOLUMETRIC PROCESSING MODE


elif processing_mode == MODE_VOLUME:

    ui.section_header(
        "01",
        "Volume intake",
        "Upload an ordered sequence of MRI slices from one patient to reconstruct "
        "the tumor in 3D inside a translucent brain envelope.",
    )

    upload_col, settings_col = st.columns([1.3, 1], gap="large")

    with upload_col:
        with st.container(border=True):

            ui.card_title("MRI slice sequence", "Multiple files")

            # Streamlit can't clear a file_uploader directly, so "Remove all"
            # gives it a fresh key, which mounts a new, empty uploader.
            st.session_state.setdefault("3d_upload_version", 0)

            def clear_3d_uploads():
                st.session_state["3d_upload_version"] += 1

            uploaded_slices = st.file_uploader(
                "MRI sequence",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"3d_upload_{st.session_state['3d_upload_version']}",
                label_visibility="collapsed",
            )

            if uploaded_slices:
                st.button(
                    f"Remove all {len(uploaded_slices)} images",
                    icon=":material/delete_sweep:",
                    on_click=clear_3d_uploads,
                    width="stretch",
                )

    with settings_col:
        with st.container(border=True):

            ui.card_title("How reconstruction works")

            st.caption(
                "Slices are ordered naturally by filename (slice_2 before slice_10), "
                "then classified and segmented in batches. Masks are interpolated "
                "between slices, so the 3D surface stays smooth instead of stepped. "
                "Depth spacing, quality and display options can be changed after "
                "analysis without re-running the models."
            )

    if uploaded_slices:

        uploaded_slices = sorted(
            uploaded_slices,
            key=lambda x: natural_sort_key(x.name),
        )

        ui.callout(
            f"{len(uploaded_slices)} slices ready",
            f"Ordered from <code>{html.escape(uploaded_slices[0].name)}</code> "
            f"to <code>{html.escape(uploaded_slices[-1].name)}</code>.",
            kind="success",
            icon="✓",
        )

        if st.button(
            "Generate volumetric analysis",
            icon=":material/view_in_ar:",
            type="primary",
            width="stretch",
        ):

            images = []
            slice_results = []
            prepared = []
            processing_failed = False
            batch_size = 16 if seg_config.DEVICE.type == "cuda" else 8

            with st.status(
                f"Analyzing {len(uploaded_slices)} slices...",
                expanded=True,
            ) as status:

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                # LOAD

                for slice_file in uploaded_slices:
                    try:
                        images.append(Image.open(slice_file).convert("RGB"))
                    except Exception as error:
                        status.update(
                            label=f"Failed on {slice_file.name}",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Could not read {slice_file.name}: {error}")
                        processing_failed = True
                        break

                # INFER (batched, no Grad-CAM, no disk writes)

                if not processing_failed:
                    try:
                        for start in range(0, len(images), batch_size):
                            end = min(start + batch_size, len(images))
                            status_text.caption(
                                f"Slices {start + 1}–{end} of {len(images)}"
                            )
                            slice_results.extend(
                                predict_slices_batch(images[start:end], batch_size)
                            )
                            progress_bar.progress(end / len(images))
                    except Exception as error:
                        status.update(
                            label="Model inference failed",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Error during analysis: {error}")
                        processing_failed = True

                # PREPARE 3D STACKS ON ONE SHARED GRID

                if not processing_failed:
                    status_text.caption("Preparing volume...")
                    prepared = [
                        volume3d.prepare_slice(np.array(img), res["tumor_prob"])
                        for img, res in zip(images, slice_results)
                    ]
                    status_text.empty()
                    status.update(
                        label=f"Analyzed {len(uploaded_slices)} slices",
                        state="complete",
                        expanded=False,
                    )

            if processing_failed:
                st.stop()

            # AGGREGATE PREDICTION

            slice_diagnostic_records = []
            pertinent_count = 0
            sum_probabilities = np.zeros(len(CLASS_NAMES))
            sum_confidence = 0.0
            volume_tumor_pixels = 0
            volume_total_pixels = 0

            for slice_file, res in zip(uploaded_slices, slice_results):
                slice_diagnostic_records.append({
                    "Filename": slice_file.name,
                    "Prediction": display_class(res["prediction_name"]),
                    "Confidence": res["confidence"],
                    "Tumor Pixels": res["tumor_pixels"],
                    "Probabilities": res["probabilities"],
                })
                if is_tumor_class(res["prediction_name"]):
                    pertinent_count += 1
                    sum_probabilities += res["probabilities"]
                    sum_confidence += res["confidence"]
                volume_tumor_pixels += res["tumor_pixels"]
                volume_total_pixels += res["total_pixels"]

            if pertinent_count > 0:
                avg_probabilities = sum_probabilities / pertinent_count
                avg_confidence = sum_confidence / pertinent_count
                best_class_idx = int(np.argmax(avg_probabilities))
                best_tumor_class = CLASS_NAMES[best_class_idx]
            else:
                avg_probabilities = np.zeros(len(CLASS_NAMES))
                avg_confidence = None
                best_class_idx = None
                best_tumor_class = None

            overall_tumor_percentage = (
                (volume_tumor_pixels / volume_total_pixels) * 100
                if volume_total_pixels > 0 else 0
            )

            # RANK SLICES

            if best_class_idx is not None:
                ranked_scans = sorted(
                    slice_diagnostic_records,
                    key=lambda x: (x["Probabilities"][best_class_idx], x["Tumor Pixels"]),
                    reverse=True,
                )
            else:
                ranked_scans = sorted(
                    slice_diagnostic_records,
                    key=lambda x: x["Confidence"],
                    reverse=True,
                )

            ranking_table_data = [
                {
                    "Rank": idx + 1,
                    "Slice": r["Filename"],
                    "Prediction": r["Prediction"],
                    "Confidence": float(r["Confidence"]),
                    "Tumor pixels": int(r["Tumor Pixels"]),
                }
                for idx, r in enumerate(ranked_scans)
            ]

            # Only the compact uint8/bool stacks are kept. The meshes are built
            # (and cached) at render time, so viewer settings apply instantly.
            st.session_state["volume_report"] = {
                "best_class": best_tumor_class,
                "avg_confidence": avg_confidence,
                "avg_probabilities": avg_probabilities,
                "pertinent_count": pertinent_count,
                "slice_count": len(uploaded_slices),
                "tumor_pct": overall_tumor_percentage,
                "tumor_voxels": volume_tumor_pixels,
                "total_voxels": volume_total_pixels,
                "ranking": ranking_table_data,
                "gray_stack": np.stack([p[0] for p in prepared]),
                "brain_stack": np.stack([p[1] for p in prepared]),
                "tumor_stack": np.stack([p[2] for p in prepared]),
            }

    elif "volume_report" not in st.session_state:

        st.write("")
        ui.empty_state(
            "No volume loaded",
            "Upload a sequence of MRI slices from a single patient to begin.",
        )

    # VOLUMETRIC RESULTS

    if "volume_report" in st.session_state:

        report = st.session_state["volume_report"]

        ui.section_header(
            "02",
            "Volumetric report",
            f"{report['slice_count']} slices analyzed · "
            f"{report['pertinent_count']} showed a tumor pattern.",
        )

        if report["best_class"] is not None:
            ui.diagnosis_banner(
                title=display_class(report["best_class"]),
                kicker="Volume result",
                note="Averaged across slices where a tumor pattern was found.",
                confidence=report["avg_confidence"],
                kind="danger",
            )
        else:
            ui.diagnosis_banner(
                title="No tumor detected",
                kicker="Volume result",
                note="No slice in this sequence showed a tumor pattern.",
                confidence=None,
                kind="success",
            )

        ui.section_header(
            "03",
            "3D reconstruction",
            "Drag to rotate, scroll to zoom, click legend items to toggle layers.",
        )

        viewer_col, controls_col = st.columns([3, 1], gap="large")

        with controls_col:
            with st.container(border=True):

                ui.card_title("Viewer")

                estimated_spacing = volume3d.estimate_slice_spacing(
                    report["brain_stack"]
                )
                auto_depth = st.toggle(
                    "Auto depth",
                    value=estimated_spacing is not None,
                    disabled=estimated_spacing is None,
                    help="Estimates slice spacing from how the head's cross-section "
                         "narrows through the sequence, so the model keeps natural "
                         "proportions. Unavailable when the sequence is too short.",
                )
                if auto_depth and estimated_spacing is not None:
                    slice_spacing = round(estimated_spacing * 2) / 2
                    st.caption(f"Estimated slice spacing: {slice_spacing:.1f}")
                else:
                    slice_spacing = st.slider(
                        "Slice depth spacing",
                        min_value=1.0,
                        max_value=12.0,
                        value=volume3d.DEFAULT_SLICE_SPACING,
                        step=0.5,
                        help="Distance between slices relative to in-plane pixels. "
                             "Raise it if the volume looks flattened.",
                    )

                quality = st.segmented_control(
                    "Mesh quality",
                    options=list(volume3d.QUALITY_PRESETS),
                    default="Balanced",
                    help="Higher quality uses a finer grid and more triangles.",
                ) or "Balanced"

                smoothness = st.slider(
                    "Surface smoothing",
                    min_value=0,
                    max_value=30,
                    value=12,
                    help="Taubin smoothing iterations. Removes voxel ripples "
                         "without shrinking the mesh.",
                )

                show_brain = st.toggle("Brain envelope", value=True)
                show_plane = st.toggle("MRI slice plane", value=True)
                show_axes = st.toggle("Axes", value=False)

                slice_count = report["slice_count"]
                default_slice = volume3d.best_slice_index(
                    report["tumor_stack"], report["brain_stack"]
                )
                plane_index = default_slice
                if show_plane and slice_count > 1:
                    plane_index = st.slider(
                        "Slice plane position",
                        min_value=1,
                        max_value=slice_count,
                        value=default_slice + 1,
                        help="Defaults to the slice with the largest tumor area.",
                    ) - 1

        with st.spinner("Building 3D meshes..."):
            recon = build_3d_meshes(
                report["brain_stack"],
                report["tumor_stack"],
                slice_spacing,
                quality,
                smoothness,
            )

        with viewer_col:
            with st.container(border=True):

                if recon["brain"] is None and recon["tumor"] is None:
                    ui.empty_state(
                        "Nothing to reconstruct",
                        "No structures were detected in the uploaded slices.",
                    )
                else:
                    fig = volume3d.build_figure(
                        recon,
                        gray_stack=report["gray_stack"],
                        brain_stack=report["brain_stack"],
                        show_brain=show_brain,
                        slice_index=plane_index if show_plane else None,
                    )
                    st.plotly_chart(
                        ui.style_3d_figure(fig, show_axes=show_axes),
                        width="stretch",
                        theme="streamlit",
                        config={"displaylogo": False},
                    )

                    faces = sum(
                        m.n_faces for m in (recon["brain"], recon["tumor"]) if m is not None
                    )
                    st.caption(
                        f"{faces:,} triangles · {recon['grid']}×{recon['grid']} grid · "
                        f"built in {recon['seconds']:.2f}s"
                    )

        tumor_mesh = recon["tumor"]

        if tumor_mesh is not None:
            first, last = tumor_mesh.stats["slice_range"]
            multi = last > first
            mass_cards = [
                {
                    "label": "Tumor volume",
                    "value": f"{tumor_mesh.stats['volume_px3']:,.0f}",
                    "sub": "px³ on the 224-px grid, from the mesh",
                    "accent": ui.COLORS["danger"],
                },
                {
                    "label": "Tumor extent",
                    "value": f"{first + 1}–{last + 1}" if multi else f"{first + 1}",
                    "sub": "Slices containing the mass" if multi else "Slice containing the mass",
                    "accent": ui.COLORS["teal"],
                },
                {
                    "label": "Separate masses",
                    "value": f"{recon['components']}",
                    "sub": "After removing noise specks",
                },
            ]
        else:
            mass_cards = [
                {
                    "label": "Tumor volume",
                    "value": "0",
                    "sub": "No tumor surface reconstructed",
                    "accent": ui.COLORS["muted"],
                },
            ]

        ui.stat_cards(mass_cards + [
            {
                "label": "Tumor area",
                "value": f"{report['tumor_pct']:.2f}%",
                "sub": f"{report['tumor_voxels']:,} of {report['total_voxels']:,} px",
                "accent": ui.COLORS["danger"],
            },
            {
                "label": "Slices",
                "value": f"{report['pertinent_count']} / {report['slice_count']}",
                "sub": "With tumor pattern",
            },
        ])

        ui.section_header(
            "04",
            "Diagnostic detail",
        )

        detail_left, detail_right = st.columns([1, 1.4], gap="large")

        with detail_left:
            with st.container(border=True):
                ui.card_title("Averaged class probabilities", "Tumor slices")
                if report["best_class"] is not None:
                    ui.probability_bars(
                        [display_class(name) for name in CLASS_NAMES],
                        report["avg_probabilities"],
                    )
                else:
                    st.caption("Not applicable. No tumor slices in this volume.")

        with detail_right:
            with st.container(border=True):
                ui.card_title("Slice influence ranking", "Highest first")
                st.dataframe(
                    report["ranking"],
                    width="stretch",
                    hide_index=True,
                    height=320,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        ),
                        "Tumor pixels": st.column_config.NumberColumn(
                            "Tumor pixels",
                            format="localized",
                        ),
                    },
                )

        new_scan_button(
            "Clear results & start a new volume",
            ["volume_report"],
            "3d_upload_version",
            key="volume_new_scan",
        )


# ABOUT NEURO SCAN


ui.section_header(
    "i",
    "About NeuroScan",
)

about_pipeline, about_model, about_disclaimer = st.tabs([
    "How it works",
    "Model information",
    "Medical disclaimer",
])

with about_pipeline:

    st.write("")

    ui.pipeline([
        ("Ingest", "The MRI is converted to RGB, resized and normalized."),
        ("Classify", "EfficientNet-B0 predicts the tumor class and confidence."),
        ("Explain", "Grad-CAM highlights the regions that drove the prediction."),
        ("Segment", "U-Net predicts a pixel-level tumor mask."),
        ("Quantify", "Tumor area and pixel counts are computed from the mask."),
        ("Report", "Findings and images are compiled into a PDF report."),
    ])

    st.caption(
        "Batch mode repeats this for every file and bundles the reports into a ZIP. "
        "3D mode batches slices through both models, interpolates the masks between "
        "slices and meshes them into smooth surfaces with marching cubes."
    )

with about_model:

    st.write("")

    ui.spec_grid([
        ("Classifier", "EfficientNet-B0"),
        ("Transfer learning", "Enabled" if classifier.config.PRETRAINED else "Disabled"),
        ("Classes", f"{classifier.config.NUM_CLASSES}"),
        ("Classifier input", "224 × 224"),
        ("Segmentation", "U-Net"),
        ("Framework", "PyTorch"),
        ("Classification device", str(classifier.config.DEVICE).upper()),
        ("Segmentation device", str(seg_config.DEVICE).upper()),
        ("Training data", "~5,000 brain MRI images"),
    ])

with about_disclaimer:

    st.write("")

    ui.callout(
        "Not a medical device",
        "NeuroScan is a research prototype. It is <b>not</b> intended to replace "
        "professional medical diagnosis or the opinion of a qualified radiologist "
        "or physician. Always consult a qualified medical professional before "
        "making clinical decisions.",
        kind="warning",
        icon="!",
    )


# PAGE FOOTER


ui.footer()
