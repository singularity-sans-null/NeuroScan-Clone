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
import tempfile
import zipfile

from io import BytesIO
from datetime import date

import cv2
import numpy as np
import re
import plotly.graph_objects as go
from skimage.measure import marching_cubes
import torch
import streamlit as st

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pdf_report import create_pdf_report

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
    page_title="Aescanus - NeuroScan",
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

        classifier.config.MODEL_SAVE_DIR /
        classifier.config.BEST_MODEL_NAME
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

        seg_config.MODEL_SAVE_DIR /
        seg_config.BEST_MODEL_NAME
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



# PAGE DESIGN


st.markdown(

    """
    <style>

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: bold;
        color: #4F8BF9;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: gray;
        margin-bottom: 25px;
    }

    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 30px;
    }

    </style>
    """,

    unsafe_allow_html=True

)



# SIDEBAR


with st.sidebar:

    st.title(
        "Aescanus - NeuroScan"
    )

    st.markdown("---")


    st.markdown(
        """
### Team Members

- Adetunji Victor
- Ayoola Marvelous
"""
    )


    st.caption(
        "NACOS UI/DATICAN Competition 2026"
    )


    st.markdown(
        """
### Brain Tumor Detection

NeuroScan is an AI-powered brain tumor
classification system built using
**EfficientNet-B0**.

The model classifies MRI images into:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor
"""
    )


    st.markdown("---")


    alpha = st.slider(

        "Grad-CAM Intensity",

        min_value=0.1,

        max_value=1.0,

        value=0.5,

        step=0.1

    )


    st.markdown("---")


    st.info(
        """
### AI Technology Stack

- Python
- PyTorch
- EfficientNet-B0 — Classification
- U-Net — Tumor Segmentation
- Grad-CAM — Explainability
- Streamlit — Web Interface
"""
    )


    st.warning(
        """
### Educational Disclaimer

This application is intended for
educational and research purposes.

It is NOT a substitute for
professional medical diagnosis.
"""
    )



# PAGE HEADER


st.markdown(f"""<p class="main-title"><img src="data:favicon-32x32.png;base64,{logo}" width="45" style="vertical-align:middle;">NeuroScan</p>""",
    unsafe_allow_html=True,
)



st.markdown(
    '<p class="sub-title">'
    'AI-Powered Brain Tumor Classification & Explainability'
    '</p>',
    unsafe_allow_html=True,
)



st.markdown("---")



# PROCESSING MODE


processing_mode = st.radio(
    "Select Processing Mode",
    [
        "Single Patient",
        "Batch Processing",
        "3D Volumetric Processing" # Add this line
    ],
    horizontal=True
)


st.markdown("---")



# SINGLE PATIENT MODE


if processing_mode == "Single Patient":


    
    # PATIENT INFORMATION
    

    st.subheader(
        "Patient Information"
    )


    patient_col1, patient_col2 = (
        st.columns(2)
    )


    with patient_col1:

        patient_id = st.text_input(

            "Patient ID",

            placeholder="Enter Patient ID"

        )


        patient_age = st.number_input(

            "Age",

            min_value=0,

            max_value=120,

            value=None,

            placeholder="Enter Patient Age"

        )


    with patient_col2:

        patient_sex = st.selectbox(

            "Sex",

            options=[
                "Select Sex",
                "Male",
                "Female"
            ]

        )


        report_date = st.date_input(
            "Scan Date"
        )


    st.markdown("---")


    
    # MRI UPLOAD
    

    uploaded_file = st.file_uploader(

        "📂 Upload a Brain MRI Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    
    # ANALYZE BUTTON
    

    analyze_button = st.button(

        "🧠 Run NeuroScan Analysis",

        use_container_width=True,

        type="primary"

    )


    if analyze_button:


        
        # VALIDATION
        

        if not patient_id:

            st.warning(
                "Please enter the Patient ID."
            )

            st.stop()


        if patient_age is None:

            st.warning(
                "Please enter the patient's age."
            )

            st.stop()


        if patient_sex == "Select Sex":

            st.warning(
                "Please select the patient's sex."
            )

            st.stop()


        if uploaded_file is None:

            st.warning(
                "Please upload a Brain MRI image."
            )

            st.stop()


        
        # LOAD IMAGE
        

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "Unable to open image."
            )

            st.stop()


        
        # ANALYSIS
        

        with st.spinner(

            "NeuroScan is analyzing the MRI..."

        ):

            results = analyze_mri(

                image=image,

                patient_id=patient_id,

                alpha=alpha

            )


        st.success(
            "Analysis completed successfully!"
        )


        
        # DISPLAY IMAGES
        

        col1, col2, col3 = st.columns(3)


        with col1:

            st.subheader(
                "Original MRI"
            )

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            st.subheader(
                "Grad-CAM"
            )

            st.image(

                results["heatmap"],

                caption=(
                    "Regions influencing "
                    "the classifier"
                ),

                use_container_width=True

            )


        with col3:

            st.subheader(
                "Segmentation Mask"
            )

            st.image(

                results["predicted_mask"],

                caption=(
                    "Predicted Tumor Region"
                ),

                clamp=True,

                use_container_width=True

            )


        st.subheader(
            "Tumor Overlay"
        )


        st.image(

            results["overlay"],

            caption=(
                "Predicted tumor region "
                "overlaid on MRI"
            ),

            use_container_width=True

        )


        
        # CREATE PDF
        

        pdf_buffer = create_pdf_report(

            # Patient Information

            patient_id=patient_id,

            patient_age=patient_age,

            patient_sex=patient_sex,

            report_date=report_date,


            # AI Results

            prediction_name=(
                results["prediction_name"]
            ),

            confidence=(
                results["confidence"]
            ),

            tumor_percentage=(
                results["tumor_percentage"]
            ),

            tumor_pixels=(
                results["tumor_pixels"]
            ),

            total_pixels=(
                results["total_pixels"]
            ),


            # Image Paths

            original_image_path=str(

                results["image_paths"]
                ["original_path"]

            ),

            gradcam_path=str(

                results["image_paths"]
                ["gradcam_path"]

            ),

            mask_path=str(

                results["image_paths"]
                ["mask_path"]

            ),

            overlay_path=str(

                results["image_paths"]
                ["overlay_path"]

            ),

            logo_path=LOGO_PATH

        )


        
        # AI ANALYSIS DASHBOARD
        

        st.markdown("---")

        st.markdown(
            "## AI Analysis Dashboard"
        )


        left_info, right_info = (
            st.columns([1.2, 1])
        )


        
        # CLASSIFICATION
        

        with left_info:

            st.subheader(
                "Classification"
            )


            st.metric(

                "Predicted Tumor",

                results[
                    "prediction_name"
                ].title()

            )


            st.metric(

                "Confidence",

                f"{results['confidence'] * 100:.2f}%"

            )


            st.progress(
                float(
                    results["confidence"]
                )
            )


            if (

                results[
                    "prediction_name"
                ].lower()

                == "notumor"

            ):

                st.success(
                    "No Tumor Detected"
                )

            else:

                st.error(

                    "Tumor Detected: "

                    f"{results['prediction_name'].title()}"

                )


        
        # SEGMENTATION
        

        with right_info:

            st.subheader(
                "Segmentation"
            )


            stat1, stat2 = (
                st.columns(2)
            )


            with stat1:

                st.metric(

                    "Tumor Area",

                    f"{results['tumor_percentage']:.2f}%"

                )


                st.metric(

                    "Tumor Pixels",

                    f"{results['tumor_pixels']:,}"

                )


            with stat2:

                st.metric(

                    "Image Pixels",

                    f"{results['total_pixels']:,}"

                )


                mask_shape = (
                    results[
                        "predicted_mask"
                    ].shape
                )


                st.metric(

                    "Mask Size",

                    f"{mask_shape[0]}×{mask_shape[1]}"

                )


        
        # CLASS PROBABILITIES
        

        st.markdown("---")

        st.subheader(
            "Class Probability Distribution"
        )


        for class_name, probability in zip(

            classifier.config.CLASS_NAMES,

            results["probabilities"]

        ):

            st.write(

                f"**{class_name.title()}** "
                f"— {probability * 100:.2f}%"

            )


            st.progress(
                float(probability)
            )


        
        # DOWNLOAD PDF
        

        st.markdown("---")

        st.subheader(
            "Download AI Report"
        )


        st.download_button(

            label="📄 Download NeuroScan Report",

            data=pdf_buffer,

            file_name=(
                f"NeuroScan_Report_{patient_id}.pdf"
            ),

            mime="application/pdf",

            use_container_width=True

        )



# BATCH PROCESSING MODE


elif processing_mode == "Batch Processing":


    st.subheader(
        "Batch MRI Processing"
    )


    st.info(
        """
### Batch Upload Instructions

Upload multiple Brain MRI images.

Each filename must follow this format:

**PATIENTID_AGE_SEX**

Examples:

- NS001_24_Male.png
- NS002_45_Female.jpg
- NS003_31_Male.png

The patient information will automatically
be extracted from each filename.
"""
    )


    
    # DATE
    

    report_date = st.date_input(

        "Report / Scan Date",

        value=date.today(),

        key="batch_date"

    )


    
    # MULTIPLE FILE UPLOAD
    

    uploaded_files = st.file_uploader(

        "📁 Upload MRI Images",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        accept_multiple_files=True,

        key="batch_upload"

    )


    if uploaded_files:


        st.success(

            f"{len(uploaded_files)} MRI image(s) selected."

        )


        
        # FILE PREVIEW
        

        st.subheader(
            "Uploaded Files"
        )


        preview_data = []


        valid_files = []


        for uploaded_file in uploaded_files:


            try:

                patient_id, age, sex = (
                    parse_patient_filename(
                        uploaded_file.name
                    )
                )


                preview_data.append({

                    "Filename":
                    uploaded_file.name,

                    "Patient ID":
                    patient_id,

                    "Age":
                    age,

                    "Sex":
                    sex,

                    "Status":
                    "Ready"

                })


                valid_files.append(
                    uploaded_file
                )


            except Exception as error:


                preview_data.append({

                    "Filename":
                    uploaded_file.name,

                    "Patient ID":
                    "Invalid",

                    "Age":
                    "-",

                    "Sex":
                    "-",

                    "Status":
                    str(error)

                })


        st.dataframe(

            preview_data,

            use_container_width=True,

            hide_index=True

        )


        
        # PROCESS BUTTON
        

        process_batch_button = st.button(

            f"🧠 Process {len(valid_files)} MRI Scan(s)",

            type="primary",

            use_container_width=True

        )


        if process_batch_button:


            if not valid_files:

                st.error(
                    "No valid files available for processing."
                )

                st.stop()


            
            # ZIP BUFFER
            

            zip_buffer = BytesIO()


            successful_reports = []


            failed_reports = []


            progress_bar = st.progress(0)


            status_text = st.empty()


            with zipfile.ZipFile(

                zip_buffer,

                "w",

                zipfile.ZIP_DEFLATED

            ) as zip_file:


                for index, uploaded_file in enumerate(
                    valid_files
                ):


                    try:


                        
                        # PARSE PATIENT INFORMATION
                        

                        patient_id, patient_age, patient_sex = (

                            parse_patient_filename(
                                uploaded_file.name
                            )

                        )


                        status_text.info(

                            f"Processing "
                            f"{index + 1} of "
                            f"{len(valid_files)}: "

                            f"{patient_id}"

                        )


                        
                        # LOAD MRI
                        

                        image = Image.open(

                            uploaded_file

                        ).convert("RGB")


                        
                        # RUN ANALYSIS
                        

                        results = analyze_mri(

                            image=image,

                            patient_id=patient_id,

                            alpha=alpha

                        )


                        
                        # CREATE PDF REPORT
                        

                        pdf_buffer = create_pdf_report(

                            patient_id=patient_id,

                            patient_age=patient_age,

                            patient_sex=patient_sex,

                            report_date=report_date,


                            prediction_name=(
                                results[
                                    "prediction_name"
                                ]
                            ),

                            confidence=(
                                results[
                                    "confidence"
                                ]
                            ),

                            tumor_percentage=(
                                results[
                                    "tumor_percentage"
                                ]
                            ),

                            tumor_pixels=(
                                results[
                                    "tumor_pixels"
                                ]
                            ),

                            total_pixels=(
                                results[
                                    "total_pixels"
                                ]
                            ),


                            original_image_path=str(

                                results["image_paths"]
                                ["original_path"]

                            ),

                            gradcam_path=str(

                                results["image_paths"]
                                ["gradcam_path"]

                            ),

                            mask_path=str(

                                results["image_paths"]
                                ["mask_path"]

                            ),

                            overlay_path=str(

                                results["image_paths"]
                                ["overlay_path"]

                            ),

                            logo_path=LOGO_PATH

                        )


                        
                        # ADD PDF TO ZIP
                        

                        pdf_filename = (

                            f"NeuroScan_Report_"
                            f"{patient_id}.pdf"

                        )


                        zip_file.writestr(

                            pdf_filename,

                            pdf_buffer.getvalue()

                        )


                        successful_reports.append({

                            "Patient ID":
                            patient_id,

                            "Filename":
                            pdf_filename,

                            "Prediction":
                            results[
                                "prediction_name"
                            ].title(),

                            "Confidence":
                            (
                                f"{results['confidence'] * 100:.2f}%"
                            )

                        })


                    except Exception as error:


                        failed_reports.append({

                            "Filename":
                            uploaded_file.name,

                            "Error":
                            str(error)

                        })


                    
                    # UPDATE PROGRESS
                    

                    progress = (

                        (index + 1)

                        /

                        len(valid_files)

                    )


                    progress_bar.progress(
                        progress
                    )


            
            # SAVE ZIP IN SESSION STATE
            

            zip_buffer.seek(0)


            st.session_state[
                "batch_zip"
            ] = zip_buffer.getvalue()


            st.session_state[
                "batch_successful_reports"
            ] = successful_reports


            st.session_state[
                "batch_failed_reports"
            ] = failed_reports


            status_text.empty()


            st.success(

                f"Batch processing complete! "

                f"{len(successful_reports)} report(s) "
                f"generated successfully."

            )


    
    # BATCH RESULTS
    

    if "batch_successful_reports" in st.session_state:


        successful_reports = (

            st.session_state[
                "batch_successful_reports"
            ]

        )


        failed_reports = (

            st.session_state.get(
                "batch_failed_reports",
                []
            )

        )


        if successful_reports:


            st.markdown("---")

            st.subheader(
                "Batch Processing Results"
            )


            st.success(

                f"Successfully generated "

                f"{len(successful_reports)} "

                f"report(s)."

            )


            st.dataframe(

                successful_reports,

                use_container_width=True,

                hide_index=True

            )


        if failed_reports:


            st.warning(

                f"{len(failed_reports)} file(s) "
                f"could not be processed."

            )


            with st.expander(
                "View Failed Files"
            ):

                st.dataframe(

                    failed_reports,

                    use_container_width=True,

                    hide_index=True

                )


        
        # DOWNLOAD ZIP
        

        if "batch_zip" in st.session_state:


            st.markdown("---")


            st.subheader(
                "Download Batch Reports"
            )


            st.download_button(

                label=(
                    "📦 Download All "
                    "NeuroScan Reports (ZIP)"
                ),

                data=st.session_state[
                    "batch_zip"
                ],

                file_name=(
                    "NeuroScan_Batch_Reports.zip"
                ),

                mime="application/zip",

                use_container_width=True,

                type="primary"

            )


 # VOLUMETRIC PROCESSING MODE

# 3D VOLUMETRIC PROCESSING MODE

elif processing_mode == "3D Volumetric Processing":

    st.subheader("3D Volumetric Tumor Reconstruction")

    st.info(
        "Upload a sequence of MRI slices for a single patient. "
        "The AI will process all slices, isolate the tumor regions, "
        "and reconstruct them inside a translucent anatomical brain envelope."
    )

    # 3D Depth Scaling Control
    z_aspect = st.slider(
        "Slice Depth Spacing (Z-Axis Aspect Multiplier)",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.5,
        help="Adjusts anatomical spacing between slices so the scan volume is not flattened."
    )

    # MULTIPLE FILE UPLOAD
    uploaded_slices = st.file_uploader(
        "📁 Upload MRI Sequence",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="3d_upload"
    )

    if uploaded_slices:
        # Step A: Sort files using the natural sorting key
        uploaded_slices = sorted(uploaded_slices, key=lambda x: natural_sort_key(x.name))
        
        st.success(f"{len(uploaded_slices)} slices sorted sequentially and ready.")

        if st.button("🧠 Generate Volumetric Analysis", type="primary", use_container_width=True):
            
            tumor_masks_2d = []
            brain_outline_masks_2d = []
            slice_diagnostic_records = []
            
            pertinent_count = 0
            sum_probabilities = np.zeros(len(classifier.config.CLASS_NAMES))
            sum_confidence = 0.0
            
            volume_tumor_pixels = 0
            volume_total_pixels = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step B: Model-Driven Processing Loop
            for index, slice_file in enumerate(uploaded_slices):
                status_text.info(f"Processing slice {index + 1} of {len(uploaded_slices)}: {slice_file.name}")
                
                try:
                    image = Image.open(slice_file).convert("RGB")
                    image_np = np.array(image)
                    
                    # Run existing analysis pipeline
                    results = analyze_mri(image=image, patient_id="3D_Temp", alpha=0.5)
                    
                    pred_name = results["prediction_name"]
                    conf = results["confidence"]
                    t_pixels = results["tumor_pixels"]
                    
                    # Record diagnostic info per slice
                    slice_diagnostic_records.append({
                        "Filename": slice_file.name,
                        "Prediction": pred_name.replace("notumor", "No Tumor").title(),
                        "Confidence": conf,
                        "Tumor Pixels": t_pixels,
                        "Probabilities": results["probabilities"]
                    })
                    
                    # Track metrics for pertinent scans
                    if pred_name.lower() != "notumor":
                        pertinent_count += 1
                        sum_probabilities += results["probabilities"]
                        sum_confidence += conf
                    
                    volume_tumor_pixels += t_pixels
                    volume_total_pixels += results["total_pixels"]
                    
                    # Extract binary tumor mask
                    mask = results["predicted_mask"]
                    binary_mask = (mask > 0).astype(np.uint8)
                    tumor_masks_2d.append(binary_mask)
                    
                    # Extract brain outer perimeter contour (boundary outline)
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    outline_mask = np.zeros_like(gray, dtype=np.uint8)
                    if contours:
                        # Draw exclusively the outer boundary line
                        cv2.drawContours(outline_mask, contours, -1, color=1, thickness=2)
                    brain_outline_masks_2d.append(outline_mask)

                except Exception as error:
                    st.error(f"Error processing {slice_file.name}: {error}")
                    st.stop()
                
                progress_bar.progress((index + 1) / len(uploaded_slices))

            status_text.success("Slices processed successfully! Assembling visualization...")
            
            # --- CALCULATE AGGREGATE PREDICTION ---
            if pertinent_count > 0:
                avg_probabilities = sum_probabilities / pertinent_count
                avg_confidence = sum_confidence / pertinent_count
                best_class_idx = np.argmax(avg_probabilities)
                best_tumor_class = classifier.config.CLASS_NAMES[best_class_idx]
            else:
                best_tumor_class = "No Tumor Detected"
                avg_confidence = 0.0
                avg_probabilities = np.zeros(len(classifier.config.CLASS_NAMES))
                best_class_idx = None
                
            overall_tumor_percentage = (volume_tumor_pixels / volume_total_pixels) * 100 if volume_total_pixels > 0 else 0

            # --- AI ANALYSIS DASHBOARD ---
            st.markdown("---")
            st.markdown("## Volumetric AI Analysis Dashboard")

            left_info, right_info = st.columns([1.2, 1])

            with left_info:
                st.subheader("Classification (Averaged Across Pertinent Slices)")
                st.metric("Predicted Tumor", best_tumor_class.title())
                
                if best_tumor_class.lower() != "no tumor detected":
                    st.metric("Average Confidence", f"{avg_confidence * 100:.2f}%")
                    st.progress(float(avg_confidence))
                    st.error(f"Tumor Detected: {best_tumor_class.title()}")
                else:
                    st.metric("Average Confidence", "N/A")
                    st.success("No Tumor Detected in Volume")

            with right_info:
                st.subheader("Volumetric Segmentation")
                stat1, stat2 = st.columns(2)
                with stat1:
                    st.metric("Overall Tumor Area", f"{overall_tumor_percentage:.2f}%")
                    st.metric("Total Tumor Voxels", f"{volume_tumor_pixels:,}")
                with stat2:
                    st.metric("Total Volume Voxels", f"{volume_total_pixels:,}")
                    st.metric("Slices Processed", f"{len(uploaded_slices)}")

            # --- RANKED INFLUENTIAL SCANS ---
            st.markdown("---")
            st.subheader("Diagnostic Influence Ranking")
            st.caption("Scans ranked from most influential to least influential in arriving at the overall diagnosis.")

            if best_class_idx is not None:
                # Rank slices by probability towards the diagnosed tumor class, then by tumor area
                ranked_scans = sorted(
                    slice_diagnostic_records,
                    key=lambda x: (x["Probabilities"][best_class_idx], x["Tumor Pixels"]),
                    reverse=True
                )
            else:
                # Fallback ranking by confidence if no tumor was detected
                ranked_scans = sorted(
                    slice_diagnostic_records,
                    key=lambda x: x["Confidence"],
                    reverse=True
                )

            ranking_table_data = [
                {
                    "Rank": idx + 1,
                    "Scan Filename": r["Filename"],
                    "Prediction": r["Prediction"],
                    "Confidence": f"{r['Confidence'] * 100:.2f}%",
                    "Tumor Pixels": f"{r['Tumor Pixels']:,}"
                }
                for idx, r in enumerate(ranked_scans)
            ]
            st.dataframe(ranking_table_data, use_container_width=True, hide_index=True)

            # --- 3D RENDERING ---
            st.markdown("---")
            st.info("Generating 3D volumetric model...")
            
            # Stack arrays along Z-axis (Z, Y, X)
            tumor_volume_3d = np.stack(tumor_masks_2d, axis=0) 
            brain_outline_3d = np.stack(brain_outline_masks_2d, axis=0)
            
            fig = go.Figure()
            
            # 1. Render Brain Boundary Envelope (Translucent Outline)
            if np.max(brain_outline_3d) > 0:
                try:
                    # spacing applies Z-axis aspect ratio so boundaries spread out anatomically
                    b_verts, b_faces, _, _ = marching_cubes(
                        brain_outline_3d, 
                        level=0.5, 
                        spacing=(z_aspect, 1.0, 1.0)
                    )
                    fig.add_trace(go.Mesh3d(
                        x=b_verts[:, 2],  # X (Width)
                        y=b_verts[:, 1],  # Y (Height)
                        z=b_verts[:, 0],  # Z (Depth/Slices)
                        i=b_faces[:, 0], j=b_faces[:, 1], k=b_faces[:, 2],
                        color='lightblue',
                        opacity=0.15,     # Translucent outer boundary
                        name='Cranial Boundary',
                        hoverinfo='skip'
                    ))
                except Exception:
                    pass
                    
            # 2. Render Tumor Volume Mesh
            if np.max(tumor_volume_3d) > 0:
                try:
                    t_verts, t_faces, _, _ = marching_cubes(
                        tumor_volume_3d, 
                        level=0.5, 
                        spacing=(z_aspect, 1.0, 1.0)
                    )
                    fig.add_trace(go.Mesh3d(
                        x=t_verts[:, 2],
                        y=t_verts[:, 1],
                        z=t_verts[:, 0],
                        i=t_faces[:, 0], j=t_faces[:, 1], k=t_faces[:, 2],
                        color='crimson',
                        opacity=0.85,
                        name='Tumor Mass'
                    ))
                except Exception as e:
                    st.error(f"Failed to extract tumor mesh: {e}")
            
            if len(fig.data) > 0:
                fig.update_layout(
                    scene=dict(
                        xaxis_title='X (Width)', 
                        yaxis_title='Y (Height)', 
                        zaxis_title='Z (Slice Depth)',
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=700
                )
                
                st.subheader("Interactive 3D Tumor Model")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No structures detected in the uploaded slices to generate a 3D model.")

st.markdown("---")


with st.expander(
    "How NeuroScan Works"
):


    st.markdown(
        """
### AI Processing Pipeline

1. Upload a brain MRI image.

2. Convert the image to RGB format.

3. Resize and normalize the image.

4. Pass the MRI through the **EfficientNet-B0 Classification Model**.

5. Predict the tumor class and confidence scores.

6. Generate a **Grad-CAM** visualization.

7. Pass the MRI through the **U-Net Segmentation Model**.

8. Generate a predicted tumor mask.

9. Calculate tumor area statistics.

10. Overlay the predicted tumor region onto the MRI.

11. Generate a detailed PDF report.

### Batch Processing

For batch processing, NeuroScan automatically:

1. Reads multiple MRI images.

2. Extracts patient information from filenames.

3. Analyzes every MRI individually.

4. Generates an individual PDF report for each patient.

5. Packages all reports into a ZIP file.

6. Allows the user to download all reports at once.
"""
    )



# MODEL INFORMATION


with st.expander(
    "Model Information"
):


    st.markdown(

        f"""
### Classification Model

**Architecture**

- EfficientNet-B0

**Transfer Learning**

- {"Enabled" if classifier.config.PRETRAINED else "Disabled"}

**Number of Classes**

- {classifier.config.NUM_CLASSES}

**Input Resolution**

- 224 × 224

### Segmentation Model

- U-Net

### Framework

- PyTorch

### Classification Device

- {classifier.config.DEVICE}

### Segmentation Device

- {seg_config.DEVICE}
"""

    )


    st.write(
        "Trained on the Brain Tumor Dataset consisting "
        "of approximately 5,000 MRI images."
    )



# MEDICAL DISCLAIMER


st.markdown("---")


st.warning(
    """
### Medical Disclaimer

NeuroScan is a prototype.

It is **NOT** intended to replace professional
medical diagnosis or the opinion of a qualified
radiologist or physician.

Always consult a qualified medical professional
before making clinical decisions.
"""
)



# PAGE FOOTER


st.markdown("---")


st.markdown(

    """
<div class="footer">

<b>NeuroScan</b><br>

AI-Powered Brain Tumor Classification,
Segmentation & Explainability<br><br>

Developed using

<b>PyTorch</b>,

<b>EfficientNet-B0</b>,

<b>U-Net</b>,

<b>Streamlit</b>,

and

<b>Grad-CAM</b>.

</div>
""",

    unsafe_allow_html=True

)