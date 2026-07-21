"""
app.py
-------
NeuroScan
AI-Powered Brain Tumor Classification System
"""
from pathlib import Path
import sys
import base64

from utils.download_models import ensure_models_exist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import cv2
from pdf_report import create_pdf_report

import numpy as np
import torch
import streamlit as st
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import classifier.config
from classifier.dataset import get_transforms
from classifier.gradcam import GradCAMGenerator
from classifier.model import create_model
from classifier.utils import load_checkpoint
import segmentation.config as seg_config
from segmentation.model import create_model as create_unet

from segmentation.dataset import (
    get_val_transform
)
from segmentation.train import (
    load_model as load_unet_model
)

temp_dir = Path(tempfile.gettempdir()) / "neuroscan"
temp_dir.mkdir(
    exist_ok=True
)

def get_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = get_base64("assets/logo.png")
logo_path = "assets/logo.png"
# Page Config
st.set_page_config(
    page_title="Aescanus - NeuroScan",
    page_icon= f"assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page Design
st.markdown(
    """
    <style>

    .main-title{
        text-align:center;
        font-size:46px;
        font-weight:bold;
        color:#4F8BF9;
    }

    .sub-title{
        text-align:center;
        font-size:20px;
        color:gray;
        margin-bottom:25px;
    }

    .prediction-card{
        background-color:#1e1e1e;
        padding:20px;
        border-radius:15px;
        border:1px solid #3b82f6;
    }

    .footer{
        text-align:center;
        color:gray;
        font-size:14px;
        margin-top:30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Load the model
@st.cache_resource
def load_model():
    """
    Load the trained EfficientNet model once.
    """

    model = create_model()

    model = load_checkpoint(
        model,
        classifier.config.MODEL_SAVE_DIR / classifier.config.BEST_MODEL_NAME,
    )

    model.eval()

    return model


@st.cache_resource
def load_segmentation_model():
    """
    Load the U-Net model.
    """
    
    model = create_unet()
    model = load_unet_model(
        model,
        seg_config.MODEL_SAVE_DIR / seg_config.BEST_MODEL_NAME
    )
    model.eval()
    return model

ensure_models_exist()
model = load_model()
segmentation_model = load_segmentation_model()

for param in model.features[-1].parameters():
    param.requires_grad = True
    
    
transform = get_transforms()
segmentation_transform = A.Compose([
    A.Resize(
        seg_config.IMAGE_SIZE,
        seg_config.IMAGE_SIZE
    ),
    A.Normalize(
        mean = seg_config.MEAN,
        std = seg_config.STD
    ),
    ToTensorV2()
])
gradcam = GradCAMGenerator(model)


# Page sidebar
with st.sidebar:

    st.title("Aescanus - NeuroScan")

    st.markdown("---")
    st.markdown("""Team Members:\n- Member 1: Adetunji Victor\n- Member 2: Ayoola Marvelous""")
    st.caption("NACOS UI/DATICAN Competition 2026")
    st.caption("Developed for the NACOS UI/DATICAN Competition 2026.")
    st.markdown(
        """
### Brain Tumor Detection

NeuroScan is an AI-powered brain tumor classification system
built using **EfficientNet-B0**.

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
        step=0.1,
    )

    st.markdown("---")

    st.info(
        """
### AI Technology Stack

- Python
- PyTorch
- EfficientNet-B0 — Brain Tumor Classification
- U-Net — Tumor Segmentation
- Grad-CAM — Model Explainability
- Streamlit — Interactive Web Application
"""
    )
    st.warning(
        """
This application is intended **only for educational and research purposes.**

It is **NOT** a substitute for professional medical diagnosis.
"""
    )

# Page Header
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

# file Upload
uploaded_file = st.file_uploader(
    "📂 Upload a Brain MRI Image",
    type=["jpg", "jpeg", "png"],
)

# Wait for user to upload a file
if uploaded_file is None:

    st.info(
        "Upload an MRI image to begin analysis."
    )

    st.stop()


# Load the Image
try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception:

    st.error("Unable to open image.")

    st.stop()

# st.button(
#     label="Run",
#     on_click=
    
# )
# Image preprocessing
input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(classifier.config.DEVICE)


image_np = np.array(
    image.resize((224, 224))
).astype(np.float32) / 255.0


segmented = segmentation_transform(
    image = np.array(image)
)

segmentation_input = (
    segmented["image"].unsqueeze(0).to(seg_config.DEVICE)
)
# Display the image area
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original MRI")
    st.image(
        image,
        use_container_width = True
    )

with col2:
    gradcam_placeholder = st.empty()
    
with col3:
    mask_placeholder = st.empty()


st.markdown("---")

# MOdel inference

with st.spinner("Analyzing MRI..."):

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )
    with torch.no_grad():
        segmentation_output = segmentation_model(
            segmentation_input
        )
        segmentation_output = torch.sigmoid(
            segmentation_output
            )
        predicted_mask = (
            segmentation_output > 0.5
        ).float()

predicted_mask = (
    predicted_mask.squeeze().cpu().numpy()
    )
predicted_mask = (
    predicted_mask * 255
).astype(np.uint8)

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
red_mask = np.zeros_like(overlay)

red_mask [:, :, 0] = 1.0
overlay_alpha = 0.45

overlay[mask_bool] = (
    (1-alpha) * overlay[mask_bool]
    + alpha * red_mask[mask_bool]
)

overlay = np.clip(
    overlay,
    0,
    1
)
tumor_pixels = np.count_nonzero(predicted_mask)
total_pixels = predicted_mask.size

tumor_percentage = (
    tumor_pixels / total_pixels
) * 100

prediction_index = prediction.item()

prediction_name = classifier.config.CLASS_NAMES[prediction_index]

confidence = confidence.item()

# Generate Grad-CAM

heatmap = gradcam.generate(
    input_tensor=input_tensor,
    original_image=image_np,
    target_class=prediction_index,
    alpha=alpha
)


# Display Grad-CAM

with gradcam_placeholder.container():
    st.subheader("Grad-CAM")
    st.image(
        heatmap,
        caption = "Regions Influencing the classifier",
        width="content"
    )
    
with mask_placeholder.container():
    st.subheader("Segmentation Mask")
    st.image(
        predicted_mask,
        caption = "Predicted Tumor Region",
        clamp = True,
        width="content"
    )
    
    st.markdown("---")
    st.subheader("Tumor Overlay")

    st.image(
        overlay,
        caption = "Predicted tumor region overlaid on MRI",
        width = "stretch"
)
temp_dir = Path(tempfile.gettempdir()) / "neuroscan"
temp_dir.mkdir(
    exist_ok=True
)
original_path = temp_dir / "original.png"
image.save(original_path)
gradcam_path = temp_dir / "gradcam.png"

cv2.imwrite(
    str(gradcam_path),
    cv2.cvtColor(
        heatmap,
        cv2.COLOR_RGB2BGR
    )
)

mask_path = temp_dir / "mask.png"

cv2.imwrite(
    str(mask_path),
    predicted_mask
)

overlay_uint8 = (
    overlay * 255
).astype(np.uint8)

overlay_path = temp_dir / "overlay.png"
cv2.imwrite(
    str(overlay_path),
    cv2.cvtColor(
        overlay_uint8,
        cv2.COLOR_RGB2BGR
    )
)
pdf_buffer = create_pdf_report(
    prediction_name=prediction_name,
    confidence= confidence,
    tumor_percentage=tumor_percentage,
    tumor_pixels= tumor_pixels,
    total_pixels=total_pixels,
    original_image_path=str(original_path),
    gradcam_path=str(gradcam_path),
    mask_path = str(mask_path),
    overlay_path = str(overlay_path),
    logo_path=logo_path
)

st.markdown("## AI Analysis Dashboard")

left_info, right_info = st.columns([1.2, 1])

with left_info:

    st.subheader("Classification")

    st.metric(
        "Predicted Tumor",
        prediction_name.title()
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )
    st.progress(float(confidence))
    if prediction_name.lower() == "notumor":
        st.success("No Tumor Detected")
    else:
        st.error(f"Tumor Detected: {prediction_name.title()}")

with right_info:

    st.subheader("Segmentation")

    stat1, stat2 = st.columns(2)

    with stat1:
        st.metric(
            "Tumor Area",
            f"{tumor_percentage:.2f}%"
        )

        st.metric(
            "Tumor Pixels",
            f"{tumor_pixels:,}"
        )

    with stat2:
        st.metric(
            "Image Pixels",
            f"{total_pixels:,}"
        )

        st.metric(
            "Mask Size",
            f"{predicted_mask.shape[0]}×{predicted_mask.shape[1]}"
        )

# Class Probabilities

st.markdown("---")

st.subheader("Class Probability Distribution")

probs = probabilities.squeeze().cpu().numpy()

for class_name, probability in zip(
    classifier.config.CLASS_NAMES,
    probs,
):

    st.write(
        f"**{class_name.title()}** — {probability*100:.2f}%"
    )

    st.progress(float(probability))


# Model Explainability

st.markdown("---")

st.info(
    """
The Grad-CAM visualization highlights the regions of the MRI
that contributed most strongly to the AI's decision.

Red and yellow regions indicate areas that had the greatest
influence on the model's prediction.
"""
)


# HOW THE MODEL WORKS

with st.expander("How NeuroScan Works"):

    st.markdown(
        """
### AI Processing Pipeline

1. Upload a brain MRI image.

2. Convert the image to RGB format.

3. Resize the image to **224 × 224** pixels.

4. Apply preprocessing and normalization.

5. Pass the image through the **EfficientNet-B0 Classification Model**.

6. Predict the tumor class and compute confidence scores.

7. Generate a **Grad-CAM** heatmap highlighting the regions that influenced the classification.

8. Pass the same MRI image through the **U-Net Segmentation Model**.

9. Predict the tumor mask and localize the tumor region.

10. Overlay the predicted tumor mask onto the original MRI for easier visualization.

11. Display:
   - Predicted tumor class
   - Confidence score
   - Class probability distribution
   - Grad-CAM explanation
   - Tumor segmentation mask
   - Tumor overlay visualization
   - Segmentation statistics

12. Generate a downloadable AI analysis report summarizing the results.
"""
    )


# Model INformation

with st.expander("Model Information"):

    st.markdown(
        f"""
**Architecture**

- EfficientNet-B0

**Transfer Learning**

- {"Enabled" if classifier.config.PRETRAINED else "Disabled"}

**Number of Classes**

- {classifier.config.NUM_CLASSES}

**Input Resolution**

- 224 × 224

**Framework**

- PyTorch

**Device**

- {classifier.config.DEVICE}
"""
    )
    st.write("Trained on the Brain Tumor Dataset consisting of ~5,000 MRI images. *to be replaced*")

# Disclaimer
st.markdown("---")
st.subheader("Download AI Report")
st.download_button(
    label = "NeuroScan Report",
    data = pdf_buffer,
    file_name="NeuroScan_Report.pdf",
    mime = "application/pdf",
    use_container_width=True,
)
st.markdown("---")

st.warning(
    """
### Medical Disclaimer

NeuroScan is a prototype.
It is **NOT** intended to replace professional medical diagnosis
or the opinion of a qualified radiologist or physician.

Always consult a medical professional before making any clinical
decisions.
"""
)


# Page Footer

st.markdown("---")

st.markdown(
    """
<div class="footer">

<b>NeuroScan</b><br>

AI-Powered Brain Tumor Classification & Explainability<br><br>

Developed using
<b>PyTorch</b>,
<b>EfficientNet-B0</b>,
<b>Streamlit</b>,
and <b>Grad-CAM</b>.

</div>
""",
    unsafe_allow_html=True,
)