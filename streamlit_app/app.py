"""
app.py
-------
NeuroScan
AI-Powered Brain Tumor Classification System
"""
from pathlib import Path
import sys

# Add the project root to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



import numpy as np
import torch
import streamlit as st
from PIL import Image

import classifier.config
from classifier.dataset import get_transforms
from classifier.gradcam import GradCAMGenerator
from classifier.model import create_model
from classifier.utils import load_checkpoint


# Page Config
st.set_page_config(
    page_title="Aescanus - NeuroScan",
    page_icon="🧠",
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


model = load_model()

for param in model.features[-1].parameters():
    param.requires_grad = True
    
    
transform = get_transforms()

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
### Technology

- Python
- PyTorch
- EfficientNet-B0
- Streamlit
- Grad-CAM
"""
    )

    st.success(
        "Electrical & Electronics Engineering Final Year Project"
    )

    st.warning(
        """
This application is intended **only for educational and research purposes.**

It is **NOT** a substitute for professional medical diagnosis.
"""
    )


# Page Header
st.markdown(
    '<p class="main-title">🧠 NeuroScan</p>',
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


# Image preprocessing
input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(classifier.config.DEVICE)


image_np = np.array(
    image.resize((224, 224))
).astype(np.float32) / 255.0


# Display the image area
left_col, right_col = st.columns(2)

with left_col:

    st.subheader("Original MRI")

    st.image(
        image,
        use_container_width=True,
    )

with right_col:

    st.subheader("AI Explanation")

    placeholder = st.empty()


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

with right_col:

    st.subheader("Grad-CAM Explanation")

    st.image(
        heatmap,
        caption="Highlighted regions most influenced the prediction.",
        use_container_width=True,
    )


# Prediction Section

st.markdown("## Prediction Results")

metric1, metric2 = st.columns(2)

with metric1:

    st.metric(
        "Predicted Class",
        prediction_name.title()
    )

with metric2:

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )


st.progress(confidence)


if prediction_name.lower() == "notumor":

    st.success("No Brain Tumor Detected")

else:

    st.error(
        f"Tumor Detected: {prediction_name.title()}"
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
### Image Processing Pipeline

1. Upload an MRI image.

2. Convert the image to RGB.

3. Resize to **224 × 224** pixels.

4. Convert to a PyTorch tensor.

5. Normalize using ImageNet statistics.

6. Pass through the pretrained EfficientNet-B0 model.

7. Compute prediction probabilities.

8. Select the class with the highest probability.

9. Generate a Grad-CAM explanation showing the regions
   responsible for the prediction.

10. Display the prediction, confidence score, and
    explainability heatmap.
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