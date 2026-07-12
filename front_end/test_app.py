import streamlit as st
from PIL import Image
from pathlib import Path
import time

# --- Page Configuration ---
st.set_page_config(page_title="Aescanus - NeuroScan", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("Aescanus") # Team identity
    st.caption("Development Team")
    st.info("Team Members:\n- Member 1: Adetunji Victor\n- Member 2: Ayoola Marvellous")
    st.divider()
    st.caption("NACOS UI/DATICAN Competition 2026")
    st.caption("Developed for the NACOS UI/DATICAN Competition 2026.")

# --- Main Interface ---


# --- Header / App Identity ---
try:
    raw_light = Path("front_end/static/logo-light.svg").read_text()
    raw_dark = Path("front_end/static/logo-dark.svg").read_text()
    
    # This finds the start of the actual image code ("<svg") 
    # and drops the XML header completely.
    light_logo_svg = raw_light[raw_light.find("<svg"):]
    dark_logo_svg = raw_dark[raw_dark.find("<svg"):]
except FileNotFoundError:
    st.error("Could not find the SVG files. Check your 'static' folder path.")
    st.stop()

# 2. Inject the raw SVG directly into the HTML
st.markdown(f"""
    <style>
        .header-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 30px;
        }}
        
        /* Target the injected SVG directly */
        .logo-wrapper svg {{
            width: 120px;
            height: auto;
            margin-bottom: 10px;
        }}
        
        /* Default state: hide both */
        .logo-light, .logo-dark {{ display: none; }}
        
        /* Theme toggle */
        [data-theme="light"] .logo-light {{ display: block !important; }}
        [data-theme="dark"] .logo-dark {{ display: block !important; }}
    </style>

    <div class="header-container">
        <!-- Wrap the raw SVG code in divs to control visibility -->
        <div class="logo-wrapper logo-light">{light_logo_svg}</div>
        <div class="logo-wrapper logo-dark">{dark_logo_svg}</div>
        <h1 style="text-align: center;">NeuroScan: Brain Tumor MRI Detection</h1>
    </div>
""", unsafe_allow_html=True)

# Create Tabs
tab1, tab2 = st.tabs(["Analyze Scan", "Project Details"])

with tab1:
    # 2. Upload Section (Moved into the Analysis tab)
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if st.button("Analyze Scan"):
            with st.spinner("NeuroScan is processing the MRI..."):
                time.sleep(1.5) 
                
            # 3. Refined Results Container
            results_container = st.container(border=True)
            with results_container:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(f"**Prediction:** Meningioma")
                with col_b:
                    st.metric(label="Model Confidence", value="90%")
            
            # 4. Side-by-Side Analysis
            st.subheader("Visual Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption='Original MRI Scan', use_container_width=True)
            with col2:
                st.warning("Grad-CAM Heatmap")
                st.image(image, caption='Localized Tumor Region', use_container_width=True)

with tab2:
    # Moved your Documentation here
    st.header("Project Information")
    st.write("### Architecture")
    st.write("Aescanus utilizes an EfficientNet-B0 architecture for classification, "
             "supplemented by Grad-CAM for tumor localization. *to be replaced*")
    st.write("### Dataset")
    st.write("Trained on the Brain Tumor Dataset consisting of ~5,000 MRI images. *to be replaced*")
    
