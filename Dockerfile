# NeuroScan — container for Hugging Face Spaces (or any Docker host).
# Model weights are not baked in: they download from the Hugging Face Hub
# on first start (see streamlit_app/model_store.py).

FROM python:3.12-slim

# Shared libraries OpenCV needs (grad-cam pulls in the non-headless build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user streamlit_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# Run from the project root: the app loads assets/ and .streamlit/ from here.
# XSRF/CORS are disabled because Spaces serve the app inside an iframe,
# where Streamlit's XSRF check otherwise blocks file uploads (HTTP 403).
CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false"]
