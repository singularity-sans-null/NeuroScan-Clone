"""
Locate model weights, downloading them from the Hugging Face Hub if needed.

Lookup order for each weights file:

1. saved_models/<filename> in the project (local development).
2. The Hugging Face model repo (hosted deployments). The file is cached on
   disk after the first download, so later starts reuse it.

Environment variables:

    HF_MODEL_REPO   Repo holding the weights (default: Ayoola-In/NeuroScan_AI)
    HF_TOKEN        Access token, required only if that repo is private
"""

import os
from pathlib import Path

DEFAULT_MODEL_REPO = "Ayoola-In/NeuroScan_AI"


def resolve_model_path(local_dir, filename):
    local_path = Path(local_dir) / filename

    if local_path.is_file():
        return local_path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=os.environ.get("HF_MODEL_REPO", DEFAULT_MODEL_REPO),
            filename=filename,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
