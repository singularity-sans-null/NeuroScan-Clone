"""
download_models.py
------------------
Automatically downloads NeuroScan models
if they do not already exist.
"""

from pathlib import Path
import requests
from tqdm import tqdm

CLASSIFIER_URL = (
    "https://huggingface.co/Ayoola-In/Neuroscan-models/resolve/main/efficientnet_brain_tumor_best.pth"
)

UNET_URL = (
    "https://huggingface.co/Ayoola-In/Neuroscan-models/resolve/main/best_unet.pth"
)


MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def download_model(url: str, destination: Path):
    """
    Downloads a model with a progress bar.
    """

    if destination.exists():

        print(f"✅ {destination.name} already exists.")

        return

    print(f"\nDownloading {destination.name}...")

    response = requests.get(
        url,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    with open(destination, "wb") as file:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
            colour="green"
        ) as progress:

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if chunk:

                    file.write(chunk)

                    progress.update(
                        len(chunk)
                    )

    print(f"✅ Finished downloading {destination.name}")


def ensure_models_exist():
    """
    Downloads all required models if they
    do not already exist.
    """

    print("=" * 60)
    print("Checking AI Models...")
    print("=" * 60)

    download_model(
        CLASSIFIER_URL,
        MODEL_DIR / "efficientnet_brain_tumor_best.pth"
    )

    download_model(
        UNET_URL,
        MODEL_DIR / "best_unet.pth"
    )

    print("\nAll models are ready.\n")

if __name__ == "__main__":

    ensure_models_exist()