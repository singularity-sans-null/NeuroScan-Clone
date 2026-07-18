"""
test_model.py
-------------
Test the trained U-Net segmentation model on a single MRI image
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import create_model
from train import load_checkpoint


device = config.DEVICE

model = create_model()

def load_model(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(config.DEVICE)
    model.eval()

    return model
model = load_model(
    model,
    config.MODEL_SAVE_DIR / "best_unet.pth"
)

model.to(device)
model.eval()

transform = A.Compose([
    A.Resize(
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ),
    A.Normalize(
        mean = config.MEAN,
        std=config.STD
    ),
    ToTensorV2()
]
)

IMAGE_PATH = r"""segmentation/enh_209.png"""

image = cv2.imread(IMAGE_PATH)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
original = image.copy()

transformed = transform(image = image)

input_tensor = transformed["image"]
input_tensor=input_tensor.unsqueeze(0)
input_tensor=input_tensor.to(device)


with torch.no_grad():
    output = model(input_tensor)
    print("=" * 50)
    print("RAW OUTPUT")
    print("shape :", output.shape)
    print("min   :", output.min().item())
    print("max   :", output.max().item())
    print("mean  :", output.mean().item())
    prediction = torch.sigmoid(output)
    print("=" * 50)
    print("AFTER SIGMOID")
    print("min   :", prediction.min().item())
    print("max   :", prediction.max().item())
    print("mean  :", prediction.mean().item())
    prediction = prediction.squeeze().cpu().numpy()
    
    
mask = (prediction> 0.5).astype(np.uint8)
print("=" * 50)
print("MASK")
print("Tumor pixels:", mask.sum())
print("Total pixels:", mask.size)
mask = cv2.resize(
    mask,
    (
        original.shape[1],
        original.shape[0]
    ),
    interpolation=cv2.INTER_NEAREST
)


overlay = original.copy()

overlay[mask == 1] = [255, 0, 0]

overlay = cv2.addWeighted(
    original,
    0.7,
    overlay,
    0.3,
    0
)


plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask, cmap = "gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3,3)
plt.imshow(overlay)
plt.title("Tumor Overlay")
plt.axis("off")

plt.figure(figsize=(6,6))
plt.imshow(prediction, cmap="jet")
plt.colorbar()
plt.title("Probability Map")
plt.show()
plt.tight_layout()
plt.show()



