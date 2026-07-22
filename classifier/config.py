"""
config.py
----------
Central configuration file for the NeuroScan project.
Modify project settings here.
"""

from pathlib import Path
import torch

# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Directory
DATASET_DIR = PROJECT_ROOT / "dataset" / "classification"

# Training Images
TRAIN_DIR = DATASET_DIR / "Training"

# Testing Images
TEST_DIR = DATASET_DIR / "Testing"
 
 # Saved Models
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"

# Device Selection

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Image Parameters
IMAGE_SIZE = 224

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Dataset Parameters

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no tumor",
    "pituitary"
]

# DataLoader Parameters
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True

# Training Parameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
DROPOUT_RATE  = 0.2

# Model Parameters
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
FREEZE_FEATURES =True

# Saving Parameters
BEST_MODEL_NAME = "efficientnet_brain_tumor_best.pth"
LAST_MODEL_NAME = "efficientnet_brain_tumor_last.pth"

# Random Seed
RANDOM_SEED = 42

