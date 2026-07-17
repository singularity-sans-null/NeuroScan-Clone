"""
config.py
--------
Configuration file for the segementation part of the proj
"""

from pathlib import Path
import torch

# Project Directories

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directory

DATA_DIR = PROJECT_ROOT / "dataset" / "segmentation"

# Training dataset
TRAIN_DIR = DATA_DIR

# Training dataset split
TRAIN_SPLIT = 0.8

# Validation dataset split
VALID_SPLIT = 0.2


MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"

MODEL_SAVE_DIR.mkdir(
    parents= True,
    exist_ok=True
)

BEST_MODEL_NAME = "best_unet.pth"

LAST_MODEL_NAME = "last_unet.pth"


# OUtput Directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(
    parents = True,
    exist_ok=True
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok= True
)

# model Training Params

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY =  1e-5
NUM_WORKERS = 2
PIN_MEMORY = True
SHUFFLE = True
SEED = 42

# Device

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Model params

IN_CHANNELS = 3
OUT_CHANNELS = 1

FEATURES = [
    64,
    128,
    256,
    512
]

# Image Normalization

MEAN = (
    0.485,
    0.456,
    0.406
)

STD = (
    0.229,
    0.224,
    0.225
)


# Threshold

MASK_THRESHOLD = 0.5

# Early stopping

EARLY_STOPPING_PATIENCE = 5

# CHeckpoinst


SAVE_BEST_ONLY = True

NUM_VISUALIZATION_IMAGES = 10

# Logging
PRINT_EVERY = 10

# Dataset

IMAGE_SUFFIX = ".png"
MASK_SUFFIX = "_mask.png"

# Tumor CLass Names
CLASSES = [
    "glioma",
    "meningioma",
    "pituitary"
]

NUM_CLASSES = len(CLASSES)

# Loss func Settings

USE_DICE_LOSS = True
USE_BCE_LOSS = True

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
# Inference

DEFAULT_CONFIDENCE = 0.5

# File extensns

VALID_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg"
)

# Display config

LINE_WIDTH = 60

