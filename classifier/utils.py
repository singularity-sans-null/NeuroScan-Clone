"""
utils.py
-----
General utility functions for the project.
"""

import random
from pathlib import Path
import numpy as np
import torch
from classifier import config



def set_seed(seed= config.RANDOM_SEED):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def create_directory(path):
    """
    Create a directory if it does not already exist
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    
    
# Save Model

def save_checkpoint(model, filename):
    """
    Save model weights
    """
    create_directory(config.MODEL_SAVE_DIR)
    save_path = config.MODEL_SAVE_DIR / filename
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to:\n{save_path}")
    
    
# Load model

def load_checkpoint(model, filename):
    """
    Load model weights
    """
    load_path = config.MODEL_SAVE_DIR / filename
    
    model.load_state_dict(
        torch.load(
            load_path,
            map_location= config.DEVICE
        )
    )
    
    model.to(config.DEVICE)
    
    print(f"Model loaded from:\n{load_path}")
    
    return model


# Accuracy

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy for one batch
    """
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions==labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    print(f"The accuracy for this batch is {100*accuracy}%")
    
    return accuracy

def count_parameters(model):
    """
    Returns total and trainable parameters
    """
    
    total = sum(
        p.numel()
        for p in model.parameters()
    )
    
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    
    return total, trainable
    
    
def print_model_summary(model):
    """
    Print model parameter statistics
    """
    total, trainable = count_parameters(model)
    
    print("="*50)
    print("MODEL SUMMARY😁")
    print("="*50)
    
    print(f"Device                  :{config.DEVICE}")
    print(f"Total Parameters        :{total:,}")
    print(f"Trainable Parameters    :{trainable:,}")
    
    print("="*50)
    
class AverageMeter:
    """
    Computes and stores the average value.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def update(self, value, n =1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count