"""
Creates and Returns the EfficientNet Model.
"""
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, 
    EfficientNet_B0_Weights
    )
from classifier import config

def freeze_feature_extractor(model):
    """
    Freeze all feature extraction layers.
    """
    for param in model.features.parameters():
        param.requires_grad = False
        
# Replace the model classifier for the new number of classes

def replace_classifier(model):
    """
    Replace the ImageNet classifier with a classifier
    suitable for brain tumor classification.
    """
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(config.DROPOUT_RATE),
        nn.Linear(
            in_features=in_features,
            out_features=config.NUM_CLASSES
        )
    )

# create the model

def create_model():
    """
    Creates an EfficientNet_B0 model.
    
    Returns
    ------
    model : torch.nn.Module
    """
    model = efficientnet_b0(
        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if config.PRETRAINED
            else None
        )
    )
    
    if config.FREEZE_FEATURES:
        freeze_feature_extractor(model)
    
    replace_classifier(model)
    return model

def print_model_summary(model):
    """
    Print basic information about the model
    """
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    total = sum(
        p.numel()
        for p in model.parameters()
    )
    print(f"Total Parameters        :{total:,}")
    print(f"Trainable Parameters    :{trainable:,}")
    
