"""
dataset.py
---------
Dataset and Dataloader utilities for BT segmentation
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    Subset
)

import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

from segmentation import config
# import config


# Data Aug

def get_train_transforms():
    """Training Augmentations"""
    
    
    return A.Compose([
        A.Resize(
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        ),
        A.HorizontalFlip(p= 0.5),
        A.VerticalFlip(p=0.2),
        
        A.Rotate(
            limit = 15,
            p = 0.5
        ),
        A.Normalize(
            mean = config.MEAN,
            std = config.STD
        ),
        
        ToTensorV2()
    ])
    

def get_val_transform():
    """
    Validation Transform
    """
    
    return A.Compose([
        A.Resize(
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        ),
        A.Normalize(
            mean = config.MEAN,
            std = config.STD
        ),
        ToTensorV2()
        ])


class BrainTumorSegmentationDataset(Dataset):
    """
    Brain Tumor Segmentation Dataset
    """
    
    def __init__(
        self,
        root_dir,
        transform = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        self._build_dataset()
    
    def _build_dataset(self):
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            for image_path in class_dir.glob("*.png"):
                if image_path.name.endswith("_mask.png"):
                    continue
                
                mask_path = image_path.with_name(
                    image_path.stem + "_mask.png"
                )
                
                if mask_path.exists():
                    
                    self.samples.append(
                        
                        (
                            image_path,
                            mask_path,
                            class_dir.name
                        )
                    )
        self.samples.sort(
            key=lambda sample: str(sample[0])
        )
                    
    def __len__(self):
        return len(self.samples)
                
    def __getitem__(self, index):
        image_path, mask_path, class_name = self.samples[index]
        
        # the image
        
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        
        # mask
        
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        mask = (mask > 128).astype(np.float32)
        
        # Augmentation part
        
        if self.transform:
            transformed = self.transform(
                image = image,
                mask = mask
            )
            
            image = transformed["image"]
            mask = transformed["mask"]
        mask = mask.unsqueeze(0)

        return image, mask
    
# dataloaders 

def create_dataloaders():
    """
    Create training and validation dataloaders
    """

    base_dataset = BrainTumorSegmentationDataset(
        root_dir=config.TRAIN_DIR,
        transform=None
    )

    # ratio split

    train_size = int(
        len(base_dataset) * config.TRAIN_SPLIT
    )

    val_size = len(base_dataset)- train_size

    generator = torch.Generator().manual_seed(
        config.SEED
    )
    
    indices = torch.randperm(
        len(base_dataset),
        generator=generator
    ).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = BrainTumorSegmentationDataset(
        root_dir=config.TRAIN_DIR,
        transform=get_train_transforms()
    )
    
    val_dataset = BrainTumorSegmentationDataset(
        root_dir=config.TRAIN_DIR,
        transform=get_val_transform()
    )
    
    train_dataset= Subset(
        train_dataset,
        train_indices
    )
    
    val_dataset = Subset(
        val_dataset,
        val_indices
    )

    # data loaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle= True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle = False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return (train_loader, val_loader)
       
       
# testing testing mic 1 2 
if __name__ == "__main__":
    
    train_loader, val_loader = create_dataloaders()
    
    print("=" * 50)
    
    print("Brain Tumor Segmentation Dataset")
    
    print("=" * 50)
    print(f"Training batches     : {len(train_loader)}")
    print(f"Validation batches   : {len(val_loader)}")
    
    images, masks = next(iter(train_loader))
    
    print()
    
    print(f"Image Shape : {images.shape}")
    print(f"Mask Shape  : {masks.shape}")
    
    print("=" * 50)
    