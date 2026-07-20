"""
train.py
Training pipeline for Brain tumor segmentation using U-Net
"""

from pathlib import Path
from tqdm import tqdm


import torch
import torch.nn as nn

# import config
from segmentation import config

from segmentation.model import (
    create_model,
    initialize_weights
)

# from model import (
#     create_model,
#     initialize_weights
# )
from segmentation.dataset import create_dataloaders
# from dataset import create_dataloaders

# Dice Coefficient func

def dice_coefficient(
    predicitions,
    targets,
    smooth=1e-6
):
    """
    Compute Dice Coefficient
    
    Parameters
    ----------
    predicitions: Tensor
        Raw model outputs (logits).
    
    targets : Tensor
        Ground truth mask.
        
    Returns
    -------
    float
        Dice score
    """
    predicitions = torch.sigmoid(predicitions)
    predicitions = (
        predicitions >
        config.MASK_THRESHOLD
    ).float()
    
    predicitions = predicitions.view(-1)
    
    targets = targets.view(-1)
    
    intersections = (
        predicitions *
        targets
    ).sum()
    dice = (
        2* intersections + smooth
    ) / (
        predicitions.sum() + targets.sum() + smooth
    )
    
    return dice

# Dice loss

class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Segmentation.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (
            2 * intersection + self.smooth
        ) / (
            predictions.sum()
            + targets.sum()
            + self.smooth
        )
        return 1 - dice
# combined BCE + DIce loss

class CombinedLoss(nn.Module):
    """
    combined loss
    ------------
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(self, predictions, targets):
        bce_loss = self.bce(
            predictions,
            targets
        )
        dice_loss = self.dice(
            predictions,
            targets
        )
        total_loss = (
            config.BCE_WEIGHT * bce_loss
            +
            config.DICE_WEIGHT * dice_loss
        )
        return total_loss
    
    
# Save checkpoint

def save_checkpoint(checkpoint, filename):
    """
    Save training checkpoint
    """
    
    torch.save(
        checkpoint,
        filename
    )
    
    print(f"Checkpoint saved to {filename}")
    
    
def load_checkpoint(
    checkpoint_path, 
    model, 
    optimizer=None, 
    scheduler = None
    ):
    """
    Load checkpoint.
    
    Returns
    -------
    start_epoch
    """
    
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.DEVICE
    )
    
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    
    if optimizer is not None:
        
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        
    if (
        scheduler is not None and
        checkpoint.get(
            "scheduler_state_dict"
        ) is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        
    print(
        f"Loaded checkpoint from {checkpoint_path}"
    )
    
    return checkpoint["epoch"]

DEVICE = config.DEVICE


# training func

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):
    """
    Train the model for one epoch
    
    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : Loss Function
    optimizer : Optimizer
    
    Returns
    -------
    epoch_loss : float
    epoch_dice : float
    """
    
    model.train()
    
    running_loss = 0.0
    running_dice = 0.0
    
    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )
    
    for images, masks in progress_bar:
        # move tensors to device
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(
            set_to_none = True
        )
        
        
        outputs = model(images)
        
        loss = criterion(
            outputs,
            masks
        )
        
        loss.backward()
        optimizer.step()

        dice = dice_coefficient(
            outputs.detach(),
            masks
        )
        running_loss += loss.item()
        running_dice += dice.item()
        
        # update the progress bar
        
        progress_bar.set_postfix(
            loss = f"{loss.item():.4f}",
            dice = f"{dice.item():.4f}"
        )
        
    # Epoch stats
    
    epoch_loss = (
        running_loss / len(loader)
    )
    
    epoch_dice = (
        running_dice / len(loader)
    )
    
    return (
        epoch_loss,
        epoch_dice
    )
    

# epoch validation Func
def validate_one_epoch(
    model,
    loader,
    criterion
):
    """
    Validate the model for one epoch
    
    Returns
    -------
    epoch_loss : float
    epoch_Dice : float
    """
    
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    
    progress_bar  = tqdm(
        loader,
        desc="Validation",
        leave = False
    )
    
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            
            loss = criterion(
                outputs,
                masks
            )
            dice = dice_coefficient(
                outputs,
                masks
            )
            running_loss += loss.item()
            running_dice += dice.item()
            
            progress_bar.set_postfix(
                loss = f"{loss.item():.4f}",
                dice = f"{dice.item():.4f}"
            )
            
        epoch_loss = running_loss/ len(loader)
        epoch_dice = running_dice/ len(loader)
        
        return epoch_loss, epoch_dice
    
    # training loop
    
def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs
):
    """
    complete training loop
    """
    best_dice = 0.0
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": []
    }
    
    for epoch in range(epochs):
        print("\n" + "="*70)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("="*70)
        
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
        val_loss, val_dice = validate_one_epoch(
            model,
            val_loader,
            criterion
        )
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        
        # epoch summary.
        
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Dice : {train_dice:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Dice   : {val_dice:.4f}")
        
        # the best model
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_dice": train_dice,
                "val_dice": val_dice
            }
            
            save_checkpoint(
                checkpoint,
                Path(config.MODEL_SAVE_DIR) / 
                config.BEST_MODEL_NAME
            )
            
            print(f"New Best Dice: {best_dice:.4f}")
        
        else:
            patience_counter += 1
            print(
                f"No improvement "
                f"({patience_counter})/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )
        
        # stop if patience exceeded
        if (patience_counter>= config.EARLY_STOPPING_PATIENCE):
            print()
            print("Early stopping triggered")
            
            break
    return history


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
        
def main():
    Path(config.MODEL_SAVE_DIR).mkdir(
        parents = True,
        exist_ok=True
    )
    
    # Dataloaders
    
    train_loader, val_loader = create_dataloaders()
    
    print(f"Training batches        : {len(train_loader)}")
    print(f"Validation batches      : {len(val_loader)}")
    
    # Model
    model = create_model().to(DEVICE)
    
    initialize_weights(model)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config.LEARNING_RATE
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        factor=0.5,
        patience=2
    )
    # train
    
    history = fit(
        model = model,
        train_loader= train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = config.NUM_EPOCHS
    )
    
    # save the final checkpoint
    
    last_checkpoint = {
        "epoch": config.NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history
    }
    
    save_checkpoint(
        last_checkpoint,
        Path(config.MODEL_SAVE_DIR) / 
        config.LAST_MODEL_NAME
    )
    
    print()
    print("=" * 70)
    print("Training Finished Successfully!")
    print("="* 70)
    print(f"Best model = "
          f"{Path(config.MODEL_SAVE_DIR / config.BEST_MODEL_NAME)}")
    
    print(f"Last model : "
          f"{Path(config.MODEL_SAVE_DIR / config.LAST_MODEL_NAME)}")
    

if __name__ == "__main__":
    main()