"""
train.py
-----
Trains the EfficientNet-B0 brain tumor classifier
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from classifier import config
from classifier.dataset import get_data
from classifier.model import create_model
from classifier.utils import (
    set_seed,
    save_checkpoint,
    calculate_accuracy,
    AverageMeter,
    print_model_summary
)


# Training function

def train_one_epoch(model, loader, criterion, optimizer):
    """
    Train the model for one epoch.
    """
    model.train()
    
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    
    for images,labels in loader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        batch_accuracy = calculate_accuracy(outputs, labels)
        
        loss_meter.update(loss.item(), images.size(0))
        accuracy_meter.update(batch_accuracy, images.size(0))
        
    return loss_meter.avg, accuracy_meter.avg


# Validation function

def validate(model, loader, criterion):
    """
    Evaluate the model on the test dataset
    """
    
    model.eval()
    
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    
    with torch.no_grad():
        
        for images, labels in loader:
            
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            ouputs = model(images)
            
            loss = criterion(ouputs, labels)
            
            batch_accuracy = calculate_accuracy(ouputs, labels)
            loss_meter.update(loss.item(), images.size(0))
            accuracy_meter.update(batch_accuracy, images.size(0))
            
    return loss_meter.avg, accuracy_meter.avg


# The training loop

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer):
    best_accuracy = 0.0
    
    best_weights = copy.deepcopy(model.state_dict())
    
    patience = 3
    # Number of epcoh allowance for model improvement
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
        
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )
        
        print("="*60)
        print(f"Epoch[{epoch+1}/{config.NUM_EPOCHS}]")
        print(f"Train Loss          :{train_loss:.4f}")
        print(f"Train Accuracy      :{train_acc*100:.2f}%")
        print(f"Validation Loss     :{val_loss:.4f}")
        print(f"Validation Acc.     :{val_acc*100:.2f}%")
        
        print("="*60)
        
        # save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_weights = copy.deepcopy(
                model.state_dict()
            )
            save_checkpoint(
                model,
                config.BEST_MODEL_NAME
            )
            
            print("Best model updated.\n")
            patience_counter = 0
            
        else:
            patience_counter += 1
            
            print(f"No improvement({patience_counter}/{patience})\n")
            
        # Stop once patience is exceeded
        if patience_counter >= patience:
            print("Early stopping triggered. Patience exceeded!")
            break
    
    model.load_state_dict(best_weights)
    
    save_checkpoint(
        model,
        config.LAST_MODEL_NAME
    )
    
    print(f"\nBest Validation Accuracy: {best_accuracy*100:.2f}%")
    
    return model


def main():
    # Set the random seed
    set_seed()
    # Load dataset
    
    train_loader, test_loader, class_names = get_data(
        dataset_root=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    print("Classes:")
    print(class_names)
    print()
    
    # create the model
    
    model = create_model()
    print_model_summary(model)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr = config.LEARNING_RATE
    )
    
    # Train
    
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer
    )
    
    print("\nTraining Complete!")
    
    

if __name__ == "__main__":
    main()