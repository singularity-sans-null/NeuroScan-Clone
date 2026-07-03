import torch

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from classifier import config
from classifier.dataset import get_data
from classifier.model import create_model
from classifier.utils import load_checkpoint


# Evaluation Function
def evaluate(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            
            _, predictions = torch.max(outputs, dim=1)
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            
            all_labels.extend(
                labels.numpy()
            )
    return all_labels, all_predictions

# Print Metrics

def print_metrics(labels, predicitions, class_names):
    
    accuracy = accuracy_score(labels, predicitions)
    
    precision = precision_score(
        labels,
        predicitions,
        average = "weighted"
    )
    
    recall = recall_score(
        labels,
        predicitions,
        average="weighted"
    )
    f1 = f1_score(
        labels,
        predicitions,
        average="weighted"
    )
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    print(f"Accuracy    :{accuracy*100:.2f}%")
    print(f"Precision   :{precision:.4f}")
    print(f"Recall      :{recall:.4f}")
    print(f"F1 Score    :{f1:.4f}")
    
    print("\nClassification Report\n")
    
    print(
        classification_report(
            labels,
            predicitions,
            target_names=class_names
        )
    )
    
    print("Confusion Matrix\n")
    print(
        confusion_matrix(
            labels,
            predicitions
        )
    )
    
    
    

def main():
    _, test_loader, class_names = get_data(
        dataset_root=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    model = create_model()
    
    model = load_checkpoint(
        model,
        config.MODEL_SAVE_DIR / config.BEST_MODEL_NAME
    )
    
    labels, predictions = evaluate(
        model,
        test_loader
    )
    
    print_metrics(
        labels,
        predictions,
        class_names
    )
    

if __name__ == "__main__":
    main()