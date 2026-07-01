from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])
train_path = "/content/drive/MyDrive/BrainTumorProject/Dataset/DATASET/classification/Training"

train_dataset = ImageFolder(
    root = train_path,
    transform = transform
)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True
)
