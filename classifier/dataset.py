from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from classifier import config

# This function does the Image transformation
def get_transforms():
    """
    Returns the preprocessing pipeline used for EfficientNet.
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = config.IMAGE_MEAN,
            std = config.IMAGE_STD
        )
    ])
    
    return transform


def load_datasets(dataset_root):
    """
    Loads the training and testing datasets.

    Parameters
    ----------
    dataset_root : str or Path
        Root dataset directory.

    Returns
    -------
    train_dataset
    test_dataset
    """
    dataset_root = Path(dataset_root)
    
    train_dataset = ImageFolder(
        root = f"{dataset_root}/Training",
        transform= get_transforms()
    )
    
    test_dataset = ImageFolder(
        root = f"{dataset_root}/Testing",
        transform = get_transforms()
    )
    return train_dataset, test_dataset

def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_Size = 32,
    num_workers = 2
):
    """
    Creates Pytorch DataLoaders
    
    Returns
    ------
    train_loader, test_loader
    """
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_Size,
        shuffle= True,
        num_workers= num_workers
    )
    
    test_loader = DataLoader(
        dataset= test_dataset,
        batch_size=batch_Size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_class_names(dataset):
    """
    Returns the class names of the dataset
    """
    return dataset.classes

def get_data(
    dataset_root,
    batch_size = 32,
    num_workers = 2
):
    """
    Loads everything needed for training.

    Returns
    -------
    train_loader
    test_loader
    class_names
    """
    train_dataset, test_dataset = load_datasets(dataset_root)
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_Size= batch_size,
        num_workers=num_workers
    )
    class_names = get_class_names(train_dataset)
    return train_loader, test_loader, class_names




