"""Contains functionality for creating PyTorch DataLoader's for image classification data."""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_worker: int=os.cpu_count()):
    """Takes in a training directory and testing directory path and turns them into PyTorch DataSets and then into PyTorch DataLoaders.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (transforms.Compose): torchvision transforms to perform on training and testing data.
        batch_size (int): Number of samples per batch in each of the DataLoaders.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
        test_dir=path/to/test_dir,
        transform=some_transform,
        batch_size=32,
        num_worker=amount_of_cpu_in_your_device)
    """

    # Using ImageFolder to create image datasets (s)
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)
    
    # Get class names
    class_names = train_data.classes


    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_worker,
                                pin_memory=True)

    return train_dataloader, test_dataloader, class_names
