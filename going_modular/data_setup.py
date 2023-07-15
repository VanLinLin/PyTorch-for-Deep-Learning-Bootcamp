"""Contains functionality for creating PyTorch DataLoader's for image classification data."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int):
    pass
