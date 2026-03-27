"""
Waste Classifier - CNN Dataset & DataLoader
Handles loading, augmentation, and batching of waste images for CNN training.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils.logger import get_logger

logger = get_logger(__name__)


class WasteDataset(Dataset):
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str,
        transform=None,
        class_names: Optional[list] = None,
    ) -> None: 
        """
        
        
        
        
        
        
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Dataset root not found: {self.root_dir}")
        
        # Discover classes from subdirectories
        if class_names is not None: 
            self.class_names = class_names
        else:
            self.class_names = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Collect all image paths and labels
        self.samples = self._collect_samples()
        logger.info(
            "WasteDataset: %d images, %d classes from '%s'",
            len(self.samples), len(self.class_names), self.root_dir,
        )

    def _collect_samples(self) -> list:
        """Scan directories and collect (image_path, label) pairs."""
        samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Class directory not found: %s", class_dir)
                continue

            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    samples.append((str(img_path), label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        
        
        
        
        
        
        
        """
        from PIL import Image

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
   
def get_transforms(
    img_size: int = 224,
    is_training: bool = True,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
):
    """
    
    
    
    
    
    
    
    
    
    
    """
    from torchvision import transforms

    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        

def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    class_names: Optional[list] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset = WasteDataset(
        root_dir=train_dir,
        transform=get_transforms(img_size, is_training=True),
        class_names=class_names,
    )
    val_dataset = WasteDataset(
        root_dir=val_dir,
        transform=get_transforms(img_size, is_training=False),
        class_names=class_names,
    )
    test_dataset = WasteDataset(
        root_dir=test_dir,
        transform=get_transforms(img_size, is_training=False),
        class_names=class_names,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders created - Train=%d, Val=%d, Test=%d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )
    return train_loader, val_loader, test_loader