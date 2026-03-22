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

    def _collect_Samples(self) -> list:
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
            image - self.transform(image)

        return image, label
    
   
def get_transforms(
    img_size: int = 24,
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
            transforms.RandomRotation(degree=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, huw=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)).
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        

def create_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    class_names: Optional[list] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"
    
    # Create separate datasets with appropriate transforms
    train_dataset = WasteDataset(
        root_dir=dataset_root,
        transform=get_transforms(img_size, is_training=True),
        class_name=class_names,
    )
    val_dataset = WasteDataset(
        root_dir=dataset_root,
        transform=get_transforms(img_size, is_training=False),
        class_name=class_names,
    )

    total = len(train_dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size

    # split indices ( not datasets) so we can apply different transforms
    generator = torch.Generator().manual_seed(42)
    shuffled = torch.randperm(total, generator=generator).tolist()

    train_indices = shuffled[:train_size]
    val_indices = shuffled[train_size:train_size + val_size]
    test_indices = shuffled[train_size + val_size]

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    test_ds = Subset(val_dataset, test_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,
    )

    logger.info(
        "DataLoaders created - Train=%d, val=%d, test=%d",
        train_size, val_size, test_size,
    )
    return train_loader, val_loader, test_loader