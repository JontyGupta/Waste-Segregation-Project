"""


"""


from typing import Optional

import torch 
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)

# Waste categories the CNN classifies into
WASTE_CATEGORIES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


class WasteCNN(nn.Module):
    """
    
    
    
    
    
    
    
    
    """

    def __init__(
        self, 
        architecture: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        """
        
        
        
        
        
        
        
        """
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        self.backbone, in_features = self._build_backbone(architecture, pretrained)

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        logger.info(
            "WasteCNN initialized - backbone=%s, classes=%d, pretrained=%s",
            architecture, num_classes, pretrained,
        )

    def _build_backbone(self, architecture: str, pretrained: bool):
        """
        
        
        
        
        """
        try:
            import torchvision.models as models
        except ImportError as e : 
            raise ImportError(
                "torchvision is required. Install via : pip install torchvision"
            ) from e
        
        weights = "IMAGENET1K_V1" if pretrained else None

        if architecture == "resnet18":
            base = models.resent18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()  # Remove original FC

        elif architecture == "resnet50":
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity() 

        elif architecture == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity() 

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}."
                f"Choose from: resnet18, resnet50, efficientnet_b0"
            )

        return base, in_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        
        
        
        
        
        
        """
        features =self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def freeze_backbone(self) -> None: 
        """Freeze backbone parameters (for fine-tuning only the head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen - only classifier head will be trained.")
    
    def unfreeze_backbone(self) -> None: 
        """Unfreezeall backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen - all parameters are trainable.")

    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""    
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Model weights saved to %s", path)
    
    def load_weights(self, path: str, device: str = "cpu") -> None:
        """Load model weights from disk."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info("Model weights loaded from %s", path)

    @staticmethod
    def get_category_name(index: int) -> str:
        """Get category name from class index."""
        if 0 <= index < len(WASTE_CATEGORIES):
            return WASTE_CATEGORIES[index]
        return "unknown"
    
    @staticmethod
    def get_category_index(name: str) -> int: 
        """Get class index from category name."""
        try:
            return WASTE_CATEGORIES.index(name)
        except ValueError:
            return -1