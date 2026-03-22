"""


"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from CNN.model import WasteCNN, WASTE_CATEGORIES
from utils.image_processing import preprocess_for_cnn
from utils.logger import get_logger

logger = get_logger(__name__)


class CNNPredictor:
    """
    
    
    
    
    """

    def __init__(
        self,
        weights_path: str,
        architecture: str = "resnet50",
        num_classes: int = 10,
        device: str = "auto",
    ) -> None:
        """
        
        
        
        
        
        
        
        """
        self.weights_path = weights_path
        
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build and load model
        self.model = WasteCNN(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=False,
            dropout=0.0,
        )

        self.load_weights()
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_weights(self) -> None:
        """Load trained weights."""
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"CNN weights not found: {self.weights_path}."
                f"Train the CNN model first."
            )
        self.model.load_weights(self.weights_path, device=str(self.device))
        logger.info("CNN model loaded from %s on %s", self.weights_path, self.device)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        top_k: int = 3,
    )-> Dict[str, Any]:
        """
        Classify a single BGR image into a waste category.

        Args:
            image: BGR image (OpenCV format), e.g., a cropped detection.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary containing:
              - 'category': predicted category name (str)
              - 'confidence': confidence score (float)
              - 'class_id': predicted class index (int)
              - 'probabilities': dict of category -> probability
              - 'top_k': list of (category, probability) tuples
        """
        # Preprocess
        processed = preprocess_for_cnn(image)
        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)

        # Inference
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Parse results
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        category = WASTE_CATEGORIES[class_id] if class_id < len(WASTE_CATEGORIES) else "unknown"

        #All probabilities
        probabilities = {
            WASTE_CATEGORIES[i]: float(probs[i]) 
            for i in range(min(len(probs), len(WASTE_CATEGORIES))) 
        }

        # Top-K
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            (WASTE_CATEGORIES[i], float(probs[i])) 
            for i in top_indices 
            if i < len(WASTE_CATEGORIES)
        ]

        return {
            "category": category,
            "confidence": confidence,
            "class_id": class_id,
            "probabilities": probabilities,
            "top_k": top_k_results,
        }
    
    @torch.no_grad()
    def predict_batch(
        self, images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of images.

        Args:
            images: List of BGR images.

        Returns:
            List of prediction dictionaries (same format as predict()).
        """
        if not images:
            return []
        
        # Preprocess all
        tensors = []
        for img in images:
            processed = preprocess_for_cnn(img)
            tensors.append(torch.from_numpy(processed))

        batch = torch.stack(tensors).to(self.device)
        logits = self.model(batch)
        probs_batch = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for probs in probs_batch:
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            category = WASTE_CATEGORIES[class_id] if class_id < len(WASTE_CATEGORIES) else "unknown"

            probabilities = {
                WASTE_CATEGORIES [i]: float(probs[i])
                for i in range(min(len(probs), len(WASTE_CATEGORIES)))
            }

            results.append({
                "category": category,
                "confidence": confidence,
                "class_id": class_id,
                "probabilities": probabilities,
            })

        return results