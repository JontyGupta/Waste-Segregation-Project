"""
Waste Classifier - YOLOv8 Prediction / Inference Pipeline
High-level interface for running YOLOv8 on single images or batches
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from YoloV8.model import YOLOv8Dectector
from utils.logger import get_logger

logger = get_logger(__name__)


class Yolov8Predictor:
    """
    High-level prediction wrapper for YOLOv8 waste detection.

    Provides methods for:
        - Single image prediction
        - Batch prediciton on a directory
        - Extracting cropped detection for downstream CNN
    """

    def __init__(self, detector: YOLOv8Dectector) -> None:
        """
        Initialize YOLOv8Predictor

        Args:
            detector: An initialized YOLOv8Detector instance.
        """
        self.detector = detector
        if not self.detector.is_loaded:
            self.detector.load_model()

    def predict_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        
        
        
        
        
        
        
        """
        return self.detector.detect(image)
    
    def predict_file(self, image_path: str) -> List[Dict[str, Any]]:
        """
        
        
        
        
        
        
        
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.detector.detect(image)
    
    def predict_directory(
        self,
        dir_path: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        
        
        
        
        
        
        
        
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        results: Dict[str, List[Dict[str, Any]]] = {}

        image_files = [
            f for f in sorted(dir_path.iterdir())
            if f.suffix.lower() in extensions
        ]

        logger.info("Running YOLOv8 on %d images in '%s'...", len(image_files), dir_path)

        for img_file in image_files:
            try:
                detections = self.predict_file(str(img_file))
                results[img_file.name] = detections
            except Exception as e:
                logger.error("Error processing %s: %s", img_file.name, e)
                results[img_file.name] = []

        return results
    
    def get_cropped_detections(
            self, 
            image: np.ndarray,
            detections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """








        """
        if detections is None:
            detections = self.detector.detect(image)

        cropped: List[Dict[str, Any]] = []
        h, w = image.shape[:2]

        for det in detections:
            bbox = det["bbox"]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning("Empty crop for detection: %s", det)
                continue

            cropped.append(
                {
                    "crop": crop,
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                }
            )

        logger.info("Extracted %d cropped detections.", len(cropped))
        return cropped
    
    