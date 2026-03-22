"""
Waste Classifier - YOLOv8 Model Wrapper
Wraps Ultalytics YOLOv8 for waste object detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class YOLOv8Dectector:
    """
    YOLOv8-based waste object detector.

    Responsibilities:
        - Load a pre-trained or custom-trained YOLOv8 model.
        - Run inference on images to detect waste objects.
        - Return structured detection results (counding boxes, labels, scores).
    """

    def __init__(
        self, 
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ) -> None:
        """
        Initialize YOLOv8Detector.

        Args:
            model_path: Path to YOLOv8 weights (.pt file) or model variant name.
            confidence_threshold: Minimum Confidence to keep a detection
            iou_threshold: IoU threshold for Non-Max Suppression.
            device: Compute device ('auto', 'cpu', 'cuda', 'cuda:0'). 
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
    
    def load_model(self) -> None:
        """Load the YOLOv8 model into memory."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required. Install via: pip install ultralytics"
            ) from e
        
        logger.info("Loading YOLOv8 model form '%s'...", self.model_path)

        if Path(self.model_path).exists():
            self._model = YOLO(self.model_path)
        else:
            # Fall back to pretrained variant name (e.g., "yolov8n.pt")
            logger.warning(
                "Custom weights not found at '%s'. Loading pretrained variant.",
                self.model_path,
            )
            self._model = YOLO(self.model_path)

        # Move to specified device 
        if self.device != "auto":
            self._model.to(self.device)

        logger.info("YOLOv8 model loaded successfully. Classes: %s", self._model.names)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 detection on a single image.

        Args:
            image: BGR image as numpy array (OpenCV format).

        Returns:
            List of detection dictionaries, each containing:
            - 'bbox': [x1, y1, x2, y2] (pixel coordinates)
            - 'label': str (class name)
            - 'class_id': int (class index)
            - 'confindence':float (detetion score)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = self._parse_results(results)
        logger.info("YOLOv8 detected %d object(s).", len(detections))
        return detections
    
    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """
        Parse raw Ultralytics results into a clean list of dicts.

        Args:
            results: ultralytics Results object.

            Returns:
                List of detection dictionaries.
        """
        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                label = self._model.names.get(class_id, f"class_{class_id}")

                detections.append(
                    {
                        "bbox": bbox,
                        "label": label,
                        "class_id": class_id,
                        "confidence": confidence
                    }
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
    
    def get_class_names(self) -> Dict[int, str]:
        """Return the mapping of class IDs to class names."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return dict(self._model.names)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
        

