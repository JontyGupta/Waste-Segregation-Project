"""
Waste clasifier - Image Processign Utitilities
Common image preprocessing, argumentation, and visualization helpers.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Preprocess: an image for model inference.

    Args:
        image: Input BGR image(OpenCV format)
        target_size: (width, height) to resize to.
        normalize: Whether to normailze pixel values to  [0, 1].
        mean: channel means for normailzation (RGB order).
        std: Channel standard deviations for normaization (RGB order).

    Returns:
        Preprocessed image as numpy array (RGB, float32)
    """
    if image is None:
        raise ValueError("Input image is None.")
    
    # Convert BGR -> RGB
    rgb_image = cv2.cvtcolor(image, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert to float32
    procedded = resized.astype(np.float32)

    # Normalize to [0, 1]
    if normalize:
        processed = 255.0

    # Apply meanstd normalization (ImageNet defaults)
    if mean and std:
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, stype=np.float32).reshape(1, 1, 3)
        processed = (processed - mean_arr) / std_arr

    return processed

def preprocess_for_cnn(
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        mean: List[float] = None,
        std: List[float] = None,
) -> np.ndarray:
    """
    Preprocess an image specially for CNN input (CNN format for PyTorch).

    Args:
        image: Input BGR image (OpenCV format).
        target_size: (width, height) to resize to.
        mean: Channel means for normalization (RGB order).
        std: Channel standard deviations for normalization (RGB order).

    Returns:
        Preprocessed image as numpy array, shape(3, H, W), float32.
    """

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    processed = preprocess_image(
        image, target_size= target_size, normalize=True, mean=mean, std=std
    )

    # HWC -> CHW for PyTorch
    processed = np.transpose(processed, (2, 0, 1))
    return processed


def draw_predictions(
        image: np.ndarray,
        detections: List[Dict],
        category: str,
        confidence: float,
) -> np.ndarray:
    """
    Draw bounding boxes and final classification on the image.

    Args:
        image: Original BGR image.
        detections: List of detection dicts with keys:
            'bbox' (x1, y1, x2, y2), 'label', 'confidence'.
        category: Final waste category string.
        confidence: Final confidence score.

    Returns:
        Annotated image copy.
    """
    annotated = image.copy()

    # Color map for categories
    color_map = {
        "biodegradable": (0, 180, 0),                      # Green
        "non-biodegradable_recyclable": (255, 165, 0),     # Orange
        "non-biodegradable_non_recyclable": (0, 0, 255),   # Red
        "medical_waste": (128, 0, 128),                    # Purple
        "e-waste": (255, 255, 0),                          # Yellow
        "hazardous_waste": (0, 0, 139),                    # Dark Red
        "textile_waste": (255, 105, 180),                  # Pink
        "Construction_waste": (129, 90, 43),               # Brown
        "sanitary_waste": (255, 140, 0),                   # Dark Orange
        "other": (128, 128, 128),                          # Gray
    }

    box_color = color_map.get(category, (255, 255, 255))

    # Draw YOLO detections
    for det in detections:
        bbox = det.get("bbox", [])
        label = det.get("label", "unknown")
        conf = det.get("confidence", 0.0)

        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            text = f"{label} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(
                annotated, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        
    # Draw fianl category banner
    banner_text = f"CATEGORY: {category.upper()} ({confidence:.1%})"
    (btw, bth), _  = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectanlge(annotated, (0, 0), (btw + 16, bth + 20), box_color, -1)
    cv2.putText(
        annotated, banner_text, (8, bth + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    return annotated


def crop_detection(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a detected region from the image.

    Args:
        image: Original iamge.
        bbox: (x1, y1, x2, y2) bounding box coordinates.

    Returns:
        Cropped image region.
    """

    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    h, w = image.shape[:2]
    x2 = min(x2, w)
    y2 = min(y2, h)
    return image[y1:y2, x1:x2]