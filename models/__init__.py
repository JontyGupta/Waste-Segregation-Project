"""
Models package - contains YOLOv8 and CNN sub-modules.

Sub-packages:
    models.Yolov8  - YOLOv8 objects detection (detector, trainer, predictor)
    models.CNN     - CNN waste category classifier (model, trainer, predictor)
"""

from models.YoloV8 import YOLOv8Detector, YOLOv8Trainer, YOLOv8Predictor
from models.CNN import WasteCNN, CNNTrainer, CNNPredictor

__all__ = [
    "YOLOv8Detector",
    "YOLOv8Trainer",
    "YOLOv8Predictor",
    "WasteCNN",
    "CNNTrainer",
    "CNNPredictor",
]