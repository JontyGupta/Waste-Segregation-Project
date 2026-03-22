"""CNN Waste classification module."""

from CNN.model import WasteCNN
from CNN.train import CNNTrainer
from CNN.predict import CNNPredictor

__all__ = ["WasteCNN", "CNNTrainer", "CNNPredictor"]