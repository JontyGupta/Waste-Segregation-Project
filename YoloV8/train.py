"""
Waste Classifier - YOLOv8 Training Script
Handles training / fine-tuning YOLOv8 on a custom wste dataset.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class YOLOv8Trainer:
    """
    Training or fine-tunes a YOLOv8 model on a custom waste detection dataset.
    
    Expected dataset structure (YOLO format):
        data/
        |-- images/
        |       |-- train/
        |       |-- val/
        |-- labels/
                |-- train/
                |-- val/
    
    A dataset YAML file is required (see data/yolo_dataset.yaml)
    """

    def __init__(
        self,
        model_variant: str = "yolov8s-oiv7.pt",
        data_yaml: str = "data/YOLO_Dataset_V2/data.yaml",
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        patience: int = 20,
        save_dir: str = "models/yolov8",
        device: str = "auto",
        augment: bool = True,
    ) -> None:
        """
        Initialize YOLOv8Trainer.
        
        Args:
            model_variant: Base model to start from (e.g. 'yolov8n.pt').   
            data_yaml: Path to the dataset YAML configuration.
            epochs: Maximum training epochs.
            batch_size: Training batch size.
            img_size: Training image size (square).
            patience: Early-stopping patience (epochs with no improvement).
            save_dir: Directory to save trained weights.
            device: Compute device.
            augment: Whether to use augmentation during training.
        """
        self.model_variant = model_variant
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.device = device
        self.augment = augment

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """
        Train the YOLOv8 model.
        
        Returns:
            Dictionary with training results and best weights path.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required. Install via: pip install ultralytics"
            ) from e
        
        # Validate dataset YAML exists
        if not Path(self.data_yaml).exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {self.data_yaml}."
                "Create a YOLO-format dataset config first."
            )

        logger.info("=" * 60)
        logger.info("Starting YOLOv8 Training")
        logger.info("=" * 60)
        logger.info("Base model         : %s", self.model_variant)
        logger.info("Dataset YAML       : %s", self.data_yaml)
        logger.info("Epochs             : %s", self.epochs)
        logger.info("Batch size         : %s", self.batch_size)
        logger.info("Image size         : %s", self.img_size)
        logger.info("Save directory     : %s", self.save_dir)

        model = YOLO(self.model_variant)  # Load base model first
        # Then load your best previous weights if they exist
        best_weights_path = Path("runs/detect") / str(self.save_dir) / "train/weights/best.pt"
        if Path(best_weights_path).exists():
            model = YOLO(best_weights_path)  # ← Load previous trained model
            print(f"[INFO] Loaded previous YOLO weights from {best_weights_path}")

        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            patience=self.patience,
            project=str(self.save_dir),
            name="train",
            exist_ok=True,
            augment=self.augment,
            device=0 if self.device != "auto" else None,
            verbose=True,
        )

        # Locate best weights
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            logger.warning("best.pt not found as expected location: %s", best_weights)
            best_weights = None
        
        logger.info("Training complete. Best weights: %s", best_weights)

        return {
            "result": results,
            "best_weights": str(best_weights) if best_weights else None,
        }
    
    def validate(self, weights_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run validation on the trained model.
        
        Args:
            weights_path: Path to weights tp validate. Uses latest best.pt if None.
        
        Returns:
            Validation metrics dictionary
        """
        from ultralytics import YOLO

        if weights_path is None:
            weights_path = str(self.save_dir / "train" / "weights" / "best.pt")

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        logger.info("Running validation with weights: %s", weights_path)

        model = YOLO(weights_path)
        metrics = model.val(data=self.data_yaml)

        logger.info("Validation mAP50: %.4f | mAP50-95: %.4f", metrics.box.map50, metrics.box.map)
        return {"metrics": metrics}