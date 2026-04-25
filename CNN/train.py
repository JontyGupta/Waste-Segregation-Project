"""
Waste Classifier - CNN Training Script
Handles end-to-end training of the CNN waste classifier with 
early stopping, learning rate scheduling, and checkpointing.
"""

import time
from pathlib import Path 
from typing import Any, Dict, Optional, Tuple

import torch
import shutil
import torch.nn as nn 
from torch.optim import Adam 
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import DataLoader

from CNN.model import WasteCNN, WASTE_CATEGORIES 
from CNN.dataset import create_dataloaders 
from utils.logger import get_logger

logger = get_logger(__name__)


class CNNTrainer:
    """
    Trains the WasteCNN model with best-practice techniques:
      - Transfer learning with backbone freezing/unfreezing
      - Learning rate scheduling
      - Early stopping
      - Model checkpointing
      - Train/val metric tracking
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        scheduler_step: int = 10,
        scheduler_gamma: float = 0.1,
        epochs: int = 50,
        patience: int = 15,
        save_dir: str = "models/cnn",
        device: str = "auto",
    )-> None:
        """
        Initialize CNNTrainer.

        Args:
            architecture: Backbone architecture.
            num_classes: Number of waste categories.
            pretrained: Use pretrained backbone.
            dropout: Dropout rate.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization.
            scheduler_step: LR scheduler step size (epochs).
            scheduler_gamma: LR decay factor.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            save_dir: Directory to save checkpoints.
            device: Compute device ('auto', 'cpu', 'cuda').
        """
        self.epochs = epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)

        # Build model
        self.model = WasteCNN(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout, 
        ).to(self.device)

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate, 
            weight_decay=weight_decay,
        ) 
        self.scheduler = StepLR(
            self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )

        #Tracking
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        freeze_backbone_epochs: int = 5,
    )-> Dict[str, Any]:
        """
        Full training loop with early stopping and LR scheduling.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            freeze_backbone_epochs: Number of initial epochs with frozen backbone.

        Returns:
            Dictionary with training history and best metrics.
        """
        logger.info("="*60)
        logger.info("Starting CNN Training")
        logger.info("=" * 60)
        logger.info("Device            : %s", self.device)
        logger.info("Architecture      : %s", self.model.architecture)
        logger.info("Max epochs        : %d", self.epochs)
        logger.info("Patience          : %d", self.patience)
        logger.info("Freeze epochs     : %d", freeze_backbone_epochs)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_no_improve = 0
        best_epoch = 0
         
        # Phase 1: Freeze backbone
        if freeze_backbone_epochs > 0:
            self.model.freeze_backbone()

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # Unfreeze backbone after initial phase
            if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
                self.model.unfreeze_backbone()
                logger.info("Unfreezing backbone at epoch %d", epoch)

            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Step LR scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Record history
            self.history["train_loss"].append(train_loss) 
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %3d/%d | Train Loss: %.4f Acc: %.2f%% | "
                "Val Loss: %.4f Acc: %.2f%% | LR: %.6f | Time: %.1fs", 
                epoch, self.epochs, train_loss, train_acc * 100, 
                val_loss, val_acc * 100, 
                self.optimizer.param_groups[0]["lr"], elapsed,
            ) 

            #Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0

                # Save best model
                best_path = self.save_dir / "best_cnn.pth"
                self.model.save_weights(str(best_path))
                logger.info(" -> New best model saved (val_loss=%.4f)", val_loss)
            else:
                epochs_no_improve += 1
                 
            # Early stopping
            if epochs_no_improve >= self.patience:
                logger.info( 
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, self.patience, 
                )
    
                break

        # Save final model
        best_path = self.save_dir / "best_cnn.pth"
        final_path = self.save_dir / "final_cnn.pth"
        if best_path.exists():
            shutil.copy(best_path, final_path)
            logger.info("Final model saved (copy of best epoch %d)", best_epoch)
        else:
            self.model.save_weights(str(final_path))

        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info("Best epoch: %d | Val Loss: %.4f | Val Acc: %.2f%%", 
                    best_epoch, best_val_loss, best_val_acc * 100)
        logger.info("-" * 60)

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "history": self.history,
        }
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Tuple [float, float]:
        """Run a single validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss/ total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc