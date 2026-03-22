"""
Waste Classifier - CNN Utility Functions
Helper functions specific to the CNN classification pipeline.
"""

from typing import Dict, List

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_class_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute inverse-frequency class weights for handling imbalanced datasets.

    Args:
        class_counts: Dictionary mapping class name to sample count.

    Returns:
        Dictionary mapping class name to weight.
    """
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {}

    for cls_name, count in class_counts.items():
        if count > 0:
            weights[cls_name] = total / (num_classes * count)
        else:
            weights[cls_name] = 1.0

    logger.info("Class weights computed: %s", weights)
    return weights


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    Compute a confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes). 
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64) 
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> str:
    """
    Generate a text-based classification report with precision, recall, F1.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.

    Returns:
        Formatted report string.
    """
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    lines = []
    lines.append(f"{'Class':<40} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}") 
    lines.append("-" * 82)

    total_correct = 0
    total_samples = 0

    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp 
        fn = cm[i,:].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        lines.append(f"{name:<40} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}") 
        total_correct += tp
        total_samples += support

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0 
    lines.append("-" * 82) 
    lines.append(f"{'Accuracy':<40} {'':>10} {'':>10} {accuracy:>10.4f} {total_samples:>10d}")
    
    return "\n".join(lines)