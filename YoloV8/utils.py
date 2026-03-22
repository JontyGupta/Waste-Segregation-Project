"""
Waste Classifier - YOLOv8 Utility Functions
Helper functions specific to the YOLO detection pipeline.
"""

from pathlib import Path
from typing import Dict, List


def create_dataset_yaml(
    dataset_root: str,
    class_name: List[str],
    output_path: str = "data/yolo_dataset.yaml",
) -> str:
    """
    Generate a YOLO-format datasete YAML file.

    Args:
        dataset_root: Root directory of the dataset.
        class_name: List of class names.
        output_path: where to save the YAML file.

    Returns:
        Path to the generated YAML file.
    """
    import yaml
    
    data = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_name)},
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return str(output)


def convert_bbox_formats(
    bbox: List[float],
    img_width: int, 
    img_height: int,
    from_format: str = "xyxy",
    to_format: str = "xywh_norm",
) -> List[float]:
    """
    Convert bounding box between formats.

    Supported formats:
        - 'xyxy': [x1, y1, x2, y2] (pixel coords)
        - 'xywh': [x_center, y_center, width, height] (pixel coords)
        - 'xywh_norm': [x_center, y_center, width, height] (normalized 0-1)

    Args:
        bbox: Input bounding box.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        from_format: Source format.
        to_format: Target format.

    Returns:
        Converted bounding box.
    """
    # First convert to xyxy
    if from_format == "xyxy":
        x1, y1, x2, y2 = bbox
    elif from_format == "xywh":
        cx, cy, w, h = bbox
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
    elif from_format == "xywh_norm":
        cx, cy, w, h = bbox
        x1 = (cx - w / 2) * img_width
        y1 = (cx - h / 2) * img_height
        x2 = (cx + w / 2) * img_width
        y2 = (cx + h / 2) * img_height
    else:
        raise ValueError(f"Unknown from_foramt: {from_format}")
    
    # Convert from xyxy to target
    if to_format == "xyxy":
        return [x1, y1, x2, y2]
    elif to_format == "xywh":
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return [cx, cy, w, h]
    elif to_format == "xywh_norm":
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        cx = (x1 + (x2 - x1) / 2) / img_width
        cy = (y1 + (y2 - y1) / 2) / img_height
        return [cx, cy, w, h]
    else:
        raise ValueError(f"Unknown to_foramt: {to_format}")
