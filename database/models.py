"""
Waste Classifier
Defines the WasteRecord dataclass used across all storage backends.
"""

from dataclasses import dataclass, field, asdict 
from datetime import datetime 
from typing import Any, Dict, List, Optional


@dataclass
class WasteRecord:
    """
    A single waste classification record stored in the database.

    Fields:
        record_id          : Unique identifier (auto-generated if not provided).
        timestamp          : When the classification was performed.
        image_path         : Path to the original image.
        final_category     : Final ensemble waste category.
        final_confidence   : Final ensemble confidence score (0.1.0).
        strategy           : Ensemble strategy used.
        yolo_detections    : List of YOLO detection dicts (label, confidence, bbox).
        cnn_predictions    : List of CNN prediction dicts (category, confidence, probs).
        all_probabilities  : Full category probability distribution from ensemble.
        source             : Image source ('camera', 'file', 'batch').
        notes              : Optional user notes or metadata.
    """

    image_path: str
    final_category: str
    final_confidence: float
    strategy: str = "confidence_adaptive" 
    yolo_detections: List[Dict[str, Any]] = field(default_factory=list) 
    cnn_predictions: List[Dict[str, Any]] = field(default_factory=list)
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    sources: str = "file"
    notes: str = ""
    record_id: Optional[str] = None
    timestamp: Optional [str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp= datetime.now().isoformat()
        if self.record_id is None:
            import uuid
            self.record_id = str(uuid.uuid4())
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a plain dictionary."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to a flat dictionary suitable for tabular export (CSV/Excel).
        Nested fields are serialized to JSON strings.
        """
        import json

        flat = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "final_category": self.final_category,
            "final_confidence": round(self.final_confidence, 4),
            "strategy": self.strategy,
            "source": self.source,
            "notes": self.notes,
            "num_yolo_detections": len(self.yolo_detections),
            "num_cnn_predictions": len(self.cnn_predictions),
        }

        # Add individual category probabilities as columns (great for PowerBI)
        for cat, prob in self.all_probabilities.items(): 
            flat[f"prob_{cat}"]= round(prob, 4)

        # Top YOLO detection
        if self.yolo_detections:
            top_det = self.yolo_detections[0]
            flat["top_yolo_label"] = top_det.get("label", "")
            flat["top_yolo_confidence"] = round(top_det.get("confidence", 0.0), 4)
        else:
            flat["top_yolo_label"] = ""
            flat["top_yolo_confidence"] = 0.0

        # Top CNN prediction
        if self.cnn_predictions:
            top_cnn = self.cnn_predictions[0]
            flat["top_cnn_category"] = top_cnn.get("category", "")
            flat["top_cnn_confidence"] = round(top_cnn.get("confidence", 0.0), 4)
        else:
            flat["top_cnn_category"] = ""
            flat["top_cnn_confidence"] = 0.0

        # Full nested data as JSON strings (for detailed analysis)
        flat["yolo_detections_json"] = json.dumps(self.yolo_detections)
        flat["con_predictions_json"] = json.dumps(self.cnn_predictions)

        return flat
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WasteRecord":
        """Create a WasteRecord from a dictionary."""
        return cls(
            record_id=data.get("record_id"),
            timestamp=data.get("timestamp"),
            image_path=data.get("image_path", ""),
            final_category=data.get("final_category", "other"),
            final_confidence=data.get("final_confidence", 0.0),
            strategy=data.get("strategy", ""),
            yolo_detections=data.get("yolo_detections", []),
            cnn_predictions=data.get("cnn_predictions", []),
            all_probabilities=data.get("all_probabilities", {}),
            source=data.get("source", "file"),
            notes=data.get("notes", ""),
        )