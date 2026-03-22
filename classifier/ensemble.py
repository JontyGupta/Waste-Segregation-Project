"""
Waste Classifier - Ensemble Classifier


"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from CNN.model import WASTE_CATEGORIES
from utils.logger import get_logger

logger = get_logger(__name__)


class WasteEnsembleClassifier:
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    """

    def __init__(
        self,
        object_category_map: Dict[str, str],
        strategy: str = "confidence_adaptive",
        yolo_weight: float = 0.55,
        cnn_weight: float = 0.45,
        min_confidence: float = 0.3,
        adaptive_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        
        
        
        
        
        
        
        
        
        """
        self.object_category_map = object_category_map
        self.strategy = strategy
        self.yolo_weight = yolo_weight
        self.cnn_weight = cnn_weight
        self.min_confidence = min_confidence

        # Confidence-adaptive defaults
        self.adaptive = adaptive_config or {
            "high_conf_threshold": 0.85,
            "high_yolo_weight": 0.70,
            "high_cnn_weight": 0.30,
            "medium_yolo_weight": 0.55,
            "medium_cnn_weight": 0.45,
            "low_yolo_weight": 0.30,
            "low_cnn_weight": 0.70,
        }

        assert abs(yolo_weight + cnn_weight - 1.0) < 1e-6, \
        "yolo_weight + cnn_weight must equal 1.0"

        logger.info(
            "Ensemble initialized - strategy=%s, yolo_w=%.2f, cnn_w=%.2f",
            strategy, yolo_weight, cnn_weight,
        )

    def classify(
        self,
        yolo_detections: List[Dict[str, Any]],
        cnn_predictions: List[Dict[str, Any]],   
    ) -> Dict[str, Any]: 
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        """
        if not yolo_detections and not cnn_predictions:
            return self._default_result("No detections from either models.")
        
        if self.strategy == "confidence_adaptive":
            return self._confidence_adaptive(yolo_detections, cnn_predictions)
        elif self.strategy == "weighted_average":
            return self._weighted_average(yolo_detections, cnn_predictions)
        elif self.strategy == "majority_vote":
            return self._majority_vote(yolo_detections, cnn_predictions)
        elif self.strategy == "yolo_prioirty":
            return self._yolo_prioirty(yolo_detections, cnn_predictions)
        else:
            logger.warning("Unknown strategy '%s'. Falling back to confidence_adaptive.", self.strategy)
            return self._confidence_adaptive(yolo_detections, cnn_predictions)
        
    #--------------------------------------------------------------------------#
    # Fusion strategies
    #--------------------------------------------------------------------------#

    def _confidence_adaptive(
        self, 
        yolo_dets: List[Dict[str, Any]],
        cnn_preds: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
        """
        combined_probs = {cat: 0.0 for cat in WASTE_CATEGORIES}

        if not yolo_dets and not cnn_preds:
            return self._default_result("No items to classify.")
        
        high_thresh = self.adaptive["high_conf_threshold"]

        # Determine per-detection adaptive weights, then accumulate
        for i, det in enumerate(yolo_dets):
            yolo_conf = det.get("confidence", 0.0)

            # Select weight tier based on YOLO confidence
            if yolo_conf >= high_thresh:
                yw = self.adaptive["high_yolo_weight"]
                cw = self.adaptive["high_cnn_weight"]
                tier = "high"
            elif yolo_conf >= self.min_confidence:
                yw = self.adaptive["medium_yolo_weight"]
                cw = self.adaptive["medium_cnn_weight"]
                tier = "medium"
            else:
                yw = self.adaptive["low_yolo_weight"]
                cw = self.adaptive["low_cnn_weight"]
                tier = "low"

            # YOLO contribution: map label -> category (hard assignment)
            label = det.get("label", "")
            mapped_cat = self.object_category_map.get(label, "other")
            combined_probs[mapped_cat] += yw * yolo_conf

            # CNN contribution (if available for this detection)
            if i < len(cnn_preds):
                cnn_prob = cnn_preds[i].get("probabilities", {})
                for cat in WASTE_CATEGORIES:
                    combined_probs[cat] += cw * cnn_prob.get(cat, 0.0)

            logger.debug(
                "  Det %d: YOLO=%s (%.2f, tier=%s) -> yw=%.2f cw=%.2f",
                i, label, yolo_conf, tier, yw, cw,
            )

        # Handle any extra CNN predictions (no matching YOLO detection)
        for pred in cnn_preds[len(yolo_dets):]:
            cnn_prob = pred.get("probabilities", {})
            for cat in WASTE_CATEGORIES:
                combined_probs[cat] += self.adaptive["low_cnn_weight"] * cnn_prob.get(cat, 0.0)

        # Normalize 
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}

        final_cat = max(combined_probs, key=combined_probs.get)
        final_conf = combined_probs[final_cat]

        return {
            "category": final_cat,
            "confidence": final_conf,
            "strategy": "confidence_adaptive",
            "all_probabilities": combined_probs,
            "details": {
                "yolo_detections": len(yolo_dets),
                "cnn_predictions": len(cnn_preds),
                "adaptive_thresholds": self.adaptive,
            },
        }
    
    def _weighted_average(
        self, 
        yolo_dets: List[Dict[str, Any]],
        cnn_preds: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        
        
        
        
        """
        # Initialize probability accumulator
        combined_probs = {cat: 0.0 for cat in WASTE_CATEGORIES}
        num_items = max(len(yolo_dets), len(cnn_preds))

        if num_items == 0:
            return self._default_result("No items to classify.")
        
        # Accumulate YOLO contributions
        for det in yolo_dets:
            label = det.get("label", "")
            conf = det.get("confidence", 0.0)
            mapped_cat = self.object_category_map.get(label, "other")

            # Hard assignment weighted by detection confidence
            combined_probs[mapped_cat] += self.yolo_weight * conf

        # Accumulate CNN contributions
        for pred in cnn_preds:
            probs = pred.get("probabilities", {})
            for cat in WASTE_CATEGORIES:
                combined_probs[cat] += self.cnn_weight * probs.get(cat, 0.0)

        # Normalize
        total = sum(combined_probs.values())
        if total > 0: 
            combined_probs = {k: v / total for k, v in combined_probs.items()}

        # Pick best
        final_cat = max(combined_probs, key=combined_probs.get)
        final_conf = combined_probs[final_cat]

        return {
            "category": final_cat,
            "confidence": final_conf,
            "strategy": "confidence_adaptive",
            "all_probabilities": combined_probs,
            "details": {
                "yolo_detections": len(yolo_dets),
                "cnn_predictions": len(cnn_preds),
            },
        }
    
    def _majority_vote(
        self, 
        yolo_dets: List[Dict[str, Any]],
        cnn_preds: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        
        """
        votes: Dict[str, float] = {cat: 0.0 for cat in WASTE_CATEGORIES}

        # YOLO votes
        for det in yolo_dets:
            label = det.get("label", "")
            conf = det.get("confidence", 0.0)
            mapped_cat = self.object_category_map.get(label, "other")
            votes[mapped_cat] += conf

        # CNN votes
        for pred in cnn_preds:
            cat = pred.get("category", "other")
            conf = pred.get("confidence", 0.0)
            if cat in votes:
                votes[cat] += conf

        # Pick Winner
        final_cat = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        final_conf = votes[final_cat] / total_votes if total_votes > 0 else 0.0

        return {
            "category": final_cat,
            "confidence": final_conf,
            "strategy": "majority_vote",
            "all_probabilities": {k: v / total_votes if total_votes > 0 else 0 for k, v in votes.items()},
            "details": {"votes": votes},
        }
    
    def _yolo_priority(
        self, 
        yolo_dets: List[Dict[str, Any]],
        cnn_preds: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        
        
        """
        # Use highest-confidence YOLO detection
        if yolo_dets:
            best_det = max(yolo_dets, key=lambda d : d.get("confidence", 0.0))
            label = best_det.get("label", "")
            confidence = best_det.get("confidence", 0.0)
            mapped_cat = self.object_category_map.get(label, "other")

            if confidence >= self.min_confidence:
                return {
                    "category": mapped_cat,
                    "confidence": confidence,
                    "strategy": "yolo_priority",
                    "source": "yolo",
                    "details": {"yolo_label": label, "yolo_confidence": confidence},
                }
            
        # Fall back to CNN
        if cnn_preds:
            best_cnn = max(cnn_preds, key=lambda p: p.get("confidence", 0.0))
            return {
                "category": best_cnn.get("category", "other"),
                "confidence": best_cnn.get("confidence", 0.0),
                "strategy": "yolo_priority",
                "source": "cnn_fallback",
                "details": {"cnn_probabilities": best_cnn.get("probabilites", {})},
            }
        
        return self._default_result("Neither model produced confident results.")
    
    #--------------------------------------------------------------------------#
    # Helpers
    #--------------------------------------------------------------------------#

    def _default_result(self, reason:str) -> Dict[str, Any]:
        """Return a default/unknown classification result."""
        logger.warning("Default classification: %s", reason)
        return {
            "category": "other",
            "confidence": 0.0,
            "strategy": self.strategy,
            "details": {"reason": reason},
        }

    @staticmethod
    def format_result(result: Dict[str, Any]) -> str:
        """format a classification result for display."""
        cat = result.get("category", "unknown").replace("_", " ").title()
        conf = result.get("confidence", 0.0)
        strategy = result.get("strategy", "N/A")
        return (
            f"|---------------------------------------------|\n"
            f"|  WASTE CLASSIFICATION RESULT                |\n"
            f"|---------------------------------------------|\n"
            f"|  Category   : {cat:<24}|\n"
            f"|  Confidence : {conf:<24.1%}|\n"
            f"|  Strategy   : {strategy:<24}|\n"
            f"|---------------------------------------------|\n"
        )