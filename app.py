"""
Waste Classifier - Flask Web Application (FIXED)
Provides live camera feed, capture, and classification via browser.
"""

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# ── Lazy-load heavy models once ──────────────────────────────────────────────
_detector = None
_predictor = None
_cnn_predictor = None
_ensemble = None
_config = None


def get_config():
    global _config
    if _config is None:
        import sys
        # CHANGE TO: use parent of parent since config is at project root
        project_root = str(Path(__file__).resolve().parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from config import load_config
        _config = load_config()
    return _config


def get_models():
    global _detector, _predictor, _cnn_predictor, _ensemble
    if _detector is None:
        try:
            from YoloV8.model import YOLOv8Detector
            from YoloV8.predict import YOLOv8Predictor
            from CNN.predict import CNNPredictor
            from classifier.ensemble import WasteEnsembleClassifier

            cfg = get_config()
            yolo_cfg = cfg["yolov8"]
            cnn_cfg = cfg["cnn"]
            ens_cfg = cfg["ensemble"]

            print("Loading YOLO...")
            _detector = YOLOv8Detector(
                model_path=yolo_cfg["pretrained_weights"],
                confidence_threshold=yolo_cfg["confidence_threshold"],
                iou_threshold=yolo_cfg["iou_threshold"],
            )
            _detector.load_model()
            _predictor = YOLOv8Predictor(_detector)
            print("YOLO loaded ✓")

            print("Loading CNN...")
            _cnn_predictor = CNNPredictor(
                weights_path=cnn_cfg["weights_path"],
                architecture=cnn_cfg["architecture"],
                num_classes=cnn_cfg["num_classes"],
            )
            print("CNN loaded ✓")

            _ensemble = WasteEnsembleClassifier(
                object_category_map=cfg["categories"]["object_category_map"],
                strategy=ens_cfg["strategy"],
                yolo_weight=ens_cfg["yolo_weight"],
                cnn_weight=ens_cfg["cnn_weight"],
                min_confidence=ens_cfg["min_confidence"],
                adaptive_config=ens_cfg["adaptive_config"],
            )
            print("Ensemble loaded ✓")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR loading models: {e}")
            # Reset so next call tries again
            _detector = None
            _predictor = None
            _cnn_predictor = None
            _ensemble = None
            raise

    return _predictor, _cnn_predictor, _ensemble


# ── Camera ────────────────────────────────────────────────────────────────────
_camera = None

import threading
_capture_lock = threading.Lock()
_is_capturing = False

def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        for idx in [0, 1, 2]:
            _camera = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if _camera.isOpened():
                _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera opened at index {idx}")
                break
        else:
            print("ERROR: No camera found!")
            _camera = None
    return _camera


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/frame")
def frame():
    global _camera
    if _is_capturing:          # ← skip frame during capture
        return "", 204
    cam = get_camera()
    if cam is None or not cam.isOpened():
        return "", 204
    success, f = cam.read()
    if not success:
        cam.release()
        _camera = None
        return "", 204
    _, buffer = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buffer.tobytes(), mimetype="image/jpeg")


@app.route("/capture", methods=["POST"])
def capture():
    global _is_capturing
    try:
        print("=== CAPTURE STARTED ===")
        _is_capturing = True        # ← pause live feed
        time.sleep(0.1)             # ← let current frame request finish

        cam = get_camera()
        if cam is None:
            return jsonify({"error": "No camera available"}), 500

        success, frame = cam.read()
        print(f"Frame read: success={success}")
        if not success:
            return jsonify({"error": "Failed to capture frame"}), 500

        _is_capturing = False       # ← resume live feed early
        
        print("Loading models...")
        predictor, cnn_predictor, ensemble = get_models()
        print(f"Models loaded. Running YOLO...")

        detections = predictor.predict_image(frame)
        print(f"YOLO detections: {len(detections)}")

        crops = predictor.get_cropped_detections(frame, detections)
        cnn_preds = []
        for crop_info in crops:
            pred = cnn_predictor.predict(crop_info["crop"])
            cnn_preds.append(pred)

        result = ensemble.classify(detections, cnn_preds)
        print(f"Result: {result.get('category')} {result.get('confidence')}")

        annotated = draw_annotated(frame, detections, result)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        save_dir = Path("data/captured")
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(save_dir / f"capture_{ts}.jpg"), annotated)

        print("=== CAPTURE COMPLETE ===")
        return jsonify({
            "image": img_b64,
            "category": result.get("category", "unknown"),
            "confidence": round(result.get("confidence", 0.0) * 100, 1),
            "strategy": result.get("strategy", ""),
            "detections": [
                {"label": d["label"],
                 "confidence": round(d["confidence"] * 100, 1),
                 "bbox": d["bbox"]}
                for d in detections
            ],
            "num_objects": len(detections),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        _is_capturing = False       # ← always reset on error
        return jsonify({"error": str(e)}), 500


def draw_annotated(frame, detections, result):
    """Draw bounding boxes and labels on frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    COLOR_MAP = {
        "battery":    (0, 100, 255),
        "biological": (0, 200, 50),
        "cardboard":  (180, 120, 0),
        "clothes":    (200, 0, 150),
        "glass":      (0, 220, 220),
        "metal":      (150, 150, 150),
        "paper":      (255, 200, 0),
        "plastic":    (0, 80, 255),
        "shoes":      (160, 60, 0),
        "trash":      (80, 80, 80),
    }

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf = det["confidence"]
        color = COLOR_MAP.get(label.lower(), (0, 200, 100))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cat = result.get("category", "").upper()
    conf_pct = f"{result.get('confidence', 0):.0%}"
    overlay_text = f"  {cat}  {conf_pct}"
    (ow, oh), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    cv2.rectangle(img, (10, h - oh - 20), (10 + ow + 10, h - 8), (20, 20, 20), -1)
    cv2.putText(img, overlay_text, (15, h - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 150), 2)

    return img


@app.route("/release_camera", methods=["POST"])
def release_camera():
    global _camera
    if _camera:
        _camera.release()
        _camera = None
    return jsonify({"status": "released"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)


@app.route("/capture", methods=["POST"])
def capture():
    try:
        print("=== CAPTURE STARTED ===")  # ← ADD
        cam = get_camera()
        if cam is None:
            return jsonify({"error": "No camera available"}), 500

        success, frame = cam.read()
        print(f"Frame read: success={success}")  # ← ADD
        if not success:
            return jsonify({"error": "Failed to capture frame"}), 500

        print("Loading models...")  # ← ADD
        predictor, cnn_predictor, ensemble = get_models()
        print("Models loaded. Running YOLO...")  # ← ADD

        detections = predictor.predict_image(frame)
        print(f"YOLO detections: {len(detections)}")  # ← ADD

        crops = predictor.get_cropped_detections(frame, detections)
        print(f"Crops: {len(crops)}")  # ← ADD

        cnn_preds = []
        for crop_info in crops:
            pred = cnn_predictor.predict(crop_info["crop"])
            cnn_preds.append(pred)

        print("Running ensemble...")  # ← ADD
        result = ensemble.classify(detections, cnn_preds)
        print(f"Result: {result}")  # ← ADD

        annotated = draw_annotated(frame, detections, result)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        save_dir = Path("data/captured")
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(save_dir / f"capture_{ts}.jpg"), annotated)

        print("=== CAPTURE COMPLETE ===")  # ← ADD
        return jsonify({
            "image": img_b64,
            "category": result.get("category", "unknown"),
            "confidence": round(result.get("confidence", 0.0) * 100, 1),
            "strategy": result.get("strategy", ""),
            "detections": [
                {
                    "label": d["label"],
                    "confidence": round(d["confidence"] * 100, 1),
                    "bbox": d["bbox"],
                }
                for d in detections
            ],
            "num_objects": len(detections),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500