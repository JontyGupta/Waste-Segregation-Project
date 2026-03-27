"""
Waste Classifier - Flask Web Application
Provides live camera feed, capture, and classification via browser.
"""

import base64
import io
import json
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
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import load_config
        _config = load_config()
    return _config


def get_models():
    global _detector, _predictor, _cnn_predictor, _ensemble
    if _detector is None:
        from YoloV8.model import YOLOv8Detector
        from YoloV8.predict import YOLOv8Predictor
        from CNN.predict import CNNPredictor
        from classifier.ensemble import WasteEnsembleClassifier

        cfg = get_config()
        yolo_cfg = cfg["yolov8"]
        cnn_cfg = cfg["cnn"]
        ens_cfg = cfg["ensemble"]

        _detector = YOLOv8Detector(
            model_path=yolo_cfg["pretrained_weights"],
            confidence_threshold=yolo_cfg["confidence_threshold"],
            iou_threshold=yolo_cfg["iou_threshold"],
        )
        _detector.load_model()
        _predictor = YOLOv8Predictor(_detector)

        _cnn_predictor = CNNPredictor(
            weights_path=cnn_cfg["weights_path"],
            architecture=cnn_cfg["architecture"],
            num_classes=cnn_cfg["num_classes"],
        )

        _ensemble = WasteEnsembleClassifier(
            object_category_map=cfg["categories"]["object_category_map"],
            strategy=ens_cfg["strategy"],
            yolo_weight=ens_cfg["yolo_weight"],
            cnn_weight=ens_cfg["cnn_weight"],
            min_confidence=ens_cfg["min_confidence"],
            adaptive_config=ens_cfg["adaptive_config"],
        )

    return _predictor, _cnn_predictor, _ensemble


# ── Camera stream ─────────────────────────────────────────────────────────────
camera = None


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def generate_frames():
    """Yield MJPEG frames for live feed."""
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
        time.sleep(0.03)  # ~30fps


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def capture():
    """Capture frame, run full pipeline, return annotated image + results."""
    try:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            return jsonify({"error": "Failed to capture frame"}), 500

        predictor, cnn_predictor, ensemble = get_models()

        # YOLO detection
        detections = predictor.predict_image(frame)
        crops = predictor.get_cropped_detections(frame, detections)

        # CNN on crops
        cnn_preds = []
        for crop_info in crops:
            pred = cnn_predictor.predict(crop_info["crop"])
            cnn_preds.append(pred)

        # Ensemble
        result = ensemble.classify(detections, cnn_preds)

        # Annotate frame
        annotated = draw_annotated(frame, detections, result)

        # Encode to base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        # Save capture
        save_dir = Path("data/captured")
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(save_dir / f"capture_{ts}.jpg"), annotated)

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
        return jsonify({"error": str(e)}), 500


def draw_annotated(frame, detections, result):
    """Draw bounding boxes and labels on frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # Category color map
    COLOR_MAP = {
        "battery": (0, 100, 255),
        "biological": (0, 200, 50),
        "cardboard": (180, 120, 0),
        "clothes": (200, 0, 150),
        "glass": (0, 220, 220),
        "metal": (150, 150, 150),
        "paper": (255, 200, 0),
        "plastic": (0, 80, 255),
        "shoes": (160, 60, 0),
        "trash": (80, 80, 80),
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

    # Overlay final category
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
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "released"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)