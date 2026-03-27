"""
Waste Classifier - Main Entry Point
=========================================================
End-to-end pipeline
    1. Capture / load image
    2. YOLOv8 detection -> Identify waste object
    3. CNN Classification -> categorize each detection
    4. Ensemble fusion -> final waste category
    5. Store result in DB (MongoDB / SQLite)
    6. Export data for PowerBI dashboard
=========================================================

Usage:
    # Classify a single image file
    python main.py --image path/to/waste.jpg

    # Classify from webcam capture
    python main.py --camera

    # Run on a directory of images
    python main.py --dir path/to/images

    # Train YOLOv8
    python main.py --train-yolo

    # Train CNN
    python main.py --train-cnn --dataset data/CNN_Dataset

    # Export DB data for PowerBI
    python main.py --export

    # View DB summary stats
    python main.py --db-stats
"""

import argparse
import sys
from pathlib import Path 

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, get_device
from utils.logger import get_logger
from utils.image_processing import draw_predictions
from capture.camera import Camera


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Waste Classifier - YOLOv8 + CNN Ensemble Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Path to a single image file.")
    mode.add_argument("--camera", action="store_true", help="Capture from webcam.")
    mode.add_argument("--dir", type=str, help="Directory of image to classify.")
    mode.add_argument("--train-yolo", action="store_true", help="Train YOLOv8 model.")
    mode.add_argument("--train-cnn", action="store_true", help="Train CNN model.")
    mode.add_argument("--export", action="store_true", 
                      help="Export DB records for PowerBI (CSV, Excel, JSON).")
    mode.add_argument("--db-stats", action="store_true",
                      help="Print database summary statistics.")
    
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Dataset root for CNN training (folder with class sub-dirs).")
    parser.add_argument("--save-output", action="store_true",
                        help="Save annotated output images.")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated images on screen")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip saving results to database.")
    parser.add_argument("--hardware", action="store_true",
                        help="Enable Arduino hardware to route waste to bins.")
    parser.add_argument("--simulate-hw", action="store_true",
                        help="Simulate hardware (no real Arduino needed).")
    parser.add_argument("--port", action="store_true",
                        help="Arduino serial port (overrides config).")
    
    return parser.parse_args()


# ========================================================= #
# INFERENCE PIPELINE
# ========================================================= #

def run_inference(image_path: str, config: dict, logger, save: bool = False, show: bool = False, db_storage=None, source: str = "file", hw_controller=None):
    """
    Run the full inference pipeline on a single image.

    Steps: Load image -> YOLOv8 detect -> CNN classify crops -> Ensemble -> DB -> Hardware.
    """
    from YoloV8.model import YOLOv8Detector
    from YoloV8.predict import YOLOv8Predictor
    from CNN.predict import CNNPredictor
    from classifier.ensemble import WasteEnsembleClassifier

    logger.info("=" * 60)
    logger.info("Processing: %s", image_path)
    logger.info("=" * 60)

    # 1. Load image
    image = Camera.load_image(image_path)
    logger.info("Image loaded - shape=%s", image.shape)

    # 2. YOLOv8 detection
    yolo_cfg = config["yolov8"]
    device = get_device(yolo_cfg["device"])

    detector = YOLOv8Detector(
        model_path=yolo_cfg["pretrained_weights"],
        confidence_threshold=yolo_cfg["confidence_threshold"],
        iou_threshold=yolo_cfg['iou_threshold'],
        device=device,
    )
    detector.load_model()

    predictor = YOLOv8Predictor(detector)
    yolo_detections = predictor.predict_image(image)
    cropped = predictor.get_cropped_detections(image, yolo_detections)

    logger.info("YOLO detected %d objects.", len(yolo_detections))
    for det in yolo_detections:
        logger.info("  -> %s (%.2f%%)", det["label"], det["confidence"] * 100)

    # 3. CNN classification on cropperd detections
    cnn_cfg = config["cnn"]
    cnn_device = get_device(cnn_cfg["device"])

    cnn_predictor = CNNPredictor(
        weights_path=cnn_cfg["weights_path"],
        architecture=cnn_cfg["architecture"],
        num_classes=cnn_cfg["num_classes"],
        device=cnn_device,
    )

    cnn_predictions = []
    for crop_info in cropped:
        prediction = cnn_predictor.predict(crop_info["crop"])
        cnn_predictions.append(prediction)
        logger.info(
            "  CNN -> %s (%.2f%%) for YOLO label '%s'",
            prediction["category"], prediction["confidence"] * 100,
            crop_info["label"],
        )

    # 4. Ensemble final classificaiton
    ens_cfg = config["ensemble"]
    ensemble = WasteEnsembbleClassifier(
        object_category_map=config["categories"]["object_category_map"],
        strategy=ens_cfg["strategy"],
        yolo_weight=ens_cfg["yolo_weight"],
        cnn_weight=ens_cfg["cnn_weight"],
        min_confidence=ens_cfg["min_confidence"],
        adaptive_config=ens_cfg["adaptive_config"],
    )

    result = ensemble.classify(yolo_detections, cnn_predictions)

    # Display result
    print("\n" + WasteEnsembbleClassifier.format_result(result) + "\n")
    logger.info("Final category: %s (%.1f%%)", result["category"], result["confidence"] * 100)

    # 5. Store result in database
    if db_storage is not None:
        from database.models import WasteRecord

        # Strip non-serializable 'crop' arrays from YOLO detections for DB storage
        serializable_dets = [
            {k: v for k, v in det.items() if k != "crop"}
            for det in yolo_detections
        ]

        record = WasteRecord(
            image_path=image_path,
            final_category=result["category"],
            final_confidence=result["confidence"],
            strategy=result.get("strategy", ""),
            yolo_detections=serializable_dets,
            cnn_prediction=cnn_predictions,
            all_probabilities=result.get("all_probabilities", {}),
            source=source,
        )
        db_storage.insert_record(record)
        logger.info("Result saved to database (ID: %s)", record.record_id)

    # 6. Route to hardware bin (if hardware enabled)
    if hw_controller is not None:
        category = result["category"]
        logger.info("Routing waste to hardware bin: %s", category)
        hw_success = hw_controller.route_waste(category)
        if hw_success:
            logger.info("Hardware: waste routed successfully.")
        else:
            logger.warning("Hardware: failed t o route waste.")

    # 7. Annotate and optionally save/show
    if save or show:
        annotated = draw_predictions(
            image, yolo_detections, result["category"], result["confidence"]
        )

        if save:
            out_dir = Path(config["output"]["results_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"result_{Path(image_path).stem}.jpg"
            import cv2
            cv2.imwrite(str(out_path), annotated)
            logger.info("Annotated image saved to %s", out_path)

        if show:
            Camera.show_image(annotated, window_name="Waste Classificaton Result")

    return result


def run_camera_inference(config: dict, logger, save: bool = False, show: bool = False, db_storage=None, hw_controller=None):
    """Capture an image from webcam and classify it."""
    cap_cfg = config["capture"]
    camera = Camera(
        camera_index=cap_cfg["camera_index"],
        frame_width=cap_cfg["frame_width"],
        frame_height=cap_cfg["frame_height"],
        save_dir=cap_cfg["save_dir"],
    )

    with camera:
        frame, saved_path = camera.capture_and_save()
        logger.info("Captured frame saved to %s", saved_path)

    return run_inference(saved_path, config, logger, save=save, show=show, db_storage=db_storage, source="camera", hw_controller=hw_controller)


def run_directory_inference(dir_path: str, config: dict, logger, save: bool = False, db_storage=None, hw_controller=None):
    """Run inference on all images in a directory"""
    dir_path = Path(dir_path)
    extension = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in sorted(dir_path.iterdir()) if f.suffix.lower() in extension]

    logger.info("Found %d images in '%s", len(images), dir_path)

    results = {}
    for img_file in images:
        try:
            result = run_inference(str(img_file), config, logger, save=save, db_storage=db_storage, source="batch", hw_controller=hw_controller)
            result[img_file.name] = result
        except Exception as e:
            logger.error("Error processing %s: %s", img_file.name, e)
            result[img_file.name] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH RESULTS SUMMARY")
    logger.info("=" * 60)
    for name, res in results.items():
        cat = res.get("category", "error")
        conf = res.get("confidence", 0.0)
        logger.info(" %s -> %s (%.1f%%)", name, cat, conf * 100)

    return results


# ========================================================= #
# DATABASE & EXPORT
# ========================================================= #

def get_db_storage(config: dict, logger):
    """Create and connect to the configured storage backend"""
    from database.factory import get_storage_backend

    db_cfg = config.get("database", {})
    backend = db_cfg.get("backend", "mongodb")

    kwargs=  {}
    if backend == "mongodb":
        mongo_cfg = db_cfg.get("mongodb", {})
        kwargs = {
            "connnection_url": mongo_cfg.get("connection_uri", "mongodb://localhost:27017"),
            "database_name": mongo_cfg.get("database_name", "waste_classifier"),
            "collection_name": mongo_cfg.get("colleciton_name", "wste_records"),
        }
    elif backend == "sqlite":
        sqlite_cfg = db_cfg.get("sqlite", {})
        kwargs = {"db_path": sqlite_cfg.get("db_path", "data/waste_classifier.db")}

    storage = get_storage_backend(backend, **kwargs)

    try:
        storage.connect()
        logger.info("Database connected (%s)", backend)
    except Exception as e:
        logger.warning("DB connection failed (%s): %s, Trying SQLite fallback,", backend, e)
        try:
            sqlite_cfg = db_cfg.get("sqlite", {})
            storage = get_storage_backend(
                "sqlite", db_path=sqlite_cfg.get("db_path", "data/waste_classifier.db")
            )
            storage.connect()
            logger.info("SQLite fallback connected.")
        except Exception as e2:
            logger.error("All DB backend failed: %s. Proceeding without DB.", e2)
            return None
        
    return storage


def run_export(config: dict, logger):
    """Export database records for PowerBI."""
    from database.export import PowerBIExporter

    storage = get_db_storage(config, logger)
    if storage is None:
        logger.error("Cannot export - no database connection.")
        return
    
    export_cfg = config.get("export", {})
    output_dir = export_cfg.get("output_dir", "outputs/exports")

    exporter = PowerBIExporter(storage=storage, output_dir=output_dir)
    paths = exporter.export_all()

    print("\n" + "=" * 50)
    print("  POWERBI EXPORT COMPLETE")
    print("=" * 50)
    for fmt, path in paths.items():
        if path:
            print(f" {fmt.upper():>6}: {path}")
    print("=" * 50 + "\n")

    storage.disconnect()


def show_db_stats(config: dict, logger):
    """Print Database summary statistics."""
    storage = get_db_storage(config, logger)
    if storage is None:
        logger.error("Cannot show stats = no database connection.")
        return
    
    stats = storage.get_summary_stats()

    print("\n" + "=" * 50)
    print("  DATABASE Summary")
    print("=" * 50)
    print(f"  Total records       : {stats['total_records']}")
    print(f"  Avg confidence      : {stats['average_confidence']:.1%}")
    print(f"  Date range          : {stats['data_range'].get('earliest', 'N/A')}")
    print(f"                      : {stats['data_range'].get('latest', 'N/A')}")
    print()
    print("    Category Breakdown:")
    for cat, cnt in stats.get("category_counts", {}).items():
        pct = cnt / stats['total_records'] * 100 if stats['total_records'] > 0 else 0
        print(f"   {cat:<40} {cnt:>50}  ({pct:.1f}%)")
    print()
    print("  Source Breakdown:")
    for src, cnt in stats.get("source_counts", {}).items():
        print("=" * 50 + "\n")
    print("=" * 50 + "\n")

    storage.disconnect()


def get_hw_controller(config: dict, logger, simulate: bool = False, port_override: str = None):
    """
    Create hardware controller (real or simulated).
    Fully independent - failure does not affect classification.
    """
    hw_cfg = config.get("hardware", {})

    if simulate:
        from hardware.simulator import ArduinoSimulator
        controller = ArduinoSimulator(delay_factor=hw_cfg.get("sim_delay_factor", 0.1))
        try:  
            controller.connect()
            logger.info("hardware simulator connected.")
            return controller
        except Exception as e:
            logger.warning("Hardware simulator failed: %s", e)
            return None
        
    # Real hardware
    try:
        from hardware.controller import ArduinoController
    except ImportError:
        logger.warning("pyserial not installed. Hardware disabled")
        return None
    
    port = port_override or hw_cfg.get("port", "COM3")
    baud = hw_cfg.get("baud_rate", 9600)

    controller = ArduinoController(port=port, baud_rate=baud, timeout=hw_cfg.get("timeout", 2.0))
    try:
        controller.connect()
        logger.info("Arduino connected on %s.", port)
        return controller
    except Exception as e:
        logger.warning("Arduino connection failed on %s: %s. Proceeding without hardware.", port, e)
        return None


# ========================================================= #
# TRAINING PIPELINES
# ========================================================= #

def train_yolo(config: dict, logger):
    """Train the YOLOv8 model."""
    from YoloV8.train import YOLOv8Trainer

    yolo_cfg = config["yolov8"]
    trainer = YOLOv8Trainer(
        model_variant = yolo_cfg["model_variant"],
        data_yaml = yolo_cfg["training"]["data_yaml"],
        epochs = yolo_cfg["training"]["epochs"],
        batch_size = yolo_cfg["training"]["batch_size"],
        img_size = yolo_cfg["training"]["img_size"],
        patience = yolo_cfg["training"]["patience"],
        save_dir = yolo_cfg["training"]["save_dir"],
        device = yolo_cfg["device"],
        augment = yolo_cfg["training"]["augment"],
    )

    results = trainer.train()
    logger.info("YOLOv8 training complete. Best weights: %s", results.get("best_weights"))
    return results


def train_cnn(config: dict, dataset_root: str, logger):
    """Train the cnn model."""
    from CNN.train import CNNTrainer
    from CNN.dataset import create_dataloaders
    from CNN.model import WASTE_CATEGORIES

    cnn_cfg = config["cnn"]

    # CHANGE TO: use pre-split folders directly
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dir=dataset_root + "/train",
        val_dir=dataset_root + "/valid",
        test_dir=dataset_root + "/test",
        batch_size=cnn_cfg["training"]["batch_size"],
        img_size=cnn_cfg["image_processing"]["cnn_input_size"][0],
        class_names=WASTE_CATEGORIES,
    )

    trainer = CNNTrainer(
        architecture=cnn_cfg["architecture"],
        num_classes=cnn_cfg["num_classes"],
        pretrained=cnn_cfg["pretrained"],
        dropout=cnn_cfg["dropout"],
        learning_rate=cnn_cfg["training"]["learning_rate"],
        weight_decay=cnn_cfg["training"]["weight_decay"],
        scheduler_step=cnn_cfg["training"]["scheduler_step"],
        scheduler_gamma=cnn_cfg["training"]["scheduler_gamma"],
        epochs=cnn_cfg["training"]["epochs"],
        patience=cnn_cfg["training"]["patience"],
        save_dir=cnn_cfg["training"]["save_dir"],
        device=cnn_cfg["device"],
    )

    results = trainer.train(train_loader, val_loader)
    logger.info("CNN training complete. Best weights: %.2f%%", results["best_val_acc"] * 100)
    return results


# ========================================================= #
# MAIN
# ========================================================= #

def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logger
    log_cfg = config["logging"]
    logger = get_logger(
        "WasteClassifier",
        log_file=log_cfg["log_file"],
        level=log_cfg["level"],
        console=log_cfg["console"],
    )

    logger.info("Waste Classifier Pipeline Started")
    logger.info("Project root: %s", PROJECT_ROOT)

    # Initialize database storage (if enabled)
    db_storage = None
    db_cfg = config.get("database", {})
    track_results = db_cfg.get("track_results", True)

    if track_results and not getattr(args, "no_db", False):
        db_storage = get_db_storage(config, logger)

    # Initialize hardware controller (if requested)
    hw_controller = None
    if getattr(args, "hardware", False) or getattr(args, "simulate_hw", False):
        hw_controller = get_hw_controller(config, logger, simulate=getattr(args, "simulate_hw", False), port_override=getattr(args, "port", None))

    try:
        if args.image:
            run_inference(args.image, config, logger, save=args.save_output, show=args.show, db_storage=db_storage, hw_controller=hw_controller)

        elif args.camera:
            run_camera_inference(config, logger, save=args.save_output, show=args.show, db_storage=db_storage, hw_controller=hw_controller)

        elif args.dir:
            run_directory_inference(args.dir, config, logger, save=args.save_output, show=args.show, db_storage=db_storage, hw_controller=hw_controller)

        elif args.train_yolo:
            train_yolo(config, logger)

        elif args.train_cnn:
            if not args.dataset:
                logger.error("--dataset is required for cNN training.")
                sys.exit(1)
            train_cnn(config, args.dataset, logger)
        
        elif args.export:
            run_export(config, logger)

        elif args.db_stats:
            show_db_stats(config, logger)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by users.")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)
    finally:
        if hw_controller is not None:
            hw_controller.disconnect()
            logger.info("Hardware Disconnected.")
        if db_storage is not None:
            db_storage.disconnect()
            logger.info("Database connection closed.")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()