# To enter into virtual env
python -m venv .venv

# To enter into virtual env
.venv\Scripts\activate

# Install the project with all core dependencies
pip install -e .
pip install flask

# Include hardware (Arduino/serial) support
pip install -e ".[hardware]"

# Include dev tools (pytest, ruff)
pip install -e ".[dev]"

# Include everything
pip install -e ".[hardware,dev]"

# To creating mapping
python scripts/prepare_dataset.py --zip data/downloads/Yolo_Dataset1.zip --auto-map --min-score 0.4 --prefix "ds1_" --fresh

# Script to split dataset
python split_val.py

# To Train Yolo Model
python main.py --train-yolo

# To Train CNN Model
python main.py --train-cnn --dataset data/CNN_Dataset

YOLO: models/yolov8/train/weights/best.pt   ✅ trained
CNN:  models/cnn/best_cnn.pth               ✅ trained (96.31% accuracy)