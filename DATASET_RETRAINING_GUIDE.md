# Dataset Analysis & Retraining Strategy Guide

## 1. CURRENT DATASET STRUCTURE

### 1.1 YOLO Dataset (`data/YOLO_Dataset/`)

**Directory Structure:**
```
data/YOLO_Dataset/
├── data.yaml                 # Dataset configuration (6 classes)
├── README.dataset.txt
├── README.roboflow.txt
├── train/
│   ├── images/              # Training images
│   └── labels/              # YOLO format labels (.txt files)
├── valid/                   # Validation split
│   ├── images/
│   └── labels/
└── test/                    # Test split
    ├── images/
    └── labels/
```

**Current Configuration (data.yaml):**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6  # Number of classes
names: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

roboflow:
  workspace: syd-ak5e1
  project: garbage-detection-kd1cv
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/syd-ak5e1/garbage-detection-kd1cv/dataset/2
```

**Format:** YOLO v8 format (one .txt label file per image with: class_id, x_center, y_center, width, height)

**Training Configuration (config.yaml):**
- Model: `yolov8n.pt` (nano - lightweight)
- Epochs: 100
- Batch size: 8
- Image size: 640×640
- Augmentation: Enabled
- Patience: 20 (early stopping)
- Saved model: `runs/detect/models/yolov8/train/weights/best.pt`

---

### 1.2 CNN Dataset (`data/CNN_Dataset/`)

**Directory Structure:**
```
data/CNN_Dataset/
├── README.dataset.txt
├── README.roboflow.txt
├── train/                   # Training images (8 splits / 10 categories)
│   ├── battery/             # ~850-900 images per category
│   ├── biological/
│   ├── cardboard/
│   ├── clothes/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   └── trash/
├── valid/                   # Validation images (~100-150 per category)
│   ├── 0/                   # (Note: odd numbering, likely v0 format)
│   └── [battery, biological, ...]
└── test/                    # Test images (~100-150 per category)
    ├── battery/
    ├── biological/
    └── [...]
```

**Number of Classes:** 10 (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash)

**Format:** Image files in class subdirectories (standard image classification dataset)

**Training Configuration (config.yaml):**
- Architecture: ResNet50 (pretrained ImageNet)
- Num Classes: 10
- Epochs: 100
- Batch size: 16
- Learning rate: 0.0001
- Scheduler step: 15 epochs
- Patience: 20 (early stopping)
- Dropout: 0.3
- Saved model: `models/cnn/best_cnn.pth`
- Previous accuracy: **96.31%** ✅

---

## 2. WHY PREVIOUS TRAINING DIDN'T MEET EXPECTATIONS

### Possible Reasons:

1. **YOLO Only Has 6 Classes (But System Expects 10)**
   - YOLO is trained on: BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC
   - System handles 10 categories (missing: BATTERY, CLOTHES, SHOES, TRASH)
   - Unmapped categories default to generic detection
   - **Impact:** Limited object detection → CNN has to work harder

2. **Class Imbalance**
   - Unclear if training data has balanced distribution per class
   - Some categories may have fewer samples → Poor generalization

3. **Insufficient Training Data**
   - RoboFlow datasets are often small (~100-500 images per class)
   - Deep learning benefits from larger datasets

4. **Mismatch Between YOLO & CNN**
   - YOLO detects 6 object types
   - CNN classifies 10 waste categories
   - Some waste items may not be detected → CNN never gets to classify them

5. **Model Architecture Bottleneck**
   - YOLOv8 Nano is very lightweight (fast but less accurate)
   - Could benefit from YOLOv8 Small (yolov8s) for better accuracy

---

## 3. WHERE TO ADD NEW DATASETS FOR RETRAINING

### Option A: Merge New Data with Existing Dataset (RECOMMENDED FOR INCREMENTAL IMPROVEMENT)

**Best for:** Adding more samples to existing categories

#### For YOLO:
```
data/YOLO_Dataset/
├── train/
│   ├── images/           # ADD NEW IMAGES HERE
│   └── labels/           # ADD NEW ANNOTATIONS (.txt files)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Steps:**
1. Prepare new YOLO-format dataset (images + .txt labels)
2. Merge image files into `data/YOLO_Dataset/train/images/`
3. Merge corresponding label files into `data/YOLO_Dataset/train/labels/`
4. **Recalculate train/val/test split** (typically 80% train, 10% val, 10% test)
   - Move ~10% of all images to `valid/`
   - Move ~10% of all images to `test/`
5. Keep `data.yaml` unchanged (still 6 classes)
6. Run training: `python main.py --train-yolo`

#### For CNN:
```
data/CNN_Dataset/
├── train/
│   ├── battery/          # ADD NEW IMAGES HERE
│   ├── biological/
│   ├── cardboard/
│   ├── clothes/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   └── trash/
├── valid/
│   └── [class folders]
└── test/
    └── [class folders]
```

**Steps:**
1. Collect or download new waste images
2. **Manually label/organize into class folders** (or use a tool like Roboflow)
3. Split into train/valid/test:
   ```python
   # Example split script
   train_ratio = 0.8
   valid_ratio = 0.1
   test_ratio = 0.1
   ```
4. Distribute new images across the corresponding class folders
5. Run training: `python main.py --train-cnn --dataset data/CNN_Dataset`

---

### Option B: Create Separate Dataset (BETTER FOR MAJOR OVERHAUL)

**Best for:** Complete retraining with new/better dataset

#### Step 1: Create New Dataset Directory
```
data/CNN_Dataset_v2/        # Or YOLO_Dataset_v2
├── train/
│   └── [class folders OR images/labels/]
├── valid/
└── test/
```

#### Step 2: Update config.yaml to Point to New Dataset

**For YOLO:**
```yaml
yolov8:
  training:
    data_yaml: "data/YOLO_Dataset_v2/data.yaml"  # ← Change this
```

**For CNN:**
Modify training call:
```bash
python main.py --train-cnn --dataset data/CNN_Dataset_v2
```

#### Step 3: Update Paths in main.py (if needed)

Search for hardcoded paths and update:
```python
# In main.py, the training calls use config values
# So config.yaml is the SOURCE OF TRUTH
```

---

## 4. RECOMMENDED STRATEGY FOR RETRAINING

### Phase 1: Data Collection & Preparation

**Goal:** Expand dataset by 30-50%

1. **For YOLO (Object Detection):**
   - Collect 200-500 new waste images
   - Annotate using LabelImg or Roboflow (box annotations)
   - Ensure you have annotations for **all 6 classes** (BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC)
   - Export in YOLO format
   - **Better yet: Add 4 more YOLO classes** (BATTERY, CLOTHES, SHOES, TRASH)
     - This requires re-annotating entire dataset with 10 classes!

2. **For CNN (Classification):**
   - Collect 100-200 images per waste category (total 1000-2000 new images)
   - Organize into class folders
   - **Target improvement:** Increase from ~2000 images to ~4000 images (more data = better accuracy)

### Phase 2: Retraining Strategy

#### Conservative Approach (Incremental Fine-tuning):
```bash
# Add new data to existing dataset
# Folders: data/YOLO_Dataset/train and data/CNN_Dataset/train

# Train YOLO (starts from pretrained weights)
python main.py --train-yolo
# → Model loads yolov8n.pt, fine-tunes on all data

# Train CNN (starts from pretrained weights)
python main.py --train-cnn --dataset data/CNN_Dataset
# → Model loads pretrained ResNet50, fine-tunes on all data
```

#### Aggressive Approach (Larger Model + More Data):

```bash
# Modify config.yaml to use larger model:
yolov8:
  model_variant: "yolov8s.pt"  # ← Change from yolov8n to yolov8s (Small)
  training:
    epochs: 150  # ← Increase epochs
    batch_size: 16  # ← Increase batch (if GPU memory allows)
    patience: 30  # ← Increase patience

# Also for CNN:
cnn:
  architecture: "resnet50"  # Keep this (already good)
  training:
    epochs: 150
    batch_size: 32  # ← If GPU memory allows
    patience: 30
```

---

## 5. FILES TO MODIFY FOR RETRAINING

### Minimal Changes (Just Retrain):
Nobody to change! Just:
```bash
python main.py --train-yolo
python main.py --train-cnn --dataset data/CNN_Dataset
```

### To Improve Model Performance:

**File: `config/config.yaml`**

Changes to make CNN better:
```yaml
cnn:
  architecture: "resnet50"  # Keep or try "resnet50" / "efficientnet_b0"
  num_classes: 10           # Keep
  pretrained: true          # Keep (ImageNet pretrained)
  weights_path: "models/cnn/best_cnn.pth"  # Keep (loads pretrained)
  dropout: 0.2              # Reduce from 0.3 (less regularization if smaller model)
  device: "auto"            # Keep
  
  training:
    epochs: 150             # Increase from 100
    batch_size: 32          # Increase from 16 (if GPU memory allows)
    learning_rate: 0.0001   # Keep or reduce to 0.00005
    scheduler_step: 20      # Increase from 15
    scheduler_gamma: 0.1    # Keep
    patience: 30            # Increase from 20
```

Changes to make YOLO better:
```yaml
yolov8:
  model_variant: "yolov8s.pt"  # Larger model: yolov8n → yolov8s
  pretrained_weights: "runs/detect/models/yolov8/train/weights/best.pt"  # Keep
  device: "auto"
  training:
    data_yaml: "data/YOLO_Dataset/data.yaml"  # Keep
    epochs: 150             # Increase from 100
    batch_size: 16          # Increase from 8 (if GPU allows)
    patience: 30            # Increase from 20
    augment: true           # Keep
```

---

## 6. DATA PREPARATION CHECKLIST

### For YOLO:
- [ ] New images in `data/YOLO_Dataset/train/images/`
- [ ] Corresponding .txt labels in `data/YOLO_Dataset/train/labels/`
- [ ] Valid/test splits updated (80/10/10 ratio)
- [ ] `data.yaml` references correct paths
- [ ] All 6 classes have annotations

### For CNN:
- [ ] New images organized by class: `data/CNN_Dataset/train/[class_name]/`
- [ ] At least 100-200 images per class in training split
- [ ] Valid/test splits populated similarly
- [ ] No corrupt image files
- [ ] Consistent image formats (.jpg, .png)

---

## 7. TRAINING EXECUTION

### To Retrain YOLO:
```bash
cd c:\Users\ms675\OneDrive\Desktop\Waste-Segregation-Project

# Activate virtual environment
.venv\Scripts\activate

# Train YOLO
python main.py --train-yolo
```

**Output:**
- Best model saved to: `runs/detect/models/yolov8/train/weights/best.pt`
- Update `config.yaml` if needed to use new weights

### To Retrain CNN:
```bash
python main.py --train-cnn --dataset data/CNN_Dataset
```

**Output:**
- Best model saved to: `models/cnn/best_cnn.pth`
- Automatically picked up by inference pipeline

---

## 8. EXPECTED IMPROVEMENTS

### If You Add 30-50% More Data:
- **CNN Accuracy:** 96.31% → ~97-98% (marginal improvement due to already high accuracy)
- **YOLO Detection:** Moderate improvement in recall and precision
- **Overall System:** Better edge case handling, fewer misclassifications

### If You Switch to YOLOv8 Small:
- **Speed:** Slightly slower inference (~50-100ms vs 30-50ms)
- **Accuracy:** 5-10% improvement in detection precision
- **Trade-off:** Real-time processing still feasible

### If You Expand YOLO to 10 Classes:
- **Major improvement:** Can detect all waste categories directly
- **Impact:** CNN workload reduced, system more robust
- **Effort:** Requires re-annotating all existing YOLO images with 10 classes

---

## 9. QUICK START FOR RETRAINING

### Simplest Path (Recommended):

1. **Collect new images** (500-1000 for YOLO, 2000-3000 for CNN)

2. **For CNN:**
   ```bash
   # Simply add images to appropriate class folders in training split
   data/CNN_Dataset/train/battery/      # Add 150 new battery images
   data/CNN_Dataset/train/biological/   # Add 150 new biological images
   # ... etc for all 10 classes
   ```

3. **For YOLO:**
   - Use Roboflow to annotate new images
   - Export in YOLO format
   - Merge into `data/YOLO_Dataset/train/`
   - Split into validation/test

4. **Retrain:**
   ```bash
   python main.py --train-yolo          # ~30-45 minutes
   python main.py --train-cnn --dataset data/CNN_Dataset  # ~60-90 minutes
   ```

5. **Test:** Use `python main.py --camera` to verify improvements

---

## 10. TROUBLESHOOTING

### If Training Keeps Stopping Early:
- **Issue:** Early stopping triggered at patience limit
- **Solution:** Check data quality, increase patience in config, or add more diverse data

### If Accuracy Decreases After Adding Data:
- **Issue:** New data has different distribution (domain shift)
- **Solution:** 
  - Ensure new data matches existing data quality/domain
  - Reduce learning rate in config
  - Increase freeze_backbone_epochs in CNN training

### If GPU Out of Memory:
- **Issue:** Batch size too large
- **Solution:** Reduce batch size in config (16 → 8 for YOLO, 32 → 16 for CNN)

### If Models Don't Improve:
- **Issue:** Insufficient new data or data not diverse enough
- **Solution:** Aim for 3-5x original dataset size, include edge cases, difficult scenarios

---

## Summary of Recommended Actions

| Step | Action | Location | Command |
|------|--------|----------|---------|
| 1 | Collect new dataset | External | Download/capture images |
| 2 | Organize CNN data | `data/CNN_Dataset/train/[class]/` | Sort images into folders |
| 3 | Annotate YOLO data | External | Use Roboflow/LabelImg |
| 4 | Merge datasets | `data/YOLO_Dataset/train/` | Copy images & labels |
| 5 | Update config (optional) | `config/config.yaml` | Increase epochs, batch size |
| 6 | Train YOLO | Terminal | `python main.py --train-yolo` |
| 7 | Train CNN | Terminal | `python main.py --train-cnn --dataset data/CNN_Dataset` |
| 8 | Evaluate models | Terminal | `python main.py --camera` |
| 9 | Save best weights | Auto-saved | `models/cnn/best_cnn.pth`, `runs/detect/...` |

