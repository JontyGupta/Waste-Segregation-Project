# Waste Segregation System - Detailed Project Summary

## 1. PROJECT OVERVIEW

**Waste Segregation Project** is an intelligent, AI-powered waste classification and automatic routing system designed to automatically sort waste into 10 distinct categories using computer vision and deep learning.

---

## 2. WHAT DOES THE SOFTWARE DO?

### 2.1 Core Software Architecture

The system implements a **two-stage AI pipeline** for waste classification:

#### **Stage 1: Object Detection (YOLO)**
- **Model:** YOLOv8 Nano (lightweight variant)
- **Function:** Detects the presence and location of waste objects in images
- **Input:** Image (640×640 pixels)
- **Output:** Bounding boxes with object class labels
- **Accuracy:** Trained on YOLO dataset with labeled waste categories
- **Model File:** `models/yolov8/train/weights/best.pt`

#### **Stage 2: Fine-Grained Classification (CNN)**
- **Model:** ResNet50 (or ResNet18/EfficientNet based on config)
- **Function:** Classifies individual waste crops into detailed waste categories
- **Input:** Cropped image regions (224×224 pixels)
- **Output:** Waste category with confidence score
- **Accuracy:** 96.31% on CNN dataset (10 categories)
- **Model File:** `models/cnn/best_cnn.pth`

#### **Stage 3: Ensemble Fusion**
- **Strategy:** Confidence-adaptive weighted voting
- **Function:** Fuses YOLO detection results with CNN classification predictions
- **Logic:**
  - If YOLO confidence is high (>0.85): Trust YOLO more (70% weight)
  - If YOLO confidence is medium (0.5-0.85): Balanced fusion (55% YOLO, 45% CNN)
  - If YOLO confidence is low (<0.5): Rely on CNN more (30% weight)
- **Output:** Final waste category with combined confidence score

### 2.2 Supported Waste Categories (10 Types)

1. **Battery** - Single-use and rechargeable batteries
2. **Biological** - Biodegradable organic waste (food scraps, leaves)
3. **Cardboard** - Cardboard boxes and packaging
4. **Clothes** - Textiles and garments
5. **Glass** - Glass bottles, jars, and items
6. **Metal** - Metal cans, containers, aluminum
7. **Paper** - Paper, newspapers, magazines
8. **Plastic** - Plastic bottles, bags, containers
9. **Shoes** - Footwear and shoe waste
10. **Trash** - General mixed/unclassifiable waste

### 2.3 Data Capture & Input Methods

The software supports multiple image acquisition sources:

- **Webcam/Live Camera Input** (640×480 resolution)
  - Real-time continuous stream processing
  - Frame-by-frame analysis
  - Flask web interface for live viewing

- **Single Image File** Processing
  - Load and classify individual images
  - Batch processing of directories

- **Database Storage**
  - MongoDB or SQLite backend (configurable)
  - Stores classification results with metadata:
    - Timestamp
    - Detected category
    - Confidence scores
    - Source image path

### 2.4 Output & Export Features

- **Terminal Output:** Real-time classification results with confidence scores
- **Annotated Images:** Saves visualizations with bounding boxes and category labels
- **Web Dashboard:** Flask-based live camera feed + classification interface (accessible at `localhost:5000`)
- **PowerBI Export:** Database export to CSV, Excel, JSON for business intelligence dashboards
- **Database Statistics:** Summary queries of classified waste records

### 2.5 ML Training Pipeline

The software includes complete model training capabilities:

**YOLO Training:**
- Uses RoboFlow YOLO dataset (labeled waste object images)
- Hyperparameters: 100 epochs, batch size 8, 640×640 input
- Augmentation enabled for robustness
- 20-epoch patience for early stopping

**CNN Training:**
- Uses RoboFlow CNN dataset (pre-cropped waste images)
- ResNet50 architecture with ImageNet pre-training
- Hyperparameters: 100 epochs, batch size 16, learning rate 0.0001
- Data split: Training/Validation/Test (8/1/1 typical split)
- GPU acceleration supported (CUDA 12.1 compatible)

### 2.6 Hardware Control Interface

**Arduino Communication Module:**
- Sends serial commands to Arduino microcontroller
- Protocol: Simple text-based commands over USB serial (9600 baud)
- Commands:
  - `ROUTE:<bin_id>` - Route waste to specific bin
  - `OPEN:<bin_id>` - Manually open a bin gate
  - `CLOSE:<bin_id>` - Manually close a bin gate
  - `RESET` - Close all gates (safety reset)
  - `PING` - Health check
  - `STATUS` - Request hardware status
  - `LED:<bin>:<R>:<G>:<B>` - Control RGB LED indicators

---

## 3. WHAT SHOULD THE HARDWARE DO?

### 3.1 Hardware System Architecture

The embedded hardware system is a **modular waste routing machine** controlled by Arduino.

```
┌─────────────────────────────────────────┐
│      Input Conveyor Belt                │
│   (Carries waste items to classifier)   │
└──────────────┬──────────────────────────┘
               │ (Waste slides down)
               ▼
┌─────────────────────────────────────────┐
│   Central Diverter Servo (motorized)    │
│   (Routes waste to correct bin)         │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────────┐
      ▼                     ▼
  [Open Gate]          [Close Gate]
  (90 degrees)         (0 degrees)
      │                     │
      ▼                     ▼
   Bin Opens          Waste falls into
   Waste drops in      correct bin
   Gate closes
   (3-4 sec hold)
```

### 3.2 Individual Bin Motion Control

**10 Independent Waste Bins** each controlled by a servo motor or stepper motor:

| Bin | Category | Arduino Pin | Gate Style | Hold Time |
|-----|----------|-------------|-----------|-----------|
| 0   | Biodegradable | 2 | Servo 90° | 3000 ms |
| 1   | Recyclable | 3 | Servo 90° | 3000 ms |
| 2   | Non-Recyclable | 4 | Servo 90° | 3000 ms |
| 3   | Medical Waste | 5 | Servo 90° | 4000 ms (longer) |
| 4   | E-Waste | 6 | Servo 90° | 3000 ms |
| 5   | Hazardous | 7 | Servo 90° | 4000 ms (longer) |
| 6   | Textile | 8 | Servo 90° | 3000 ms |
| 7   | Construction | 9 | Servo 90° | 3500 ms |
| 8   | Sanitary | 10 | Servo 90° | 3000 ms |
| 9   | Other | 11 | Servo 90° | 3000 ms |

### 3.3 Sensing Requirements

The hardware must **sense**:

- **Waste Presence Detection:**
  - IR proximity sensor (optional, on conveyor belt)
  - Detect when waste reaches classification point
  - Trigger camera capture

- **Weight/Force Sensing (Optional):**
  - Load cell per bin (optional)
  - Monitor how full each bin is
  - Trigger "bin full" alerts

- **Gate Position Feedback (Optional):**
  - Limit switches on bin gates
  - Confirm gate opened/closed successfully
  - Safety verification

### 3.4 Control Operations

The hardware must **control**:

- **Servo Motors:** Command gate opening/closing
  - Issue: 1 servo per bin = 10 servo control lines
  - Solution: I. Arduino Mega (16+ PWM pins) OR
  - Solution: 2. PCA9685 servo driver (I²C multiplexing)

- **Conveyor Belt Motor (Optional):**
  - Drive continuous motion for waste input
  - Simple on/off or speed control

- **LED Indicators (Optional but recommended):**
  - RGB LED strip per bin
  - Status indication:
    - Green: Ready to accept waste
    - Red: Bin full / Not available
    - Blue: Processing (gate open)

### 3.5 Communication Requirements

The hardware must **communicate**:

- **Serial Protocol:** USB Serial Communication (9600 baud)
  - Arduino receives commands from Python controller
  - Arduino sends acknowledgments back
  - Protocol: Simple text format (COMMAND:PAYLOAD\n)
  - Example exchange:
    ```
    Python → Arduino: ROUTE:2
    Arduino → Python: ACK:ROUTE:2
    ```

- **Connection:** USB cable (standard USB-A to micro/mini USB)
  - Plug-and-play, no special drivers needed
  - Auto-detection of COM port

### 3.6 Hardware Response Workflow

**When classification software sends "Route waste to Bin 2":**

1. **Arduino receives:** `ROUTE:2`
2. **Arduino executes:**
   - Enable servo #2 (pin 4)
   - Rotate servo to 90° (OPEN)
   - Wait 3000 ms (waste falls in)
   - Rotate servo to 0° (CLOSE)
   - Update internal gate status
3. **Arduino responds:** `ACK:ROUTE:2`
4. **Python receives:** Confirms waste was routed

---

## 4. PROJECT CONSTRAINTS

### 4.1 Budget Constraints

**Estimated Hardware Budget: $150-400 USD**

| Component | Quantity | Cost | Notes |
|-----------|----------|------|-------|
| Arduino Mega 2560 | 1 | $40-60 | Or Arduino Uno + PCA9685 |
| Servo Motors (SG90/MG90S) | 10 | $50-100 | Standard hobby servos, 5V |
| USB Power Supply | 1 | $20-30 | 5A minimum for 10 servos |
| Servo Driver (PCA9685) | 1 (optional) | $15-20 | If using Arduino Uno |
| IR Sensors | 1-10 | $20-50 | Optional, for presence detection |
| RGB LEDs + Strip | 1 | $15-25 | Optional, for status indication |
| Mechanical frame/enclosure | - | $0-100 | DIY or custom fabrication |
| Wiring/Connectors | - | $10-20 | Dupont wires, 24AWG recommended |
| **Total (Minimal)** | | **$150** | Core system (Arduino + servos + PSU) |
| **Total (Full-featured)** | | **$400** | Includes sensors, LEDs, enclosure |

**Software Budget:** $0 (Open source)
- PyTorch, OpenCV, Ultralytics: Free and open-source

### 4.2 Size Constraints

**Physical Footprint:**
- **Bin System Dimensions:** Depends on bin volume (scalable)
  - Typical: 4×2.5×2 feet (depth × width × height) for 10 bins
  - Can be compact or expanded based on waste volume

- **Arduino Control Box:** ~6×4×3 inches
  - Contains Arduino, servo driver, power distribution
  - Can be wall-mounted or integrated into main frame

- **Camera Setup:** Depends on placement
  - Overhead mounted or angled at ~45° above conveyor
  - Typical: 12-24 inches above waste input zone

### 4.3 Power Constraints

**Power Supply Requirements:**

- **Arduino Logic:** ~100 mA @ 5V
- **Servo Motors (TorqueLoad):** 2A each @ 5V (under torque)
  - **Peak power:** 10 servos × 2A = **20A worst-case**
  - **Typical average:** 3-5A (not all servos active simultaneously)

**Recommended Power Supply:**
- Input: 100-240V AC (mains)
- Output: 5V @ 10-15A minimum
- Type: Industrial-grade redundant PSU (recommended for safety)

**Optional Backup:**
- UPS (Uninterruptible Power Supply) 500W-1000W
- Ensures system completes current waste routing if power dips
- Prevents incomplete gate operations

### 4.4 Connectivity & Communication Needs

- **Primary:** USB Serial (9600 baud)
  - PC/Edge device connects to Arduino via USB
  - Range: Limited to ~10-15 feet (USB cable length)
  - No wireless communication (avoids latency/reliability issues)

- **Optional Extensions:**
  - **WiFi Module (ESP8266/ESP32):** For remote operation
  - **4G LTE Modem:** For remote telemetry and monitoring
  - **Local Network:** Python controller can be on different PC on LAN

- **Data Connectivity:**
  - MongoDB Atlas (cloud) or Local MongoDB instance
  - SQLite (local file-based, no network needed)
  - PowerBI / Cloud Dashboard (optional, for analytics)

### 4.5 Operational Constraints

**Processing Latency:**
- Image capture → YOLOv8 detection: 100-200 ms
- Crop extraction + CNN classification: 50-100 ms
- Ensemble fusion + decision: 10-20 ms
- **Total decision time:** ~200-300 ms (sub-second)
- **Hardware actuation time:** 3-4 seconds per waste item

**Throughput:**
- Maximum items processed: ~2-4 items/minute (realistically)
  - Limited by hardware gate operations (3-4 sec hold time)
  - Conveyor belt speed

**Reliability & Safety:**
- Must handle edge cases:
  - Unclassifiable waste → Route to "Trash" bin
  - Confidence score below threshold → Default to "Other"
  - Hardware failures → Failsafe to all gates closed
  - Arduino communication timeout → Abort routing, manual intervention

**Environmental:**
- Temperature range: 10°C - 40°C (typical office/facility)
- Humidity: <80% RH (electronics)
- Dust/moisture protection: Sealed electronics enclosure recommended

---

## 5. INTEGRATION POINTS (Software ↔ Hardware)

### 5.1 Software → Hardware Data Flow

```
┌──────────────────────────────────────────────┐
│   Python/main.py Classification Pipeline      │
│   1. Load image (camera/file)                │
│   2. YOLO Detection (640×640)                │
│   3. CNN Classification (224×224 crops)      │
│   4. Ensemble Fusion (confidence voting)     │
└──────────────────┬───────────────────────────┘
                   │ waste_category = "plastic"
                   │ confidence = 0.87
                   ▼
┌──────────────────────────────────────────────┐
│   ArduinoController (hardware/controller.py)  │
│   1. Lookup bin_id for category              │
│   2. Build serial command: ROUTE:6           │
│   3. Send to Arduino                         │
│   4. Wait for ACK response                   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
           ┌──────────────┐
           │   Arduino    │
           │  Firmware    │
           └──────┬───────┘
                  │ ROUTE:6
                  ▼
         ┌─────────────────┐
         │ Enable Servo #6 │
         │ Open gate 90°   │
         │ Wait 3000 ms    │
         │ Close gate 0°   │
         └─────────────────┘
                  │
                  │ ACK:ROUTE:6
                  ▼
        ┌──────────────────────┐
        │ ArduinoController    │
        │ Receives ACK         │
        │ Route confirmed ✓    │
        └──────────────────────┘
```

### 5.2 Hardware → Software Data Flow

```
Arduino Status Packet (every N seconds or on command)
        ↓
Serial Input: "STATUS: {servo_angles: [0,0,0,90,0,...], errs: []}"
        ↓
Python receives + parses
        ↓
Database Update (logs gate status, failures)
        ↓
Web Dashboard Display or error alerts
```

### 5.3 Simulation Mode

For development without real hardware:
```python
# Usage: python main.py --camera --simulate-hw
```
- Simulates Arduino responses
- No USB connection required
- Full pipeline testing on PC
- File: `hardware/simulator.py`

---

## 6. TECHNOLOGY STACK

**Software Layer:**
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Ultralytics YOLOv8
- **Computer Vision:** OpenCV
- **Web Interface:** Flask
- **Database:** MongoDB or SQLite
- **Hardware Interface:** pyserial

**Hardware Layer:**
- **Microcontroller:** Arduino Mega 2560 (primary) or Arduino Uno + PCA9685
- **Actuators:** 10× SG90/MG90S servo motors (5V)
- **Sensors:** IR proximity (optional), load cells (optional)
- **Communication:** USB serial @ 9600 baud
- **Power:** 5V regulated supply (10-15A)

**Development Tools:**
- Git, pytest, ruff (linting/formatting)
- GPU support: CUDA 12.1 (optional, for training)
- VS Code / PyCharm (IDEs)

---

## 7. KEY FILES & MODULES

| File/Folder | Purpose |
|------------|---------|
| `main.py` | End-to-end pipeline orchestrator |
| `app.py` | Flask web interface (live camera feed) |
| `YoloV8/` | Object detection model & inference |
| `CNN/` | Fine-grained classification & training |
| `classifier/ensemble.py` | Confidence-adaptive voting fusion |
| `hardware/controller.py` | Arduino serial communication |
| `hardware/bin_config.py` | Physical bin layout & pin mapping |
| `hardware/protocol.py` | Serial message protocol |
| `hardware/firmware/` | Arduino C++ firmware |
| `config/config.yaml` | Centralized configuration |
| `database/` | MongoDB/SQLite backend abstraction |
| `capture/camera.py` | Webcam image acquisition |

---

## 8. DEPLOYMENT SCENARIO

**Typical Installation at Waste Collection Facility:**

1. **Camera mounted above conveyor** → Captures waste images in real-time
2. **Python controller** runs on edge PC (local GPU/CPU)
3. **Arduino Mega** mounted in control box beside system
4. **10 servo-controlled bin gates** open/close based on classification
5. **Optional sensors** provide feedback (full bin alerts, etc.)
6. **Database** records all classifications for analytics
7. **Web dashboard** accessible by facility staff for monitoring

**Typical Processing Flow:**
- Waste item enters → Camera captures → Software classifies (300 ms decision)
- Arduino routes to bin → Gate opens (3-4 sec) → Waste sorted
- Staff empties bins periodically
- PowerBI dashboard shows waste distribution statistics

---

## 9. FUTURE ENHANCEMENTS

- [ ] 3D depth sensor for waste volume estimation
- [ ] Mobile app for remote monitoring
- [ ] Cloud-based model updates via OTA (Over-The-Air)
- [ ] Multi-camera setup for conveyor redundancy
- [ ] RFID tracking for contamination detection
- [ ] SMS/Email alerts for system failures
- [ ] Real-time weight per category tracking
- [ ] Predictive maintenance (motor wear, servo health)

---

## 10. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Software Purpose** | Classify waste into 10 categories using YOLOv8 + CNN ensemble |
| **Hardware Purpose** | Sense waste items, route via motorized servo gates into 10 bins |
| **Connectivity** | USB serial (9600 baud) between PC and Arduino Mega |
| **Budget** | $150-400 (hardware), software is free/open-source |
| **Frame Size** | ~4×2.5×2 feet (depends on bin volume) |
| **Power** | 5V @ 10-15A regulated supply |
| **Classification Latency** | ~300 ms per item |
| **Routing Latency** | ~3-4 seconds per item (hardware operation) |
| **Throughput** | 2-4 items/minute (depending on conveyor speed) |
| **Model Accuracy** | CNN: 96.31%, YOLO: trained, Ensemble: optimal weighted voting |
| **Deploy Target** | Edge PC + Arduino at waste facility |

