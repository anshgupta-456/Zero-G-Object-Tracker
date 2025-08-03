# 🛰️ Zero-G Object Tracker 🚀

A real-time object detection and monitoring system for space stations.  
Built using **YOLOv8** to track **Fire Extinguishers**, **Toolboxes**, and **Oxygen Tanks**, this system ensures astronaut safety by detecting misplaced or floating tools in microgravity.

Hackathon: **Build With India 2.0 – Space Station Track**  

📥 **[Download Dataset & Weights](https://drive.google.com/drive/folders/1IMfpSRCXRlWwtBeAVJCUMP9yhHAqYUXz?usp=sharing)**


## 🌌 Why This Project?

In microgravity, tools and equipment can float, become misplaced, or cause safety hazards. Manual monitoring is inefficient.

**Zero-G Object Tracker** solves this with:
- 🧠 Real-time detection using a YOLOv8 model
- 🛑 Rule-based safety alerts (e.g., missing extinguisher)
- 📊 Performance visualization (mAP, confusion matrix)
- 📋 Object loss logging for audit/compliance

---

## 📦 Project Structure

├── data/
│ ├── train/ # Training images + YOLO labels
│ ├── val/ # Validation images + labels
│ └── test/ # Final testing
├── runs/ # Model weights (YOLOv8 output)
├── predictions/ # Results saved after prediction
├── train.py # Model training
├── predict.py # Model evaluation + prediction
├── visualize.py # Annotation visualizer
├── yolo_params.yaml # Dataset path config
└── README.md # You're here

---

## 🧠 Model Details

| Attribute        | Value                  |
|------------------|------------------------|
| Architecture     | YOLOv8s (Ultralytics)  |
| Epochs           | 20                     |
| mAP@0.5          | **85.3%**              |
| Classes          | FireExtinguisher, Toolbox, OxygenTank |
| Optimizer        | AdamW                  |
| LR (start → end) | 0.001 → 0.0001         |
| Device           | CPU (no GPU used)      |
| Augmentation     | Mosaic = 0.1           |

---

## 🔧 How to Use

### 1️⃣ Train the Model

```bash
python train.py --epochs 20 --optimizer AdamW --lr0 0.001 --lrf 0.0001 --momentum 0.2 --mosaic 0.1
Use:

d / a — navigate forward/backward

t / v — switch between train/val modes

🛰️ Zero-G Tracker Features
✅ Object Detection
Live bounding boxes for the 3 key classes

Confidence scores shown

🛡️ Safety Rule Engine (customizable)
Oxygen Tank must always be visible

Fire Extinguisher must be fixed to wall (not floating)

Toolbox should be stationary, not drifting

📋 Object Loss Logging
Detects if an object is missing from N frames

Generates logs with time, class, and last seen location

📊 Performance Dashboard (future work)
Confusion matrix

Precision-recall curves

Detection confidence over time

🧪 Results
Metric	Value
mAP@0.5	85.3%
Precision (avg)	~83%
Recall (avg)	~85%
Inference Device	CPU

🔻 Include your result images, confusion matrix, and sample detections here.
Camera / Video Feed
        │
        ▼
YOLOv8 Model (best.pt)
        │
        ▼
Detection Logic → Rule Checker → Logs + Alerts
        │
        ▼
(Optional) Streamlit Dashboard
 Future Work
 Add object tracking (e.g., DeepSORT)

 Deploy on Jetson Nano or edge device

 Enable real-time webcam + audio alerts

 Streamlit interface for live dashboard

📝 Hackathon Relevance
This project directly addresses:

🔐 Astronaut safety

🧰 Tool visibility and inventory tracking


🧠 AI-based automation aboard space missions
