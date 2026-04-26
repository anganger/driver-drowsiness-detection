# 🚗 AI-Powered Real-Time Driver Drowsiness Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview
Drowsy driving is a critical safety hazard. This project implements a high-performance monitoring system that utilizes **Deep Learning** and **Computer Vision** to detect signs of driver fatigue (sustained eye closure and frequent yawning). 

Developed as a Final Year Project at **Forman Christian College**, this system achieves a balance between high accuracy and real-time efficiency, optimized specifically for CPU-based inference.

---

## 🚀 Key Features
- **Hybrid Detection Logic**: Combines geometric calculations (**EAR** - Eye Aspect Ratio & **MAR** - Mouth Aspect Ratio) with **MobileNetV2** deep learning classification.
- **Edge-Optimized**: Designed to run at ~30 FPS on consumer-grade CPUs (Intel i5-13500H).
- **Phase-Based Training**: Implemented a "Warmup & Unfreeze" strategy to achieve **96.9% Validation Accuracy**.
- **Audio-Visual Alerts**: Integrated persistent alarms using `pygame` to wake the driver upon detection.

---

## 🛠️ Skills & Technologies
- **Deep Learning**: Transfer Learning, MobileNetV2, Thermal-aware training management.
- **Computer Vision**: Mediapipe Face Mesh (468 landmarks), OpenCV, Grayscale Normalization.
- **Backend**: Python, PyTorch, Virtual Environments (venv).
- **Frontend (Planned)**: React/Next.js dashboard for driver analytics.

---

## 🧠 How It Works
The system follows a three-stage validation pipeline to minimize false positives (like blinking):
1. **Face Mesh Tracking**: Mediapipe identifies the eye and mouth regions.
2. **Geometric Gating**: The system calculates EAR and MAR. If thresholds are exceeded (e.g., mouth opens wide), the DL model is triggered.
3. **Neural Validation**: The MobileNetV2 model classifies the region. If the model confirms a "Yawn" or "Closed Eye" state for more than $N$ consecutive frames, an alert sounds.



---

## 📂 Project Structure
```text
├── models/               # Trained .pth weights for Eyes and Yawns
├── inference.py          # Main real-time execution script
├── face_landmarker.task  # MediaPipe landmark model
├── requirements.txt      # Project dependencies (CPU-optimized)
└── README.md             # Project documentation