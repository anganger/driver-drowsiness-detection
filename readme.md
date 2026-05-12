# 🚗 Real-Time Driver Drowsiness Detection System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 The Problem
Driver fatigue is a critical global safety issue. Studies show that 20% of road accidents are caused by drowsy driving, and it is linked to over 6,000 fatal crashes per year. Shockingly, 1 in 25 adult drivers report falling asleep while driving. 

Current vehicles lack intelligent systems that can detect driver fatigue in real time, and existing solutions are either too expensive, intrusive, or inaccurate. 

**The Goal:** To build a non-contact, affordable, and highly reliable computer vision-based detection system using only a standard webcam.

---

## 🛠️ What I Achieved (CPU-Only Engineering)
A major constraint and personal challenge for this project was developing and training Deep Learning models **without a dedicated NVIDIA GPU**. 

I successfully processed gigabytes of data and fine-tuned a **MobileNetV2** architecture entirely on an **Intel Core i5-13500H CPU**. By utilizing strict thermal management, phase-based training (Warmup & Unfreeze), and optimized PyTorch data loaders, the model achieved **96.91% validation accuracy**. The final inference script runs flawlessly in real-time at 30+ FPS on consumer-grade hardware.

---

## 🧠 How It Works (The Pipeline)

The system operates via a real-time, four-step pipeline:

1. **Face & Landmark Extraction**: Using the MediaPipe Face Mesh, the system extracts 468 facial landmarks per frame in real time to locate the exact coordinates of the eyes and mouth.
2. **Geometric Filtering**: Before running heavy neural networks, the system calculates the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to monitor physical distances.
3. **CNN Classification**: If the geometric thresholds indicate potential drowsiness (e.g., mouth open wide or eyes low), the specific eye or mouth region is cropped and passed to the fine-tuned CNN classifiers to confirm the state (Open/Closed or Talking/Yawning).
4. **Temporal Decision Logic**: The system maintains a sliding window of recent frames (e.g., 30 frames / 1 second). A persistent audio/visual alert is only triggered if the "closed" or "yawning" state is sustained over multiple frames, effectively filtering out false positives like natural blinking.

---

## 📊 Datasets Used
To train the models to recognize fatigue in diverse environments, two primary datasets were utilized:
* **MRL Eye Dataset**: Used for training the eye-state classifier. It contains nearly 85,000 images of open and closed eyes across 37 different subjects under various lighting conditions.
* **YawDD (Yawning Detection Dataset)**: Comprises 351 videos of diverse subjects talking, driving normally, and yawning. Frames were extracted via OpenCV to train the binary mouth-state CNN.

---

## 📂 Repository Structure

This repository contains the full project lifecycle, from raw training scripts to the optimized deployment app.
```text
├── models/                                      # Final trained .pth weights for real-time inference
├── training code files/                         # Source code containing data processing and model training scripts
├── drowsiness_detection proposal presentation/  # Original project pitch deck and slide visuals
├── face_landmarker.task                         # MediaPipe asset for facial landmark tracking
├── inference.py                                 # The main real-time webcam monitoring application
├── requirements.txt                             # CPU-optimized Python dependencies
└── readme.md                                    # Project documentation
```
## 🚀 How to Run the Model

To run this project on your local machine, you need to set up a Python virtual environment and install the required dependencies. The inference script is optimized to run on standard CPUs without requiring an NVIDIA GPU.

### Step 1: Clone the Repository
Open your terminal and clone this project to your machine:
```bash
git clone [https://github.com/anganger/driver-drowsiness-detection.git](https://github.com/anganger/driver-drowsiness-detection.git)
cd driver-drowsiness-detection
```

### Step 2: Create a Virtual Environment
It is highly recommended to use Python 3.11 or 3.12 for library compatibility.

**For Windows (PowerShell):**
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Requirements
Once your virtual environment is active `(venv)`, install the dependencies. The `requirements.txt` is configured to download the lightweight, CPU-only versions of PyTorch to save space and memory.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Inference Script
Make sure your webcam is available, then execute the main script using the pre-trained weights:
```powershell
python inference.py `
  --eye_model "models/eye_model.pth" `
  --yawn_model "models/yawn_model.pth" `
  --task_model "face_landmarker.task"
```
*(Note: If you are on Mac/Linux, replace the backticks \` with backslashes \\ to break the command across lines).*

### ⌨️ In-App Controls
* **Q**: Quit the application.
* **P**: Pause/Resume the video feed.
* **R**: Reset the drowsiness counter and silence active alerts.
* **S**: Save a screenshot of the current frame.