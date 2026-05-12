# 🚗 Real-Time Driver Drowsiness Detection System (GuardianAI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009485.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 The Problem
Driver fatigue is a critical global safety issue, linked to roughly 20% of road accidents and over 6,000 fatal crashes annually. Shockingly, 1 in 25 adult drivers report falling asleep while driving. Current vehicles often lack intelligent systems to detect this in real-time. 

**GuardianAI** is a non-intrusive, affordable, and highly reliable computer-vision-based system designed to monitor driver state using a standard webcam.

## 🛠️ Achievements & Engineering
Developed as a **Final Year Project (FYP)** at **Forman Christian College (A Chartered University)**, this system was engineered to run on a **CPU-only laptop** without a dedicated GPU. 
*   **High Accuracy**: The Eye State model achieved **98.65% validation accuracy** using a two-phase fine-tuning strategy on the MobileNetV2 architecture.
*   **CPU Optimization**: Utilizes smart frame skipping (AI runs every 3rd frame) and resolution downsampling (320x240) to achieve a consistent **24+ FPS** on standard consumer hardware.

---

## 🧠 The Pipeline (IoT Architecture)
The system has evolved from a local script into a full-stack monitoring product:

1.  **Backend (FastAPI)**: A high-performance server that captures webcam frames and executes the "Double-Lock" AI logic.
2.  **Inference Engine**:
    *   **MediaPipe**: Extracts 468 landmarks to calculate Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
    *   **CNN Classifiers**: Dual fine-tuned MobileNetV2 models confirm if eyes are closed and if a wide yawn is occurring.
3.  **Real-Time Dashboard**: Metadata is streamed via **WebSockets** to a modern, dark-themed UI built with **Tailwind CSS**.
4.  **Autonomous Notifications**: When fatigue is confirmed, the system automatically dispatches an emergency email report via **EmailJS** to a supervisor.

---

## ⚖️ Hybrid Risk Logic (7s/2m Rule)
To minimize false positives from natural blinking or talking, GuardianAI uses strict temporal thresholds:

*   **7-Second Eye Rule**: A critical alert is triggered only if the driver's eyes remain closed for **7.0 consecutive seconds**.
*   **2-Minute Yawn Window**: A fatigue alert is triggered if **2 or more yawns** are detected within a **2-minute sliding window**.
*   **Double-Lock Verification**: The system requires the AI model's prediction to be backed by physical geometric measurements (EAR/MAR) before escalating the risk score.

---

## 📊 Datasets Used
*   **MRL Eye Dataset**: Nearly 85,000 images of open/closed eyes used for high-precision eye-state classification.
*   **YawDD (Yawning Detection Dataset)**: 351 videos used to differentiate between normal talking and genuine yawning.

---

## 📂 Repository Structure
```text
├── models/                  # Final trained .pth weights (eye_model.pth, yawn_model.pth)
├── training code files/     # Original processing and model training scripts
├── app.py                   # FastAPI Backend & WebSocket Server
├── index.html               # Professional GuardianAI Dashboard (Frontend)
├── face_landmarker.task     # MediaPipe facial landmark asset
├── requirements.txt         # CPU-optimized dependencies
└── readme.md                # Project documentation