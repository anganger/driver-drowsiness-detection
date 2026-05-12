import cv2
import json
import time
import base64
import torch
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from torchvision import models, transforms
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# ══════════════════════════════════════════════════════════════════
# STABILITY GATES & INDICES
# ══════════════════════════════════════════════════════════════════
EAR_THRESHOLD = 0.21      
MAR_THRESHOLD = 0.52      
EYE_CONF_GATE = 0.60      
YAWN_CONF_GATE = 0.85     

LEFT_EYE_ALL  = [33,160,158,133,153,144,163,7,246,161,159,157,173,155,154,145]
OUTER_LIP     = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class DrowsinessRiskManager:
    def __init__(self):
        self.eye_closed_start = None
        self.yawn_timestamps = [] 
        self.critical_alert = False
        self.reason = ""

    def calculate(self, eye_confirmed, yawn_confirmed):
        now = time.time()
        self.critical_alert = False
        self.reason = ""

        if eye_confirmed:
            if self.eye_closed_start is None: self.eye_closed_start = now
            elapsed_eye = now - self.eye_closed_start
            if elapsed_eye >= 7.0:
                self.critical_alert = True
                self.reason = "Critical: Eyes Closed > 7s"
        else:
            self.eye_closed_start = None
            elapsed_eye = 0

        if yawn_confirmed and (not self.yawn_timestamps or now - self.yawn_timestamps[-1] > 6):
            self.yawn_timestamps.append(now)

        self.yawn_timestamps = [t for t in self.yawn_timestamps if now - t <= 120]
        yawn_count = len(self.yawn_timestamps)
        
        if yawn_count >= 2:
            self.critical_alert = True
            self.reason = f"Fatigue: {yawn_count} Yawns in 2min"

        eye_score = min(100, (elapsed_eye / 7.0) * 100) if eye_confirmed else 0
        yawn_score = 50 if yawn_count == 1 else (100 if yawn_count >= 2 else 0)
        risk_score = int(max(eye_score, yawn_score))
        
        return risk_score, float(elapsed_eye), int(yawn_count), bool(self.critical_alert), str(self.reason)

def px(lms, i, w, h): return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float32)

def compute_ear(lms, w, h):
    try:
        v1 = np.linalg.norm(px(lms, 159, w, h) - px(lms, 145, w, h))
        v2 = np.linalg.norm(px(lms, 158, w, h) - px(lms, 153, w, h))
        horiz = np.linalg.norm(px(lms, 33, w, h) - px(lms, 133, w, h))
        return float((v1 + v2) / (2.0 * horiz))
    except: return 0.5

def compute_mar(lms, w, h):
    try:
        v = np.linalg.norm(px(lms, 13, w, h) - px(lms, 14, w, h))
        horiz = np.linalg.norm(px(lms, 61, w, h) - px(lms, 291, w, h))
        return float(v / horiz) if horiz > 1 else 0.0
    except: return 0.0

def crop_region(frame, lms, indices, w, h, pad=0.35):
    try:
        pts = np.array([px(lms, i, w, h) for i in indices])
        xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
        pw, ph = (xmax - xmin) * pad, (ymax - ymin) * pad
        x1, y1, x2, y2 = max(0,int(xmin-pw)), max(0,int(ymin-ph)), min(w,int(xmax+pw)), min(h,int(ymax+ph))
        return cv2.resize(frame[y1:y2, x1:x2], (224, 224))
    except: return None

def load_and_fix_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.last_channel, 2)
    state = {(k[9:] if k.startswith("backbone.") else k): v for k, v in ckpt["model_state"].items()}
    m.load_state_dict(state)
    return m.to(device).eval()

tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = DrowsinessRiskManager()
    cap = cv2.VideoCapture(0)
    device = torch.device("cpu")

    eye_model = load_and_fix_model("models/eye_model.pth", device)
    yawn_model = load_and_fix_model("models/yawn_model.pth", device)

    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1))

    frame_count, skip_frames = 0, 2
    l_eye_label, l_yawn_label = "open", "talking"
    l_risk, l_eye_t, l_yawn_c = 0, 0, 0
    is_crit, reason = False, ""
    l_ear, l_mar = 0.3, 0.1

    try:
        while True:
            t_loop_start = time.perf_counter()
            success, frame = cap.read()
            if not success: break
            
            frame_count += 1
            h, w = frame.shape[:2]

            if frame_count % (skip_frames + 1) == 0:
                small_frame = cv2.resize(frame, (320, 240))
                rgb_small = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(rgb_small)

                eye_confirmed, yawn_confirmed = False, False
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    l_ear, l_mar = compute_ear(lms, 320, 240), compute_mar(lms, 320, 240)
                    
                    eye_crop = crop_region(small_frame, lms, LEFT_EYE_ALL, 320, 240)
                    if eye_crop is not None:
                        with torch.no_grad():
                            probs = F.softmax(eye_model(tf(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)), dim=1)[0]
                            if probs[0].item() > EYE_CONF_GATE and l_ear < EAR_THRESHOLD:
                                eye_confirmed, l_eye_label = True, "closed"
                            else: l_eye_label = "open"

                    if l_mar > MAR_THRESHOLD:
                        mouth_crop = crop_region(small_frame, lms, OUTER_LIP, 320, 240)
                        if mouth_crop is not None:
                            with torch.no_grad():
                                probs = F.softmax(yawn_model(tf(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)), dim=1)[0]
                                if probs[1].item() > YAWN_CONF_GATE:
                                    yawn_confirmed, l_yawn_label = True, "yawn"
                                else: l_yawn_label = "talking"
                    else: l_yawn_label = "talking"

                    l_risk, l_eye_t, l_yawn_c, is_crit, reason = engine.calculate(eye_confirmed, yawn_confirmed)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # CRITICAL: All values forced to Python native types (int, float, str, bool)
            await websocket.send_text(json.dumps({
                "image": f"data:image/jpeg;base64,{img_base64}",
                "eye_label": str(l_eye_label), "yawn_label": str(l_yawn_label),
                "risk": int(l_risk), "eye_time": float(l_eye_t), "yawn_count": int(l_yawn_c),
                "critical": bool(is_crit), "reason": str(reason),
                "ear": float(l_ear), "mar": float(l_mar),
                "fps": int(1 / max(0.001, (time.perf_counter() - t_loop_start)))
            }))
            await asyncio.sleep(0.005)
    except Exception as e: print(f"Error: {e}")
    finally: cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)