"""
inference.py  — fixed thresholds + MAR pre-filter
"""

import argparse
import time
import sys
import os
import collections
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("ERROR: pip install mediapipe"); sys.exit(1)

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False
    print("WARNING: pygame unavailable — visual alerts only.")

# ══════════════════════════════════════════════════════════════════
# TUNABLE THRESHOLDS — adjust these if behaviour is wrong
# ══════════════════════════════════════════════════════════════════
WINDOW_SIZE       = 60   # frames of history (~2 sec at 30fps)

# Eye alert: eyes closed for N consecutive-ish frames
EYE_CLOSED_THRESH = 20   # out of 60 frames  (~0.67 sec sustained)

# Yawn alert: N confident yawn frames in the window
YAWN_COUNT_THRESH = 8    # out of 60 frames

# Yawn model confidence gate — ignore borderline predictions
YAWN_CONF_GATE    = 0.80  # must be ≥80% confident to count as yawn

# MAR pre-filter — skip yawn model entirely if mouth is clearly closed
# Watch the MAR readout; normal talking ≈ 0.05–0.20, yawning ≈ 0.45+
MAR_YAWN_GATE     = 0.35  # only run yawn model if MAR > this value

# ══════════════════════════════════════════════════════════════════
# LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════
LEFT_EYE_ALL  = [33,160,158,133,153,144,163,7,246,161,159,157,173,155,154,145]
RIGHT_EYE_ALL = [362,385,387,263,380,373,388,466,390,249,263,466,388,387,386,374]
LEFT_EYE_V    = [(159,145),(158,153),(160,144)]
LEFT_EYE_H    = (33,133)
RIGHT_EYE_V   = [(386,374),(385,380),(387,373)]
RIGHT_EYE_H   = (362,263)
OUTER_LIP     = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
MAR_V_PAIRS   = [(82,87),(13,14),(312,317)]
MAR_H         = (61,291)

# Colors BGR
GREEN=(0,220,80); RED=(0,60,255); YELLOW=(0,200,255)
WHITE=(255,255,255); BLACK=(0,0,0); CYAN=(255,220,0); ORANGE=(0,140,255)

IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

# ══════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════
def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("config",{}).get("model",{}).get("architecture","mobilenetv2")
    if arch == "mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(m.classifier[1].in_features, 2))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3, inplace=True),
            torch.nn.Linear(m.classifier[1].in_features, 2))
    else:
        raise ValueError(arch)
    # Strip "backbone." prefix saved by EyeStateClassifier wrapper
    state = {(k[9:] if k.startswith("backbone.") else k): v
             for k,v in ckpt["model_state"].items()}
    m.load_state_dict(state)
    m.to(device).eval()
    print(f"  ✅  {arch}  epoch={ckpt.get('epoch','?')}  val_acc={ckpt.get('val_acc','?')}")
    return m

# ══════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════
eye_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
yawn_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ══════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════
def px(lms, i, w, h):
    return np.array([lms[i].x*w, lms[i].y*h], dtype=np.float32)

def compute_ear(lms, v_pairs, h_pair, w, h):
    try:
        verts = [np.linalg.norm(px(lms,a,w,h)-px(lms,b,w,h)) for a,b in v_pairs]
        horiz = np.linalg.norm(px(lms,h_pair[0],w,h)-px(lms,h_pair[1],w,h))
        return float(np.mean(verts)/horiz) if horiz>1 else 0.0
    except: return 0.0

def compute_mar(lms, w, h):
    try:
        verts = [np.linalg.norm(px(lms,a,w,h)-px(lms,b,w,h)) for a,b in MAR_V_PAIRS]
        horiz = np.linalg.norm(px(lms,MAR_H[0],w,h)-px(lms,MAR_H[1],w,h))
        return float(np.mean(verts)/horiz) if horiz>1 else 0.0
    except: return 0.0

def crop_region(frame, lms, indices, w, h, pad=0.4, size=224):
    try:
        pts = np.array([px(lms,i,w,h) for i in indices])
        xmin,ymin = pts.min(0); xmax,ymax = pts.max(0)
        pw=(xmax-xmin)*pad; ph=(ymax-ymin)*pad
        x1=max(0,int(xmin-pw)); y1=max(0,int(ymin-ph))
        x2=min(w,int(xmax+pw)); y2=min(h,int(ymax+ph))
        if x2-x1<8 or y2-y1<8: return None
        return cv2.resize(frame[y1:y2,x1:x2],(size,size))
    except: return None

# ══════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════
def make_alert_sound():
    if not PYGAME_OK: return None
    try:
        sr=44100; n=int(sr*0.6)
        t=np.linspace(0,0.6,n,endpoint=False)
        wave=(np.sin(2*np.pi*880*t)*0.6+np.sin(2*np.pi*1320*t)*0.4)
        wave=(wave*0.9*32767).astype(np.int16)
        stereo=np.column_stack([wave,wave])
        return pygame.mixer.Sound(buffer=stereo.tobytes())
    except Exception as e:
        print(f"  WARNING: Audio failed ({e})")
        return None

# ══════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════
def draw_hud(frame, s):
    h,w = frame.shape[:2]

    # Top bar background
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,90),BLACK,-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)

    # Row 1: FPS / Latency / MAR
    cv2.putText(frame,f"FPS:{s['fps']:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,CYAN,2)
    cv2.putText(frame,f"Lat:{s['latency_ms']:.0f}ms",(110,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,CYAN,2)
    cv2.putText(frame,f"MAR:{s['mar']:.2f}  EAR:{s['ear']:.2f}",
                (w-210,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,CYAN,1)

    # Row 2: Eye status
    ec=RED if s['eye_label']=="closed" else GREEN
    cv2.putText(frame,f"EYE: {s['eye_label'].upper()} ({s['eye_conf']:.0%})",
                (10,44),cv2.FONT_HERSHEY_SIMPLEX,0.6,ec,2)

    # Row 2: Yawn status  
    yc=ORANGE if s['yawn_label']=="yawn" else GREEN
    gate_txt = " [MAR gate]" if s.get('mar_gated') else ""
    cv2.putText(frame,f"MOUTH:{s['yawn_label'].upper()}({s['yawn_conf']:.0%}){gate_txt}",
                (w//2-30,44),cv2.FONT_HERSHEY_SIMPLEX,0.55,yc,2)

    # Progress bars row
    by=60; bh=10; bw=220
    # Eye bar
    ef=int(bw*min(s['eye_closed_frames'],WINDOW_SIZE)/WINDOW_SIZE)
    bcol=RED if s['eye_closed_frames']>=EYE_CLOSED_THRESH else YELLOW
    cv2.rectangle(frame,(10,by),(10+bw,by+bh),(50,50,50),-1)
    cv2.rectangle(frame,(10,by),(10+ef,by+bh),bcol,-1)
    cv2.putText(frame,f"Eye closed: {s['eye_closed_frames']}/{EYE_CLOSED_THRESH}",
                (10,by+bh+12),cv2.FONT_HERSHEY_SIMPLEX,0.4,WHITE,1)

    # Yawn bar
    yf=int(bw*min(s['yawn_count'],YAWN_COUNT_THRESH)/YAWN_COUNT_THRESH)
    ycol=ORANGE if s['yawn_count']>=YAWN_COUNT_THRESH else YELLOW
    cv2.rectangle(frame,(w-bw-10,by),(w-10,by+bh),(50,50,50),-1)
    cv2.rectangle(frame,(w-bw-10,by),(w-bw-10+yf,by+bh),ycol,-1)
    cv2.putText(frame,f"Yawns: {s['yawn_count']}/{YAWN_COUNT_THRESH}",
                (w-bw-10,by+bh+12),cv2.FONT_HERSHEY_SIMPLEX,0.4,WHITE,1)

    # Alert banner
    if s.get('alert_active'):
        alpha=0.55+0.25*abs(np.sin(time.time()*4))
        banner=frame.copy()
        cv2.rectangle(banner,(0,h//2-55),(w,h//2+55),(0,0,180),-1)
        cv2.addWeighted(banner,alpha,frame,1-alpha,0,frame)
        cv2.putText(frame,"DROWSINESS DETECTED",
                    (w//2-195,h//2+8),cv2.FONT_HERSHEY_DUPLEX,1.2,WHITE,3)
        cv2.putText(frame,s.get('alert_reason',''),
                    (w//2-140,h//2+40),cv2.FONT_HERSHEY_SIMPLEX,0.65,YELLOW,2)

    if not s.get('face_detected',True):
        cv2.putText(frame,"No face",(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,YELLOW,2)

    cv2.putText(frame,"Q=quit  S=screenshot  P=pause  R=reset",
                (10,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1)
    return frame

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════
def run(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  DROWSINESS DETECTION — Real-Time Inference")
    print(f"  Script  : {os.path.abspath(__file__)}")
    print(f"  Device  : {device}")
    print(f"  Window  : {WINDOW_SIZE} frames")
    print(f"  EyeThr  : {EYE_CLOSED_THRESH} frames closed → alert")
    print(f"  YawnThr : {YAWN_COUNT_THRESH} frames + conf>{YAWN_CONF_GATE} + MAR>{MAR_YAWN_GATE}")
    print(f"{'='*55}")

    print("\n  Loading eye model...")
    eye_model  = load_model(args.eye_model,  device)
    print("  Loading yawn model...")
    yawn_model = load_model(args.yawn_model, device)

    print("  Initializing MediaPipe...")
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.task_model),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
    )
    print("  MediaPipe ready.")

    alert_sound=make_alert_sound()

    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: camera {args.camera}"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)
    print(f"  Camera {args.camera} open — press Q to quit\n")

    eye_window  = collections.deque(maxlen=WINDOW_SIZE)
    yawn_window = collections.deque(maxlen=WINDOW_SIZE)
    alert_active= False
    paused      = False
    fps_times   = collections.deque(maxlen=30)
    frame_count = 0

    state={
        "fps":0,"latency_ms":0,
        "eye_label":"N/A","eye_conf":0,
        "yawn_label":"talking","yawn_conf":0,
        "ear":0.0,"mar":0.0,"mar_gated":False,
        "eye_closed_frames":0,"yawn_count":0,
        "alert_active":False,"alert_reason":"","face_detected":False,
    }

    while True:
        t0=time.perf_counter()
        ret,frame=cap.read()
        if not ret: print("Frame error"); break
        frame_count+=1

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('p'):
            paused=not paused
            print("Paused" if paused else "Resumed")
        elif key==ord('r'):
            eye_window.clear(); yawn_window.clear()
            alert_active=False; print("Reset")
        elif key==ord('s'):
            ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{ts}.jpg",frame)
            print(f"Screenshot saved")

        if paused:
            cv2.putText(frame,"PAUSED",(frame.shape[1]//2-60,frame.shape[0]//2),
                        cv2.FONT_HERSHEY_DUPLEX,1.5,YELLOW,3)
            cv2.imshow("Drowsiness Detection",frame); continue

        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))

        face_ok=bool(result.face_landmarks)
        state["face_detected"]=face_ok

        if not face_ok:
            eye_window.append(0); yawn_window.append(0)
        else:
            lms=result.face_landmarks[0]

            # ── EAR — use geometry directly as primary eye signal ────
            left_ear  = compute_ear(lms,LEFT_EYE_V, LEFT_EYE_H, w,h)
            right_ear = compute_ear(lms,RIGHT_EYE_V,RIGHT_EYE_H,w,h)
            avg_ear   = (left_ear+right_ear)/2
            state["ear"]=round(avg_ear,3)

            # ── Eye model inference ───────────────────────────────────
            eye_crop=crop_region(frame,lms,LEFT_EYE_ALL,w,h,pad=0.5)
            if eye_crop is not None:
                eye_rgb=cv2.cvtColor(eye_crop,cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    logits=eye_model(eye_tf(eye_rgb).unsqueeze(0).to(device))
                    probs=F.softmax(logits,dim=1)[0]
                pred=probs.argmax().item()
                conf=probs[pred].item()
                label="open" if pred==1 else "closed"
                state["eye_label"]=label; state["eye_conf"]=conf

                # Combined signal: model says closed AND EAR < 0.20
                # Both must agree to avoid false positives from blinks
                model_closed = (pred==0)
                ear_closed   = (avg_ear < 0.20)
                eye_window.append(1 if (model_closed and ear_closed) else 0)
            else:
                eye_window.append(0)

            # ── MAR ───────────────────────────────────────────────────
            mar=compute_mar(lms,w,h)
            state["mar"]=round(mar,3)

            # ── Yawn model — only run if MAR says mouth is open ───────
            if mar < MAR_YAWN_GATE:
                # Mouth clearly closed — skip model, count as talking
                state["yawn_label"]="talking"
                state["yawn_conf"]=1.0
                state["mar_gated"]=True
                yawn_window.append(0)
            else:
                state["mar_gated"]=False
                mouth_crop=crop_region(frame,lms,OUTER_LIP,w,h,pad=0.35)
                if mouth_crop is not None:
                    mouth_rgb=cv2.cvtColor(mouth_crop,cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        logits=yawn_model(yawn_tf(mouth_rgb).unsqueeze(0).to(device))
                        probs=F.softmax(logits,dim=1)[0]
                    pred=probs.argmax().item()
                    conf=probs[pred].item()
                    state["yawn_label"]="yawn" if pred==1 else "talking"
                    state["yawn_conf"]=conf
                    # Both MAR gate AND model confidence must agree
                    is_yawn=(pred==1 and conf>=YAWN_CONF_GATE and mar>=MAR_YAWN_GATE)
                    yawn_window.append(1 if is_yawn else 0)
                else:
                    yawn_window.append(0)

        # ── Decision logic ────────────────────────────────────────────
        eye_closed_n = sum(eye_window)
        yawn_n       = sum(yawn_window)
        state["eye_closed_frames"]=eye_closed_n
        state["yawn_count"]=yawn_n

        was=alert_active
        if eye_closed_n>=EYE_CLOSED_THRESH:
            alert_active=True
            state["alert_reason"]=f"Eyes closed ({eye_closed_n}/{EYE_CLOSED_THRESH} frames)"
        elif yawn_n>=YAWN_COUNT_THRESH:
            alert_active=True
            state["alert_reason"]=f"Repeated yawning ({yawn_n} detections)"
        else:
            alert_active=False; state["alert_reason"]=""
        state["alert_active"]=alert_active

        if alert_active and not was and alert_sound and PYGAME_OK:
            alert_sound.play(loops=-1)
        elif not alert_active and was and PYGAME_OK:
            pygame.mixer.stop()

        # ── FPS ───────────────────────────────────────────────────────
        t1=time.perf_counter()
        fps_times.append(t1)
        fps=(len(fps_times)-1)/(fps_times[-1]-fps_times[0]) if len(fps_times)>=2 else 0
        state["fps"]=fps; state["latency_ms"]=(t1-t0)*1000

        frame=draw_hud(frame,state)
        cv2.imshow("Drowsiness Detection — Q to quit",frame)

        if frame_count%100==0:
            print(f"  F{frame_count:5d} | FPS={fps:.1f} lat={state['latency_ms']:.0f}ms | "
                  f"EAR={state['ear']:.2f} MAR={state['mar']:.2f} | "
                  f"eye={state['eye_label']} yawn={state['yawn_label']} | "
                  f"alert={'YES' if alert_active else 'no'}")

    cap.release(); cv2.destroyAllWindows()
    if PYGAME_OK: pygame.mixer.quit()
    landmarker.close()
    print(f"\n  Done. {frame_count} frames.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--eye_model",  required=True)
    p.add_argument("--yawn_model", required=True)
    p.add_argument("--task_model", default="face_landmarker.task")
    p.add_argument("--camera",     type=int, default=0)
    args=p.parse_args()
    for name,path in [("Eye",args.eye_model),("Yawn",args.yawn_model),("Task",args.task_model)]:
        if not Path(path).exists():
            print(f"ERROR: {name} not found: {path}"); sys.exit(1)
    run(args)

if __name__=="__main__":
    main()