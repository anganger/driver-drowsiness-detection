import argparse
import random
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm

# NEW TASKS API IMPORTS
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45
    )
    LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    MP_OK = True
except ImportError:
    MP_OK = False
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)
except Exception as e:
    print(f"ERROR Initializing Tasks API: {e}")
    print("Ensure 'face_landmarker.task' is in the project folder.")
    sys.exit(1)

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

MAR_VERTICAL_PAIRS = [(82, 87), (13, 14), (312, 317)]
MAR_HORIZONTAL = (61, 291)
OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

def compute_mar(landmarks, w: int, h: int) -> float:
    def px(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])
    try:
        verts = [np.linalg.norm(px(a) - px(b)) for a, b in MAR_VERTICAL_PAIRS]
        horiz = np.linalg.norm(px(MAR_HORIZONTAL[0]) - px(MAR_HORIZONTAL[1]))
        return float(np.mean(verts) / horiz) if horiz > 1 else 0.0
    except Exception:
        return 0.0

def crop_mouth(frame_bgr, landmarks, w, h, pad=0.35, size=224):
    try:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                        for i in OUTER_LIP_IDX], dtype=np.float32)
        xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
        pw, ph = (xmax - xmin) * pad, (ymax - ymin) * pad
        x1, y1 = max(0, int(xmin - pw)), max(0, int(ymin - ph))
        x2, y2 = min(w, int(xmax + pw)), min(h, int(ymax + ph))
        if x2 - x1 < 8 or y2 - y1 < 8: return None
        crop = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return None

def session_label_from_stem(stem: str) -> str:
    while stem.lower().endswith((".avi", ".mp4")): stem = Path(stem).stem
    last = stem.split("-")[-1].lower()
    if "yawning" in last and "talking" in last: return "mixed"
    if "yawning" in last or "yawn" in last: return "yawning"
    if "talking" in last or "singing" in last: return "talking"
    if "normal" in last: return "normal"
    return "unknown"

def get_frame_label(mar, session_label, mar_yawn, mar_talking, mar_yawn_session=0.25) -> str | None:
    """
    TWO-THRESHOLD STRATEGY — fixes the 13.6x class imbalance:

    mar_yawn_session=0.25  →  used ONLY for Mirror videos with session="yawning"
        The filename already confirms this is a yawning session, so we TRUST it.
        Only reject frames where mouth is clearly shut (MAR < 0.25).
        This recovers partial yawns and yawn build-up frames (MAR 0.25-0.40)
        that the old single threshold was wrongly discarding.

    mar_yawn=0.40  →  used for Dash/mixed/unknown where there is NO session label.
        No ground truth available, so stricter MAR gate is correct here.

    OLD bug: using mar_yawn=0.40 for EVERYTHING meant confirmed yawning-session
    frames with MAR 0.25-0.40 were thrown away — causing the 13.6x imbalance.
    """
    if session_label == "yawning":
        # Trust the filename label. Loose gate = only reject clearly closed mouths.
        return "yawn" if mar > mar_yawn_session else None

    elif session_label in ("talking", "normal"):
        return "talking" if mar <= mar_talking else None

    else:  # mixed or unknown (all Dash videos)
        if mar > mar_yawn:
            return "yawn"
        elif mar <= mar_talking:
            return "talking"
        else:
            return None  # ambiguous middle zone — discard

def scan_videos(root: Path) -> list[dict]:
    records = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS: continue
        parts = p.relative_to(root).parts
        camera = parts[0].lower() if len(parts) > 0 else "unknown"
        stem = p.stem
        while stem.lower().endswith((".avi", ".mp4")): stem = Path(stem).stem
        records.append({
            "path": p, "camera": camera, "stem": stem,
            "subject": stem.split("-")[0] if "-" in stem else stem,
            "session": session_label_from_stem(stem)
        })
    return records

def subject_aware_split(records, val_ratio, test_ratio, seed):
    by_group = defaultdict(list)
    for r in records: by_group[(r["camera"], r["subject"])].append(r)
    groups = sorted(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test
    train_g = set(groups[:n_train])
    val_g   = set(groups[n_train:n_train + n_val])
    test_g  = set(groups[n_train + n_val:])
    train = [r for r in records if (r["camera"], r["subject"]) in train_g]
    val   = [r for r in records if (r["camera"], r["subject"]) in val_g]
    test  = [r for r in records if (r["camera"], r["subject"]) in test_g]
    # Print split summary
    from collections import Counter
    sess = Counter(r["session"] for r in records)
    print(f"\n  Videos found: {len(records)}")
    print(f"  Session breakdown: {dict(sess)}")
    print(f"  Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return train, val, test

def process_video(rec, every_n, mar_yawn, mar_talking, mar_yawn_session,
                  img_size, max_per_video, dry_run):
    stats = defaultdict(int)
    crops = []
    cap = cv2.VideoCapture(str(rec["path"]))
    if not cap.isOpened(): return {"open_error": 1}, []
    session, frame_i, saved = rec["session"], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_per_video: break
        if frame_i % every_n != 0: frame_i += 1; continue
        frame_i += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = LANDMARKER.detect(mp_image)
        if not result.face_landmarks:
            stats["no_face"] += 1; continue
        lms = result.face_landmarks[0]
        mar = compute_mar(lms, w, h)
        label = get_frame_label(mar, session, mar_yawn, mar_talking, mar_yawn_session)
        if label is None: stats["discarded"] += 1; continue
        crop = crop_mouth(frame, lms, w, h, size=img_size)
        if crop is None: stats["crop_fail"] += 1; continue
        stats[label] += 1
        saved += 1
        if not dry_run: crops.append((label, crop))
    cap.release()
    return dict(stats), crops

def main():
    parser = argparse.ArgumentParser(description="YawDD Frame Extractor — dual-threshold fix")
    parser.add_argument("--root",             required=True)
    parser.add_argument("--outdir",           default="yawdd_processed")
    parser.add_argument("--every_n",          type=int,   default=5)
    parser.add_argument("--mar_yawn",         type=float, default=0.40,
                        help="MAR threshold for Dash/mixed/unknown videos (default 0.40)")
    parser.add_argument("--mar_yawn_session", type=float, default=0.25,
                        help="Loose MAR gate for confirmed Mirror/Yawning sessions (default 0.25)")
    parser.add_argument("--mar_talking",      type=float, default=0.50,
                        help="MAR threshold to confirm talking frames (default 0.50)")
    parser.add_argument("--img_size",         type=int,   default=224)
    parser.add_argument("--max_per_video",    type=int,   default=300)
    parser.add_argument("--val_ratio",        type=float, default=0.15)
    parser.add_argument("--test_ratio",       type=float, default=0.15)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--dry_run",          action="store_true")
    args = parser.parse_args()

    root, outdir = Path(args.root), Path(args.outdir)
    if not root.exists():
        print(f"ERROR: {root} not found"); sys.exit(1)

    print("\n" + "="*60)
    print("  YawDD FRAME EXTRACTOR — dual-threshold fix")
    print("="*60)
    print(f"  MAR yawn (unlabeled) : {args.mar_yawn}")
    print(f"  MAR yawn (session)   : {args.mar_yawn_session}  ← KEY FIX for imbalance")
    print(f"  MAR talking          : {args.mar_talking}")
    print(f"  Max crops/video      : {args.max_per_video}")
    print(f"  Every N frames       : {args.every_n}")

    records = scan_videos(root)
    if not records:
        print("ERROR: No videos found!"); sys.exit(1)

    train_vids, val_vids, test_vids = subject_aware_split(
        records, args.val_ratio, args.test_ratio, args.seed
    )

    if not args.dry_run:
        for s in ["train", "val", "test"]:
            for l in ["yawn", "talking"]:
                (outdir / s / l).mkdir(parents=True, exist_ok=True)

    global_stats  = defaultdict(int)
    file_counters = defaultdict(int)
    t0 = time.time()

    for split_name, split_vids in [("train", train_vids),
                                    ("val",   val_vids),
                                    ("test",  test_vids)]:
        if not split_vids: continue
        print(f"\n  [{split_name.upper()}] {len(split_vids)} videos ...")
        for rec in tqdm(split_vids, desc=f"  {split_name:5s}", ncols=80):
            stats, crops = process_video(
                rec,
                every_n          = args.every_n,
                mar_yawn         = args.mar_yawn,
                mar_talking      = args.mar_talking,
                mar_yawn_session = args.mar_yawn_session,
                img_size         = args.img_size,
                max_per_video    = args.max_per_video,
                dry_run          = args.dry_run,
            )
            for k, v in stats.items(): global_stats[k] += v
            if not args.dry_run:
                for label, crop in crops:
                    key  = (split_name, label)
                    name = f"{split_name}_{label}_{file_counters[key]:06d}.jpg"
                    cv2.imwrite(
                        str(outdir / split_name / label / name),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    file_counters[key] += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE  ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\n  Frame stats:")
    for k, v in sorted(global_stats.items()):
        print(f"    {k:<15}: {v:,}")
    if not args.dry_run:
        print(f"\n  Saved crops:")
        for split in ["train", "val", "test"]:
            for label in ["yawn", "talking"]:
                n = file_counters.get((split, label), 0)
                print(f"    {split}/{label:<8}: {n:,}")
        yawn_n    = sum(file_counters.get((s,"yawn"),    0) for s in ["train","val","test"])
        talk_n    = sum(file_counters.get((s,"talking"), 0) for s in ["train","val","test"])
        if yawn_n > 0 and talk_n > 0:
            ratio = max(yawn_n, talk_n) / min(yawn_n, talk_n)
            print(f"\n  Class ratio: {ratio:.1f}x  ", end="")
            if ratio <= 5:   print("✅ Good — train now")
            elif ratio <= 10: print("⚠️  Acceptable — WeightedSampler handles this")
            else:             print("❌ Still bad — lower --mar_yawn_session to 0.20")
    print(f"\n  NEXT: python src_ydd/verify_crops.py --outdir {outdir}")

if __name__ == "__main__":
    main()