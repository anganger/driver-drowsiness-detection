"""
explore_yawdd.py  (v2 — PDF-informed)
──────────────────────────────────────
STEP 1 — Run this FIRST. Reads your actual filenames and tells you
which labeling strategy applies to each folder.

KEY INSIGHT FROM README + TABLE PDFs:
  ydd_dataset/
    Dash/          ← Table 2: 29 videos, ONE video per subject,
                     ALL conditions mixed inside (Normal + Talking + Yawning)
                     → Cannot label by filename. Use MAR per-frame.

    Mirror/        ← Table 1: 322 videos, SEPARATE video per condition.
                     Filename contains the action suffix:
                       "1-FemaleNoGlasses-Normal.avi"
                       "1-FemaleNoGlasses-Talking.avi"
                       "1-FemaleNoGlasses-Yawning.avi"
                     → Label directly from filename suffix.

Usage:
    python src/explore_yawdd.py --root ydd_dataset
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

# Action keywords we look for in Mirror filenames
YAWN_KEYWORDS    = {"yawning", "yawn"}
TALKING_KEYWORDS = {"talking", "singing", "talk"}
NORMAL_KEYWORDS  = {"normal"}
# Talking&Yawning is a mixed action — we handle it specially


def detect_action_from_filename(stem: str) -> str:
    """
    Try to extract the action label from a Mirror video filename.
    Filename examples:
        1-FemaleNoGlasses-Yawning
        3-MaleGlasses-Talking
        12-MalePrescription-Talking&Yawning
        3-FemaleGlasses           ← Dash-style (no action suffix)

    Returns: "yawning" | "talking" | "normal" | "mixed" | "unknown"
    """
    parts = stem.lower().split("-")
    last  = parts[-1] if parts else ""

    if "yawning" in last and "talking" in last:
        return "mixed"
    if any(k in last for k in YAWN_KEYWORDS):
        return "yawning"
    if any(k in last for k in TALKING_KEYWORDS):
        return "talking"
    if any(k in last for k in NORMAL_KEYWORDS):
        return "normal"
    return "unknown"   # Dash-style: no action in name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to ydd_dataset/")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} not found")
        sys.exit(1)

    print("\n" + "="*65)
    print("  YawDD DATASET EXPLORER  (PDF-informed, v2)")
    print("="*65)
    print(f"  Root: {root.resolve()}\n")

    # Collect all videos
    all_videos = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        parts  = p.relative_to(root).parts
        camera = parts[0] if len(parts) > 0 else "?"
        stem   = p.stem
        # Handle double extensions like "8-FemaleGlasses.avi.avi"
        while stem.lower().endswith(".avi") or stem.lower().endswith(".mp4"):
            stem = Path(stem).stem
        action = detect_action_from_filename(stem)
        subj_id = stem.split("-")[0] if "-" in stem else stem

        all_videos.append({
            "path":     p,
            "camera":   camera,
            "filename": p.name,
            "stem":     stem,
            "subject":  subj_id,
            "action":   action,
        })

    print(f"  Total videos: {len(all_videos)}\n")

    # Split by Dash vs Mirror
    dash_vids   = [v for v in all_videos if v["camera"].lower() == "dash"]
    mirror_vids = [v for v in all_videos if v["camera"].lower() == "mirror"]

    # ── DASH ANALYSIS ─────────────────────────────────────────
    print("─"*65)
    print("  DASH folder  (Table 2 — mixed-condition videos)")
    print("─"*65)
    print(f"  Videos found : {len(dash_vids)}")
    print(f"  Expected     : 29  (per README)")
    print()
    print("  Sample filenames:")
    for v in dash_vids[:6]:
        print(f"    {v['filename']}")
    action_counts = Counter(v["action"] for v in dash_vids)
    print(f"\n  Action detected from filename:")
    for action, n in sorted(action_counts.items()):
        print(f"    {action:<10}: {n} videos")
    if action_counts.get("unknown", 0) > 0:
        print()
        print("  ✅ 'unknown' means NO action suffix in filename.")
        print("     This confirms Dash = mixed-condition per Table 2.")
        print("     → Must use MAR threshold for per-frame labeling.")

    # ── MIRROR ANALYSIS ────────────────────────────────────────
    print()
    print("─"*65)
    print("  MIRROR folder  (Table 1 — per-condition videos)")
    print("─"*65)
    print(f"  Videos found : {len(mirror_vids)}")
    print(f"  Expected     : 322  (per README)")
    print()
    print("  Sample filenames:")
    for v in mirror_vids[:6]:
        print(f"    {v['filename']}  →  action='{v['action']}'")
    action_counts_m = Counter(v["action"] for v in mirror_vids)
    print(f"\n  Action counts from filename parsing:")
    for action, n in sorted(action_counts_m.items()):
        flag = " ← TRAINING LABEL" if action in ("yawning", "talking") else \
               " ← skip (no yawn/talk distinction)" if action in ("normal", "mixed") else \
               " ← check manually"
        print(f"    {action:<12}: {n:>4} videos{flag}")

    # ── VIDEO METADATA PROBE ───────────────────────────────────
    if CV2_OK and all_videos:
        print()
        print("─"*65)
        print("  VIDEO METADATA  (probing 3 videos)")
        print("─"*65)
        for v in all_videos[:3]:
            cap = cv2.VideoCapture(str(v["path"]))
            if cap.isOpened():
                fps   = cap.get(cv2.CAP_PROP_FPS)
                nf    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur   = nf / fps if fps > 0 else 0
                cap.release()
                print(f"  {v['filename'][:45]:<45}  "
                      f"{w}×{h}  {fps:.0f}fps  {nf}f  {dur:.1f}s")

    # ── STRATEGY SUMMARY ──────────────────────────────────────
    print()
    print("="*65)
    print("  LABELING STRATEGY CONFIRMED")
    print("="*65)
    print("""
  Dash/ folder:
    Each video has ALL 3 conditions mixed in one recording.
    → Use MAR (Mouth Aspect Ratio) to label each frame:
        MAR > 0.5  →  yawn
        MAR ≤ 0.5  →  talking / normal (merged into "talking")

  Mirror/ folder:
    Each video is a single condition (Normal / Talking / Yawning).
    Action is encoded in the filename suffix after the last hyphen.
    → Use filename to assign session label, then MAR to filter frames:
        Yawning videos   → only save frames where MAR > 0.4 (confirm yawn)
        Talking  videos  → only save frames where MAR ≤ 0.5 (confirm talking)
        Normal   videos  → treat same as Talking (closed mouth)
        Mixed    videos  → use MAR threshold (like Dash)

  Combined result: clean per-frame labels from BOTH folders.
""")
    print("  NEXT STEP:")
    print("    python src/extract_frames.py --root ydd_dataset")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
