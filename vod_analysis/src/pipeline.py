"""
VOD analysis pipeline: toast template matching + timer inference.

Scans the VOD at 1fps using sequential grab() reads (faster than seeking).
On frames where a toast template matches, runs timer inference to extract
the IGT value.

Output: formatted txt with columns:
    timestamp  frame  timer_text  timer_conf  template_name  template_score

Usage (from vod_analysis/):
    python -m src.pipeline --video part538.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.timer.model import TinyCharCNN
from src.timer.infer import infer_timer, load_timer_layout
from src.toast.template_match import (
    crop_toast_roi,
    load_toast_templates,
    match_toast_templates,
)

# Scales tightened from (0.8..1.2) to (0.95..1.05) since we confirmed
# 1.0 works on 1080p VODs. 
MATCH_SCALES = (0.95, 1.00, 1.05)


def fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main():
    parser = argparse.ArgumentParser(description="VOD analysis pipeline")
    parser.add_argument("--video", required=True, help="Path to the VOD mp4")
    parser.add_argument("--toast_templates", default="templates/toast_templates",
                        help="Directory containing toast template PNGs")
    parser.add_argument("--timer_model", default="timer_model.pth",
                        help="Path to trained TinyCharCNN weights")
    parser.add_argument("--timer_layout", default="configs/timer_layout_from_anchor.json",
                        help="Path to timer layout config")
    parser.add_argument("--min_score", type=float, default=0.55,
                        help="Minimum toast template match score")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None,
                        help="Output txt path (default: <video>_pipeline.txt)")
    args = parser.parse_args()

    # --- Load models ---
    templates = load_toast_templates(args.toast_templates)
    print(f"Loaded {len(templates)} toast template(s): {[t.name for t in templates]}")

    layout = load_timer_layout(args.timer_layout)
    timer_model = TinyCharCNN()
    timer_model.load_state_dict_compat(
        torch.load(args.timer_model, map_location=args.device, weights_only=True)
    )
    timer_model.to(args.device).eval()
    print(f"Timer model loaded from {args.timer_model}")

    # Pre-load anchor template once (avoids re-reading from disk per frame)
    anchor_tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
    if anchor_tmpl is None:
        raise FileNotFoundError(f"Anchor template not found: {layout.anchor_template_path}")

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    frame_step = max(1, int(round(fps)))  # ~1fps sampling

    out_path = args.out or str(Path(args.video).stem + "_pipeline.txt")

    # ===== Scan at 1fps using sequential grab() =====
    events = []
    n_samples = total_frames // frame_step

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    pbar = tqdm(total=n_samples, desc="Scanning", unit="frame")
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        roi, _ = crop_toast_roi(frame)
        matches = match_toast_templates(
            roi, templates, min_score=args.min_score, scales=MATCH_SCALES,
        )

        if matches:
            timer = infer_timer(
                frame, timer_model, layout,
                device=args.device, anchor_template=anchor_tmpl,
            )
            t_video = frame_idx / fps

            for m in matches:
                events.append({
                    "timestamp": fmt_timestamp(t_video),
                    "frame": frame_idx,
                    "timer_text": timer["text"] or "",
                    "timer_conf": timer["conf"],
                    "template_name": m.name,
                    "template_score": m.score,
                })

            pbar.set_postfix(events=len(events), tmpl=matches[0].name)

        # Advance to next 1fps sample by grabbing (skip-decode) frames
        for _ in range(frame_step - 1):
            if not cap.grab():
                break
        frame_idx += frame_step
        pbar.update(1)

    pbar.close()
    cap.release()

    # --- Write output ---
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Pipeline output for {args.video}\n")
        f.write(f"# toast_templates={args.toast_templates}  min_score={args.min_score}\n")
        f.write(f"# timer_model={args.timer_model}  timer_layout={args.timer_layout}\n")
        f.write(f"# fps={fps}  total_frames={total_frames}  duration={duration_s:.0f}s\n")
        f.write(f"# {len(events)} events\n\n")

        header = (
            f"{'timestamp':<16} {'frame':>8} {'timer_text':>12} "
            f"{'timer_conf':>10} {'template_name':<30} {'template_score':>14}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for e in events:
            f.write(
                f"{e['timestamp']:<16} "
                f"{e['frame']:>8d} "
                f"{e['timer_text']:>12} "
                f"{e['timer_conf']:>10.4f} "
                f"{e['template_name']:<30} "
                f"{e['template_score']:>14.4f}\n"
            )

    print(f"\n{len(events)} events written to {out_path}")


if __name__ == "__main__":
    main()
