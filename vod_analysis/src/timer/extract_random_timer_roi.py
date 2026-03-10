"""
Extract the timer ROI from a random (or specified) frame using anchor-based
layout from configs/timer_layout_from_anchor.json.

Usage (from vod_analysis/):
    python -m src.timer.extract_random_timer_roi --video part538.mp4
    python -m src.timer.extract_random_timer_roi --video part538.mp4 --frame 73740
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from .infer import load_timer_layout, _match_anchor_multiscale, _timer_xyxy_from_anchor, _crop_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4", help="Input video path")
    ap.add_argument("--layout", default="configs/timer_layout_from_anchor.json",
                    help="Anchor-based timer layout config")
    ap.add_argument("--out", default="test_timer.png", help="Output image path")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    ap.add_argument("--frame", type=int, default=None, help="Optional frame index")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    layout = load_timer_layout(args.layout)

    # Load anchor template
    anchor_tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
    if anchor_tmpl is None:
        raise FileNotFoundError(f"Anchor template not found: {layout.anchor_template_path}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count.")

    frame_idx = args.frame if args.frame is not None else random.randrange(0, total)
    frame_idx = max(0, min(total - 1, frame_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")

    # Find anchor
    found = _match_anchor_multiscale(
        frame, anchor_tmpl,
        top_frac=layout.match_top_frac,
        right_frac=layout.match_right_frac,
    )
    if found is None:
        raise RuntimeError(f"Anchor not found in frame {frame_idx}")

    anchor_score, anchor_xywh = found
    print(f"Anchor score: {anchor_score:.4f}  bbox: {anchor_xywh}")

    if anchor_score < layout.min_score:
        print(f"WARNING: anchor score {anchor_score:.4f} < min_score {layout.min_score}")

    # Derive timer ROI
    timer_xyxy = _timer_xyxy_from_anchor(anchor_xywh, layout)
    timer_crop = _crop_xyxy(frame, timer_xyxy)
    if timer_crop is None:
        raise RuntimeError(f"Timer ROI crop failed: {timer_xyxy}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), timer_crop)

    t_video = frame_idx / fps if fps > 0 else None
    print(f"Saved: {out_path}")
    print(f"Frame: {frame_idx}/{total-1}  fps={fps:.3f}  t_video={t_video}")
    print(f"Timer ROI (px): x0={timer_xyxy[0]} y0={timer_xyxy[1]} x1={timer_xyxy[2]} y1={timer_xyxy[3]}")


if __name__ == "__main__":
    main()
