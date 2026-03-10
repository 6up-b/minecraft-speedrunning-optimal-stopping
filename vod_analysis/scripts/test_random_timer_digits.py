"""
Visual test: extract timer ROI + digit crops from a random (or specified)
frame using anchor-based layout from configs/timer_layout_from_anchor.json.

Outputs a preview image with timer ROI (with digit boundaries drawn) on top
and individual digit crops below.

Usage (from vod_analysis/):
    python scripts/test_random_timer_digits.py --video part538.mp4
    python scripts/test_random_timer_digits.py --video part538.mp4 --frame 73740
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.timer.infer import (
    load_timer_layout,
    _match_anchor_multiscale,
    _timer_xyxy_from_anchor,
    _crop_xyxy,
    _digit_boxes_excluding_seps,
    clamp,
)


def make_preview_grid(digit_crops, scale=6, gap=6):
    ups = []
    for i, c in enumerate(digit_crops):
        h, w = c.shape[:2]
        if h == 0 or w == 0:
            continue
        up = cv2.resize(c, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(up, f"d{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        ups.append(up)

    if not ups:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    H = max(u.shape[0] for u in ups)
    total_w = sum(u.shape[1] for u in ups) + gap * (len(ups) - 1)
    canvas = np.zeros((H, total_w, 3), dtype=np.uint8)

    x = 0
    for u in ups:
        h, w = u.shape[:2]
        canvas[0:h, x:x+w] = u
        x += w + gap

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4", help="Input video path")
    ap.add_argument("--layout", default="configs/timer_layout_from_anchor.json",
                    help="Anchor-based timer layout config")
    ap.add_argument("--out", default="test_timer_digits.png", help="Output preview image")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    ap.add_argument("--frame", type=int, default=None, help="Optional frame index")
    ap.add_argument("--pad", type=int, default=0, help="Padding (px) for digit crops")
    ap.add_argument("--scale", type=int, default=6, help="Upscale factor for preview")
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

    # Derive timer ROI + digit boxes
    timer_xyxy = _timer_xyxy_from_anchor(anchor_xywh, layout)
    timer_crop = _crop_xyxy(frame, timer_xyxy)
    if timer_crop is None:
        raise RuntimeError(f"Timer ROI crop failed: {timer_xyxy}")

    digit_boxes = _digit_boxes_excluding_seps(
        timer_xyxy=timer_xyxy,
        bounds_x_norm=layout.digit_bounds,
        colon_band=layout.colon_band,
        dot_band=layout.dot_band,
        pad_px=args.pad,
    )

    digit_crops = []
    for bb in digit_boxes:
        crop = _crop_xyxy(frame, bb)
        if crop is not None:
            digit_crops.append(crop)
        else:
            digit_crops.append(np.zeros((1, 1, 3), dtype=np.uint8))

    # Build preview: top = timer ROI with boundary lines, bottom = digit crops
    timer_up = cv2.resize(
        timer_crop,
        (timer_crop.shape[1] * args.scale, timer_crop.shape[0] * args.scale),
        interpolation=cv2.INTER_NEAREST,
    )
    H_up, W_up = timer_up.shape[:2]

    # Draw digit boundaries on timer ROI preview
    for i, bx in enumerate(layout.digit_bounds):
        x = int(round(bx * W_up))
        cv2.line(timer_up, (x, 0), (x, H_up - 1), (0, 0, 255), 2)
        cv2.putText(timer_up, f"b{i}", (x + 3, H_up - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw separator bands
    for name, band in [("colon", layout.colon_band), ("dot", layout.dot_band)]:
        xa = int(round(band[0] * W_up))
        xb = int(round(band[1] * W_up))
        overlay = timer_up.copy()
        cv2.rectangle(overlay, (xa, 0), (xb, H_up - 1), (255, 255, 255), -1)
        timer_up = cv2.addWeighted(overlay, 0.18, timer_up, 0.82, 0)
        cv2.putText(timer_up, name, (xa + 3, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    grid = make_preview_grid(digit_crops, scale=args.scale, gap=8)

    gap = 10
    W = max(timer_up.shape[1], grid.shape[1])
    top = np.zeros((timer_up.shape[0], W, 3), dtype=np.uint8)
    bot = np.zeros((grid.shape[0], W, 3), dtype=np.uint8)
    top[:, :timer_up.shape[1]] = timer_up
    bot[:, :grid.shape[1]] = grid

    out_img = np.zeros((top.shape[0] + gap + bot.shape[0], W, 3), dtype=np.uint8)
    out_img[0:top.shape[0]] = top
    out_img[top.shape[0] + gap: top.shape[0] + gap + bot.shape[0]] = bot

    t_video = frame_idx / fps if fps > 0 else None
    meta = f"frame={frame_idx}/{total-1} fps={fps:.3f} t={t_video:.3f}s anchor={anchor_score:.4f} pad={args.pad}px"
    cv2.putText(out_img, meta, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)

    print(f"Saved: {out_path}")
    print(f"Frame: {frame_idx}/{total-1}  fps={fps:.3f}  t_video={t_video}")
    print(f"Timer ROI (px): x0={timer_xyxy[0]} y0={timer_xyxy[1]} x1={timer_xyxy[2]} y1={timer_xyxy[3]}")
    for i, bb in enumerate(digit_boxes):
        print(f"  d{i}: x0={bb[0]} y0={bb[1]} x1={bb[2]} y1={bb[3]}")


if __name__ == "__main__":
    main()
