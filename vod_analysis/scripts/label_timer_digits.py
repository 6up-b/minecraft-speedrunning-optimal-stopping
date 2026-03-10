#!/usr/bin/env python3
"""
Manual labeling tool for timer digit crops.

Extracts timer ROIs from video frames using anchor-based layout,
displays them for the user to type the 7-digit label, and saves
individual digit crops to a train/val split directory structure.

Usage:
    cd vod_analysis
    python scripts/label_timer_digits.py --video input_lowres.mp4
    python scripts/label_timer_digits.py --video part538.mp4 --target 150 --val_ratio 0.1
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Add parent so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.timer.geometry import (
    TimerLayout,
    load_timer_layout,
    match_anchor_multiscale,
    timer_xyxy_from_anchor,
    crop_xyxy,
    digit_boxes_excluding_seps,
    letterbox,
)


def extract_timer_frames(video_path: str, layout: TimerLayout, n_frames: int,
                          anchor_template: np.ndarray, min_score: float = 0.65):
    """
    Sample random frames from video, return those where anchor is found.
    Yields (frame_idx, timer_roi_bgr, digit_crops_bgr, anchor_score).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 10:
        raise RuntimeError(f"Video too short: {total} frames")

    # Sample random frame indices (spread across the video)
    indices = sorted(random.sample(range(total), min(n_frames, total)))

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        found = match_anchor_multiscale(
            frame, anchor_template,
            top_frac=layout.match_top_frac,
            right_frac=layout.match_right_frac,
        )
        if found is None:
            continue

        score, anchor_xywh = found
        if score < min_score:
            continue

        timer_xyxy = timer_xyxy_from_anchor(anchor_xywh, layout)
        timer_crop = crop_xyxy(frame, timer_xyxy)
        if timer_crop is None:
            continue

        digit_boxes = digit_boxes_excluding_seps(
            timer_xyxy=timer_xyxy,
            bounds_x_norm=layout.digit_bounds,
            colon_band=layout.colon_band,
            dot_band=layout.dot_band,
        )

        digit_crops = []
        for bb in digit_boxes:
            crop = crop_xyxy(frame, bb)
            if crop is None:
                break
            digit_crops.append(crop)

        if len(digit_crops) != 7:
            continue

        yield frame_idx, timer_crop, digit_crops, score

    cap.release()


def display_timer_with_info(timer_crop: np.ndarray, frame_idx: int,
                             class_counts: Counter, target: int, labeled: int):
    """Show upscaled timer ROI with status info."""
    h, w = timer_crop.shape[:2]
    scale = max(6, 300 // max(h, 1))
    display = cv2.resize(timer_crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Add status bar below
    bar_h = 60
    bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)

    # Class distribution
    dist_str = " ".join(f"{d}:{class_counts.get(d, 0)}" for d in range(10))
    cv2.putText(bar, f"f={frame_idx} | labeled={labeled} | {dist_str}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(bar, "Type 7 digits + Enter | s=skip | q=quit",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)

    combined = np.vstack([display, bar])
    cv2.imshow("Timer Label Tool", combined)


def save_digit_crop(bgr_crop: np.ndarray, digit: int, slot: int, frame_idx: int,
                     timer_text: str, out_root: Path, split: str):
    """Save a letterboxed 32x32 BGR crop."""
    out_dir = out_root / split / str(digit)
    out_dir.mkdir(parents=True, exist_ok=True)

    lb = letterbox(bgr_crop, 32)
    fname = f"f{frame_idx}_d{slot}_{timer_text}.png"
    cv2.imwrite(str(out_dir / fname), lb)


def main():
    ap = argparse.ArgumentParser(description="Manual timer digit labeling tool")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--layout", default="configs/timer_layout_from_anchor.json",
                    help="Timer layout config")
    ap.add_argument("--target", type=int, default=100,
                    help="Target samples per digit class")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of samples for validation split")
    ap.add_argument("--out_root", default="data/timer_digits_v2",
                    help="Output root directory")
    ap.add_argument("--n_candidates", type=int, default=2000,
                    help="Number of random frames to sample from video")
    ap.add_argument("--min_anchor_score", type=float, default=0.65,
                    help="Minimum anchor match score")
    args = ap.parse_args()

    layout = load_timer_layout(args.layout)
    out_root = Path(args.out_root)

    # Load anchor template
    tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
    if tmpl is None:
        raise FileNotFoundError(f"Anchor template not found: {layout.anchor_template_path}")

    # Count existing samples
    class_counts = Counter()
    for digit in range(10):
        for split in ("train", "val"):
            d = out_root / split / str(digit)
            if d.exists():
                class_counts[digit] += len(list(d.glob("*.png")))

    print(f"Existing samples: {dict(sorted(class_counts.items()))}")
    print(f"Target: {args.target} per digit")
    print(f"Extracting candidate frames from {args.video}...")

    # Pre-extract candidates
    candidates = list(extract_timer_frames(
        args.video, layout, args.n_candidates, tmpl, args.min_anchor_score))
    random.shuffle(candidates)
    print(f"Found {len(candidates)} frames with valid timer ROI")

    if not candidates:
        print("No valid timer frames found. Check video path and anchor template.")
        return

    labeled = 0
    idx = 0

    while idx < len(candidates):
        # Check if we've reached target for all digits
        min_count = min(class_counts.get(d, 0) for d in range(10))
        if min_count >= args.target:
            print(f"\nAll digits have >= {args.target} samples. Done!")
            break

        frame_idx, timer_crop, digit_crops, score = candidates[idx]

        display_timer_with_info(timer_crop, frame_idx, class_counts, args.target, labeled)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("\nQuit requested.")
            break
        elif key == ord('s'):
            idx += 1
            continue
        elif key == 13 or key == 10:  # Enter without input — skip
            idx += 1
            continue

        # Collect digit string via keyboard
        # The first keypress was already captured above
        digit_str = chr(key) if 48 <= key <= 57 else ""

        if not digit_str:
            idx += 1
            continue

        # Show partial input and collect remaining digits
        while len(digit_str) < 7:
            # Update display with partial input
            h, w = timer_crop.shape[:2]
            scale = max(6, 300 // max(h, 1))
            display = cv2.resize(timer_crop, (w * scale, h * scale),
                                 interpolation=cv2.INTER_NEAREST)
            bar_h = 60
            bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, f"Input: {digit_str}_ ({len(digit_str)}/7)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(bar, "Backspace=undo | Esc=cancel",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            combined = np.vstack([display, bar])
            cv2.imshow("Timer Label Tool", combined)

            k = cv2.waitKey(0) & 0xFF
            if 48 <= k <= 57:  # digit 0-9
                digit_str += chr(k)
            elif k == 8 and digit_str:  # backspace
                digit_str = digit_str[:-1]
            elif k == 27:  # escape — cancel this frame
                digit_str = ""
                break

        if len(digit_str) != 7:
            idx += 1
            continue

        # Validate: all digits
        if not digit_str.isdigit():
            print(f"  Invalid input: {digit_str}")
            continue

        digits = [int(c) for c in digit_str]
        timer_text = f"{digit_str[:2]}_{digit_str[2:4]}_{digit_str[4:]}"

        # Decide split
        split = "val" if random.random() < args.val_ratio else "train"

        # Save each digit crop
        for slot, (digit, crop) in enumerate(zip(digits, digit_crops)):
            save_digit_crop(crop, digit, slot, frame_idx, timer_text, out_root, split)
            class_counts[digit] += 1

        labeled += 1
        print(f"  frame={frame_idx} input={digit_str} "
              f"-> d0={digits[0]} d1={digits[1]} d2={digits[2]} d3={digits[3]} "
              f"d4={digits[4]} d5={digits[5]} d6={digits[6]}  [{split}]")

        idx += 1

    cv2.destroyAllWindows()

    # Final summary
    print(f"\n--- Summary ---")
    print(f"Labeled {labeled} frames")
    for d in range(10):
        print(f"  digit {d}: {class_counts.get(d, 0)} samples")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
