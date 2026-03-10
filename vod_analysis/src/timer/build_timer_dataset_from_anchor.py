#!/usr/bin/env python3
"""
Build timer digit dataset using anchor-based geometry + Google Vision API.

Extracts timer ROIs from video frames, uses Google Vision to read the full
timer string, then saves individual digit crops with Vision-derived labels.

Usage:
    cd vod_analysis
    python -m src.timer.build_timer_dataset_from_anchor \
        --videos input_lowres.mp4 part538.mp4 \
        --target 100 --out_root data/timer_digits_v2
"""
import argparse
import base64
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    TimerLayout,
    load_timer_layout,
    match_anchor_multiscale,
    timer_xyxy_from_anchor,
    crop_xyxy,
    digit_boxes_excluding_seps,
    letterbox,
)


def call_google_vision_ocr(image_bgr: np.ndarray) -> str:
    """
    Call Google Cloud Vision TEXT_DETECTION on an in-memory BGR image.
    Requires GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_VISION_API_KEY env var.
    Returns raw detected text string.
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        _, buf = cv2.imencode(".png", image_bgr)
        image = vision.Image(content=buf.tobytes())
        resp = client.text_detection(image=image)
        texts = resp.text_annotations
        if texts:
            return texts[0].description.strip()
        return ""
    except ImportError:
        # Fallback to REST API with API key
        api_key = os.environ.get("GOOGLE_VISION_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Install google-cloud-vision or set GOOGLE_VISION_API_KEY env var")

        import urllib.request
        _, buf = cv2.imencode(".png", image_bgr)
        b64 = base64.b64encode(buf.tobytes()).decode()

        body = json.dumps({
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "TEXT_DETECTION"}],
            }]
        })

        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        req = urllib.request.Request(url, data=body.encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())

        annotations = result.get("responses", [{}])[0].get("textAnnotations", [])
        if annotations:
            return annotations[0]["description"].strip()
        return ""


def parse_timer_text(raw: str) -> str | None:
    """
    Parse Vision OCR output into 7-digit string.
    Handles formats like "00:05.429", "00:05 429", "00 05.429", etc.
    Returns "0005429" or None if unparseable.
    """
    # Remove whitespace, keep digits, colons, dots
    cleaned = re.sub(r'[^0-9:.]', '', raw)

    # Try mm:ss.mmm format
    m = re.match(r'^(\d{2}):(\d{2})\.(\d{3})$', cleaned)
    if m:
        return m.group(1) + m.group(2) + m.group(3)

    # Try just extracting 7 consecutive digits
    digits_only = re.sub(r'[^0-9]', '', cleaned)
    if len(digits_only) == 7:
        return digits_only

    return None


def extract_and_label(video_path: str, layout: TimerLayout, anchor_template: np.ndarray,
                       n_frames: int, min_score: float = 0.65):
    """
    Extract timer frames and label via Google Vision.
    Yields (frame_idx, digit_crops, digit_labels_str, vision_raw).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 10:
        raise RuntimeError(f"Video too short: {total} frames")

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

        # Upscale for better OCR
        h, w = timer_crop.shape[:2]
        upscaled = cv2.resize(timer_crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

        raw_text = call_google_vision_ocr(upscaled)
        digits_str = parse_timer_text(raw_text)

        if digits_str is None:
            print(f"  frame {frame_idx}: Vision returned '{raw_text}' — skipped")
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

        yield frame_idx, digit_crops, digits_str, raw_text

    cap.release()


def main():
    ap = argparse.ArgumentParser(description="Build timer dataset from anchor + Google Vision")
    ap.add_argument("--videos", nargs="+", required=True, help="Video files")
    ap.add_argument("--layout", default="configs/timer_layout_from_anchor.json")
    ap.add_argument("--target", type=int, default=100, help="Target samples per digit")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out_root", default="data/timer_digits_v2")
    ap.add_argument("--n_frames", type=int, default=1000, help="Frames to sample per video")
    ap.add_argument("--min_anchor_score", type=float, default=0.65)
    args = ap.parse_args()

    layout = load_timer_layout(args.layout)
    out_root = Path(args.out_root)

    tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
    if tmpl is None:
        raise FileNotFoundError(f"Anchor template: {layout.anchor_template_path}")

    class_counts = Counter()
    for digit in range(10):
        for split in ("train", "val"):
            d = out_root / split / str(digit)
            if d.exists():
                class_counts[digit] += len(list(d.glob("*.png")))

    print(f"Existing: {dict(sorted(class_counts.items()))}")
    labeled = 0

    for video_path in args.videos:
        print(f"\nProcessing: {video_path}")

        for frame_idx, digit_crops, digits_str, raw_text in extract_and_label(
            video_path, layout, tmpl, args.n_frames, args.min_anchor_score
        ):
            # Check if all digits have enough samples
            min_count = min(class_counts.get(d, 0) for d in range(10))
            if min_count >= args.target:
                print(f"All digits >= {args.target}. Stopping.")
                break

            digits = [int(c) for c in digits_str]
            timer_text = f"{digits_str[:2]}_{digits_str[2:4]}_{digits_str[4:]}"
            split = "val" if random.random() < args.val_ratio else "train"

            for slot, (digit, crop) in enumerate(zip(digits, digit_crops)):
                d = out_root / split / str(digit)
                d.mkdir(parents=True, exist_ok=True)
                lb = letterbox(crop, 32)
                fname = f"f{frame_idx}_d{slot}_{timer_text}.png"
                cv2.imwrite(str(d / fname), lb)
                class_counts[digit] += 1

            labeled += 1
            print(f"  f={frame_idx} vision='{raw_text}' -> {digits_str} [{split}]")

    print(f"\n--- Summary ---")
    print(f"Labeled {labeled} frames")
    for d in range(10):
        print(f"  digit {d}: {class_counts.get(d, 0)} samples")


if __name__ == "__main__":
    main()
