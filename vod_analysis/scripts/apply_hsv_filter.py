import argparse
import json
from pathlib import Path
import glob

import cv2
import numpy as np


def load_config(config_path):
    """
    Reads hsv_config.json. We still use the HSV bounds as a *first pass*
    to restrict candidates, but we add a robust RGB-dominance rule to
    reject pale greenish-white.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    lower = np.array(cfg["lower_green"], dtype=np.uint8)
    upper = np.array(cfg["upper_green"], dtype=np.uint8)
    return lower, upper


def apply_green_dominance_mask(img_bgr, hsv_lower, hsv_upper, *, g_margin=10, min_s=0):
    """
    green extraction:
    1. candidate pixels from HSV range (hsv_lower/hsv_upper)
    2, Require green channel dominance in BGR: G > R + g_margin and G > B + g_margin
    3.saturation floor (min_s) applied in OpenCV HSV space.
         Useful if you still have pale artifacts. set min_s=0 to disable.

    (uint8 0/255), result (BGR masked)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Base HSV mask (fast coarse filter)
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Optional saturation floor (can help suppress near-white tints)
    if min_s > 0:
        s = hsv[:, :, 1]
        sat_mask = (s >= min_s).astype(np.uint8) * 255
        hsv_mask = cv2.bitwise_and(hsv_mask, sat_mask)

    # Green dominance in BGR
    b, g, r = cv2.split(img_bgr)

    # Use int16 to avoid uint8 underflow when adding margin
    g_i = g.astype(np.int16)
    r_i = r.astype(np.int16)
    b_i = b.astype(np.int16)

    dom = (g_i > (r_i + g_margin)) & (g_i > (b_i + g_margin))
    dom_mask = dom.astype(np.uint8) * 255

    # Combine masks
    mask = cv2.bitwise_and(hsv_mask, dom_mask)

    # Apply
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return mask, result


def collect_input_images(input_arg):
    p = Path(input_arg)

    # Directory
    if p.is_dir():
        imgs = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            imgs.extend(p.glob(ext))
        return sorted(imgs)

    # Glob pattern
    if "*" in input_arg:
        return [Path(x) for x in glob.glob(input_arg)]

    # Single file
    if p.exists():
        return [p]

    raise FileNotFoundError(f"Could not find input: {input_arg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Input image, directory, or glob (e.g. img.png OR images/ OR images/*.png)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output directory (for multiple images) or file path (for single image)",
    )
    ap.add_argument(
        "--config",
        default="../configs/hsv_config.json",
        help="Path to hsv_config.json",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Optional resize scale (e.g. 3.0)",
    )

    # New robust filter knobs
    ap.add_argument(
        "--g-margin",
        type=int,
        default=10,
        help="Green dominance margin: keep pixels only if G > R+margin and G > B+margin",
    )
    ap.add_argument(
        "--min-s",
        type=int,
        default=0,
        help="Optional saturation floor in HSV (0-255). Set >0 to suppress pale greenish-white.",
    )

    args = ap.parse_args()

    lower, upper = load_config(args.config)

    inputs = collect_input_images(args.input)
    if not inputs:
        raise RuntimeError("No images found.")

    output_path = Path(args.output)

    # Multiple images -> output must be directory
    if len(inputs) > 1:
        output_path.mkdir(parents=True, exist_ok=True)

    for img_path in inputs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        if args.scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=args.scale, fy=args.scale)

        mask, result = apply_green_dominance_mask(
            img,
            lower,
            upper,
            g_margin=args.g_margin,
            min_s=args.min_s,
        )

        if len(inputs) == 1 and not output_path.is_dir():
            # single file output
            mask_path = output_path.with_name(output_path.stem + "_mask.png")
            result_path = output_path
        else:
            mask_path = output_path / f"{img_path.stem}_mask.png"
            result_path = output_path / f"{img_path.stem}_filtered.png"

        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(result_path), result)

        print(f"Saved:")
        print(f"  Mask:     {mask_path}")
        print(f"  Filtered: {result_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()