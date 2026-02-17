import argparse
from pathlib import Path

import cv2
import numpy as np


CLASSES = [str(i) for i in range(10)] + ["colon", "dot"]
SPLITS = ["train", "val"]


def yellow_mask_from_bgr(img_bgr: np.ndarray, thr: int) -> np.ndarray:
    """
    Yellow score per pixel:
      score = min(R,G) - B
    Higher means "more yellow".

    Returns: uint8 mask {0,255}
    """
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    score = np.minimum(r, g) - b  # int16
    score = np.clip(score, 0, 255).astype(np.uint8)

    mask = (score >= thr).astype(np.uint8) * 255
    return mask


def morph_cleanup(mask: np.ndarray, k: int, close_iter: int, open_iter: int) -> np.ndarray:
    """
    Fill small holes and remove specks.
    """
    kernel = np.ones((k, k), np.uint8)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return mask


def apply_mask_to_gray(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = cv2.bitwise_and(gray, gray, mask=mask)
    return out


def ensure_out_dirs(out_root: Path):
    for split in SPLITS:
        for cls in CLASSES:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/timer_chars_raw", help="Input raw RGB dataset root")
    ap.add_argument("--out_root", default="data/timer_chars_yellow", help="Output processed dataset root")
    ap.add_argument("--thr", type=int, default=40, help="Yellow score threshold (min(R,G)-B)")
    ap.add_argument("--k", type=int, default=3, help="Morph kernel size (odd recommended: 3 or 5)")
    ap.add_argument("--close_iter", type=int, default=1, help="Morph CLOSE iterations")
    ap.add_argument("--open_iter", type=int, default=1, help="Morph OPEN iterations")
    ap.add_argument("--keep_mask_debug", action="store_true", help="Also save masks to out_root/_masks/")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_out_dirs(out_root)

    mask_debug_root = out_root / "_masks"
    if args.keep_mask_debug:
        mask_debug_root.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    for split in SPLITS:
        for cls in CLASSES:
            in_dir = in_root / split / cls
            out_dir = out_root / split / cls
            if not in_dir.exists():
                print(f"[WARN] missing: {in_dir}")
                continue

            for p in in_dir.glob("*.png"):
                total_in += 1
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                mask = yellow_mask_from_bgr(img, thr=args.thr)
                mask = morph_cleanup(mask, k=args.k, close_iter=args.close_iter, open_iter=args.open_iter)
                out = apply_mask_to_gray(img, mask)

                out_path = out_dir / p.name
                cv2.imwrite(str(out_path), out)
                total_out += 1

                if args.keep_mask_debug:
                    cv2.imwrite(str(mask_debug_root / f"{split}_{cls}_{p.stem}_mask.png"), mask)

    print(f"Done. Wrote {total_out}/{total_in} images to: {out_root}")
    print(f"Params: thr={args.thr}, k={args.k}, close_iter={args.close_iter}, open_iter={args.open_iter}")


if __name__ == "__main__":
    main()
