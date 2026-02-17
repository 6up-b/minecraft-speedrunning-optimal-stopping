import argparse
from pathlib import Path
import cv2
import numpy as np

CLASSES = [str(i) for i in range(10)] + ["colon", "dot"]
SPLITS = ["train", "val"]

def ensure_out_dirs(out_root: Path):
    for split in SPLITS:
        for cls in CLASSES:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

def posterize(gray: np.ndarray, levels: int) -> np.ndarray:
    """
    Quantize grayscale to `levels` evenly spaced bins.
    levels=2 -> bin-like (0/255)
    levels=4 -> 0/85/170/255
    levels=8 -> finer
    """
    if levels <= 1:
        return gray
    step = 256 // levels  # e.g. 64 for 4 levels
    q = (gray // step) * step
    # stretch top bin to 255
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q

def fixed_binarize(gray: np.ndarray, thr: int, invert: bool) -> np.ndarray:
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw
    return bw

def adaptive_binarize(gray: np.ndarray, block: int, C: int, invert: bool) -> np.ndarray:
    # block must be odd and >=3
    block = max(3, block)
    if block % 2 == 0:
        block += 1
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, C
    )
    if invert:
        bw = 255 - bw
    return bw

def morph_cleanup(img: np.ndarray, k: int, close_iter: int, open_iter: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    if close_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/timer_chars_raw", help="Input raw RGB dataset root")
    ap.add_argument("--out_root", default="data/timer_chars_bin", help="Output dataset root")

    ap.add_argument("--mode", choices=["posterize", "bin_fixed", "bin_adapt"], default="posterize")

    # posterize params
    ap.add_argument("--levels", type=int, default=4, help="Posterize levels (2,4,8...)")

    # fixed binarize params
    ap.add_argument("--thr", type=int, default=140, help="Fixed threshold (0-255)")
    ap.add_argument("--invert", action="store_true", help="Invert binary output")

    # adaptive params
    ap.add_argument("--block", type=int, default=15, help="Adaptive block size (odd)")
    ap.add_argument("--C", type=int, default=3, help="Adaptive threshold constant")

    # optional cleanup
    ap.add_argument("--k", type=int, default=0, help="Morph kernel size (0 disables)")
    ap.add_argument("--close_iter", type=int, default=0)
    ap.add_argument("--open_iter", type=int, default=0)

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_out_dirs(out_root)

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

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if args.mode == "posterize":
                    out = posterize(gray, args.levels)

                elif args.mode == "bin_fixed":
                    out = fixed_binarize(gray, args.thr, invert=args.invert)

                elif args.mode == "bin_adapt":
                    out = adaptive_binarize(gray, args.block, args.C, invert=args.invert)

                else:
                    raise ValueError("Unknown mode")

                if args.k and (args.close_iter > 0 or args.open_iter > 0):
                    out = morph_cleanup(out, k=args.k, close_iter=args.close_iter, open_iter=args.open_iter)

                cv2.imwrite(str(out_dir / p.name), out)
                total_out += 1

    print(f"Done. Wrote {total_out}/{total_in} images to: {out_root}")
    print(f"Mode={args.mode}")
    if args.mode == "posterize":
        print(f"levels={args.levels}")
    elif args.mode == "bin_fixed":
        print(f"thr={args.thr} invert={args.invert}")
    else:
        print(f"block={args.block} C={args.C} invert={args.invert}")
    print(f"morph: k={args.k} close_iter={args.close_iter} open_iter={args.open_iter}")

if __name__ == "__main__":
    main()
