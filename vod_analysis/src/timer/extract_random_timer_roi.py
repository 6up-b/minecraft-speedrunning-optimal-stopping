import argparse
import json
import random
from pathlib import Path

import cv2


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_timer_roi(roi_json_path: Path):
    cfg = json.loads(roi_json_path.read_text(encoding="utf-8"))
    t = cfg["timer"]
    x0 = clamp01(float(t["x0"]))
    y0 = clamp01(float(t["y0"]))
    x1 = clamp01(float(t["x1"]))
    y1 = clamp01(float(t["y1"]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid timer ROI in JSON (x1<=x0 or y1<=y0)")
    return x0, y0, x1, y1


def crop_norm(frame_bgr, x0, y0, x1, y1):
    h, w = frame_bgr.shape[:2]
    xa = int(x0 * w)
    ya = int(y0 * h)
    xb = int(x1 * w)
    yb = int(y1 * h)
    xa = max(0, min(w - 1, xa))
    xb = max(0, min(w, xb))
    ya = max(0, min(h - 1, ya))
    yb = max(0, min(h, yb))
    if xb <= xa or yb <= ya:
        raise ValueError("Crop collapsed; check ROI coords vs frame size")
    return frame_bgr[ya:yb, xa:xb].copy(), (xa, ya, xb - xa, yb - ya)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4", help="Input video path")
    ap.add_argument("--roi", default="configs/roi.json", help="ROI json path (expects cfg['timer'])")
    ap.add_argument("--out", default="test_timer.png", help="Output image path")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    ap.add_argument("--frame", type=int, default=None, help="Optional absolute frame index (overrides random)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count (maybe codec issue).")

    # Choose frame
    frame_idx = args.frame if args.frame is not None else random.randrange(0, total)
    frame_idx = max(0, min(total - 1, frame_idx))

    # Seek + read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {args.video}")

    # Load ROI + crop
    x0, y0, x1, y1 = load_timer_roi(Path(args.roi))
    crop, bbox = crop_norm(frame, x0, y0, x1, y1)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)

    t_video = frame_idx / fps if fps > 0 else None
    print(f"Saved: {out_path}")
    print(f"Frame: {frame_idx}/{total-1}  fps={fps:.3f}  t_video={t_video}")
    print(f"Timer ROI bbox (px): x={bbox[0]} y={bbox[1]} w={bbox[2]} h={bbox[3]}")


if __name__ == "__main__":
    main()
