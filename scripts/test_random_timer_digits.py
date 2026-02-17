import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


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


def load_digit_layout(timer_digit_json_path: Path):
    """
    New format:
      digit_bounds_x_norm_within_timer: len=8
      separators_x_norm_within_timer: {"colon":[x0,x1], "dot":[x0,x1]}
    """
    cfg = json.loads(timer_digit_json_path.read_text(encoding="utf-8"))

    bounds = cfg.get("digit_bounds_x_norm_within_timer", None)
    if bounds is None:
        raise KeyError("Expected key 'digit_bounds_x_norm_within_timer' in digit JSON")
    bounds = [float(b) for b in bounds]
    if len(bounds) != 8:
        raise ValueError(f"Expected 8 x-boundaries for 7 digits, got {len(bounds)}")
    for i, b in enumerate(bounds):
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"Boundary {i} out of [0,1]: {b}")
        if i > 0 and bounds[i] <= bounds[i - 1]:
            raise ValueError("Boundaries must be strictly increasing")

    seps = cfg.get("separators_x_norm_within_timer", None)
    if seps is None:
        raise KeyError("Expected key 'separators_x_norm_within_timer' in digit JSON")

    if "colon" not in seps or "dot" not in seps:
        raise KeyError("Expected separators_x_norm_within_timer to contain 'colon' and 'dot'")

    colon = [float(seps["colon"][0]), float(seps["colon"][1])]
    dot = [float(seps["dot"][0]), float(seps["dot"][1])]

    for name, (a, b) in [("colon", colon), ("dot", dot)]:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError(f"{name} separator out of [0,1]: {a}, {b}")
        if b <= a:
            raise ValueError(f"{name} separator must satisfy x1>x0: {a}, {b}")

    return bounds, {"colon": colon, "dot": dot}


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


def crop_timer_digits(timer_roi_bgr, bounds_x_norm, seps_x_norm, pad_px=0):
    """
    Produces 7 digit crops (d0..d6) that EXCLUDE ':' and '.' bands.

    Timer layout is assumed fixed: d0 d1 : d2 d3 . d4 d5 d6

    We exclude separators by trimming adjacent digit crops:
      - d1 right edge <= colon.x0
      - d2 left edge  >= colon.x1
      - d3 right edge <= dot.x0
      - d4 left edge  >= dot.x1
    """
    H, W = timer_roi_bgr.shape[:2]
    crops = []

    colon0, colon1 = seps_x_norm["colon"]
    dot0, dot1 = seps_x_norm["dot"]

    # convert separator bands to pixel coords inside timer ROI
    colon0_px = int(round(colon0 * W))
    colon1_px = int(round(colon1 * W))
    dot0_px = int(round(dot0 * W))
    dot1_px = int(round(dot1 * W))

    for i in range(7):
        x0 = int(round(bounds_x_norm[i] * W))
        x1 = int(round(bounds_x_norm[i + 1] * W))

        # apply pad first (we'll clamp again after trimming)
        x0 = max(0, min(W - 1, x0 - pad_px))
        x1 = max(0, min(W,     x1 + pad_px))

        # trim off separators based on fixed positions
        if i == 1:
            # digit just before ':'
            x1 = min(x1, colon0_px)
        elif i == 2:
            # digit just after ':'
            x0 = max(x0, colon1_px)
        elif i == 3:
            # digit just before '.'
            x1 = min(x1, dot0_px)
        elif i == 4:
            # digit just after '.'
            x0 = max(x0, dot1_px)

        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W, x1))

        if x1 <= x0:
            crop = timer_roi_bgr[:, 0:1].copy()
        else:
            crop = timer_roi_bgr[:, x0:x1].copy()

        crops.append(crop)

    return crops


def make_preview_grid(crops, scale=6, gap=6):
    ups = []
    for i, c in enumerate(crops):
        h, w = c.shape[:2]
        if h == 0 or w == 0:
            continue
        up = cv2.resize(c, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(up, f"d{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
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
    ap.add_argument("--roi", default="configs/roi.json", help="ROI json path (expects cfg['timer'])")
    ap.add_argument("--bounds", default="configs/timer_digit_bounds.json", help="New timer digit JSON w/ separators")
    ap.add_argument("--out", default="test_timer_digits.png", help="Output preview image path")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    ap.add_argument("--frame", type=int, default=None, help="Optional absolute frame index (overrides random)")
    ap.add_argument("--pad", type=int, default=0, help="Optional padding (px) for each digit crop")
    ap.add_argument("--scale", type=int, default=6, help="Upscale factor for preview grid")
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

    frame_idx = args.frame if args.frame is not None else random.randrange(0, total)
    frame_idx = max(0, min(total - 1, frame_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {args.video}")

    # crop timer roi
    x0, y0, x1, y1 = load_timer_roi(Path(args.roi))
    timer_roi, bbox = crop_norm(frame, x0, y0, x1, y1)

    # load bounds + separators and crop digits
    bounds, seps = load_digit_layout(Path(args.bounds))
    digit_crops = crop_timer_digits(timer_roi, bounds_x_norm=bounds, seps_x_norm=seps, pad_px=args.pad)

    # build preview: top = timer ROI with lines, bottom = digits
    timer_up = cv2.resize(
        timer_roi,
        (timer_roi.shape[1] * args.scale, timer_roi.shape[0] * args.scale),
        interpolation=cv2.INTER_NEAREST
    )

    H_up, W_up = timer_up.shape[:2]

    # draw digit boundaries
    for i, b in enumerate(bounds):
        x = int(round(b * W_up))
        cv2.line(timer_up, (x, 0), (x, H_up - 1), (0, 0, 255), 2)
        cv2.putText(timer_up, f"b{i}", (x + 3, H_up - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # draw separator bands
    for name, (a, b) in seps.items():
        xa = int(round(a * W_up))
        xb = int(round(b * W_up))
        overlay = timer_up.copy()
        cv2.rectangle(overlay, (xa, 0), (xb, H_up - 1), (255, 255, 255), -1)
        timer_up = cv2.addWeighted(overlay, 0.18, timer_up, 0.82, 0)
        cv2.putText(timer_up, name, (xa + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

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
    meta = f"frame={frame_idx}/{total-1} fps={fps:.3f} t={t_video:.3f}s  pad={args.pad}px"
    cv2.putText(out_img, meta, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)

    print(f"Saved: {out_path}")
    print(f"Frame: {frame_idx}/{total-1}  fps={fps:.3f}  t_video={t_video}")
    print(f"Timer ROI bbox (px): x={bbox[0]} y={bbox[1]} w={bbox[2]} h={bbox[3]}")
    print(f"Separators (norm within timer): colon={seps['colon']} dot={seps['dot']}")


if __name__ == "__main__":
    main()
