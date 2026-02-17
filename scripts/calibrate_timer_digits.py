import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_timer_roi(roi_json_path: Path):
    cfg = json.loads(roi_json_path.read_text(encoding="utf-8"))
    t = cfg["timer"]
    x0, y0, x1, y1 = map(float, [t["x0"], t["y0"], t["x1"], t["y1"]])
    x0, y0, x1, y1 = clamp01(x0), clamp01(y0), clamp01(x1), clamp01(y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid timer ROI in JSON (x1<=x0 or y1<=y0)")
    return x0, y0, x1, y1


def crop_norm(frame_bgr, x0, y0, x1, y1):
    h, w = frame_bgr.shape[:2]
    xa = int(round(x0 * w))
    ya = int(round(y0 * h))
    xb = int(round(x1 * w))
    yb = int(round(y1 * h))
    xa = max(0, min(w - 1, xa))
    xb = max(0, min(w, xb))
    ya = max(0, min(h - 1, ya))
    yb = max(0, min(h, yb))
    if xb <= xa or yb <= ya:
        raise ValueError("Crop collapsed; check ROI coords vs frame size")
    return frame_bgr[ya:yb, xa:xb].copy(), (xa, ya, xb - xa, yb - ya)


def make_default_digit_bounds():
    # 8 boundaries for 7 digit slots inside the timer ROI
    # d0 d1 : d2 d3 . d4 d5 d6
    return [0.00, 0.14, 0.28, 0.46, 0.60, 0.74, 0.87, 1.00]


def make_default_separators():
    # default colon band between d1 and d2, dot band between d3 and d4
    # You will drag these to tightly cover the separator glyphs.
    return {
        "colon": [0.30, 0.34],
        "dot":   [0.62, 0.65],
    }


def draw_overlay(timer_img, digit_bounds, seps, selected_key, show_grid=True):
    vis = timer_img.copy()
    h, w = vis.shape[:2]

    if show_grid:
        for gx in np.linspace(0, 1, 11):
            x = int(round(gx * (w - 1)))
            cv2.line(vis, (x, 0), (x, h - 1), (40, 40, 40), 1)

    # draw digit slots
    for i in range(len(digit_bounds) - 1):
        x0 = int(round(digit_bounds[i] * w))
        x1 = int(round(digit_bounds[i + 1] * w))
        cv2.rectangle(vis, (x0, 0), (x1, h - 1), (0, 180, 0), 1)
        cv2.putText(vis, f"d{i}", (x0 + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    # draw digit boundary lines
    for i, bx in enumerate(digit_bounds):
        x = int(round(bx * w))
        key = f"b{i}"
        is_sel = (key == selected_key)
        color = (0, 0, 255) if is_sel else (200, 200, 0)
        thick = 2 if is_sel else 1
        cv2.line(vis, (x, 0), (x, h - 1), color, thick)
        cv2.putText(vis, f"b{i}", (x + 2, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # draw separator bands
    for name, (a, b) in seps.items():
        xa = int(round(a * w))
        xb = int(round(b * w))
        # band fill
        overlay = vis.copy()
        cv2.rectangle(overlay, (xa, 0), (xb, h - 1), (0, 0, 255), -1)
        vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

        # edges of the band
        for side, x in [("0", xa), ("1", xb)]:
            key = f"{name}{side}"  # colon0/colon1/dot0/dot1
            is_sel = (key == selected_key)
            color = (255, 255, 255) if is_sel else (0, 0, 255)
            thick = 2 if is_sel else 2
            cv2.line(vis, (x, 0), (x, h - 1), color, thick)
            cv2.putText(vis, f"{name}{side}", (x + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


def nearest_key(x_px, w, digit_bounds, seps):
    candidates = []
    # digit bounds
    for i, bx in enumerate(digit_bounds):
        candidates.append((abs(int(round(bx * w)) - x_px), f"b{i}"))
    # separators: left/right edges
    for name, (a, b) in seps.items():
        candidates.append((abs(int(round(a * w)) - x_px), f"{name}0"))
        candidates.append((abs(int(round(b * w)) - x_px), f"{name}1"))
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4")
    ap.add_argument("--roi", default="configs/roi.json")
    ap.add_argument("--out", default="configs/timer_digit_bounds.json")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tries", type=int, default=200)
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--scale", type=float, default=4.0)
    args = ap.parse_args()

    random.seed(args.seed)

    x0n, y0n, x1n, y1n = load_timer_roi(Path(args.roi))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count from video.")

    # pick a frame
    frame_idx = args.frame
    if frame_idx < 0:
        frame_idx = None
        for _ in range(args.tries):
            idx = random.randrange(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            roi_crop, _ = crop_norm(frame, x0n, y0n, x1n, y1n)
            if roi_crop is not None and roi_crop.size > 0:
                frame_idx = idx
                break
        if frame_idx is None:
            cap.release()
            raise RuntimeError("Failed to sample a valid frame ROI.")
    else:
        frame_idx = clamp(frame_idx, 0, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")

    timer_roi, bbox = crop_norm(frame, x0n, y0n, x1n, y1n)

    timer_big = cv2.resize(
        timer_roi,
        (int(timer_roi.shape[1] * args.scale), int(timer_roi.shape[0] * args.scale)),
        interpolation=cv2.INTER_NEAREST
    )

    digit_bounds = make_default_digit_bounds()
    seps = make_default_separators()

    selected = "b1"
    dragging = False
    show_grid = True

    window = "calibrate_timer_digits"
    help_lines = [
        "Click/drag any line (digit boundary b0..b7, colon0/colon1, dot0/dot1).",
        "TAB: next line   G: grid   R: reset   N: new random frame",
        "S: save config   Q/ESC: quit",
        "Goal: separators are tight bands around ':' and '.' (exclude from adjacent digits).",
    ]

    def all_keys():
        keys = [f"b{i}" for i in range(len(digit_bounds))]
        keys += ["colon0", "colon1", "dot0", "dot1"]
        return keys

    def key_next(cur):
        keys = all_keys()
        i = keys.index(cur) if cur in keys else 0
        return keys[(i + 1) % len(keys)]

    def enforce_constraints():
        # digit bounds strictly increasing
        eps = 0.002
        for i in range(1, len(digit_bounds)):
            if digit_bounds[i] <= digit_bounds[i - 1] + eps:
                digit_bounds[i] = min(1.0, digit_bounds[i - 1] + eps)

        # separators must have x0<x1 and lie in the correct region:
        # colon between b2 and b2-ish (between d1 and d2) -> around digit boundary b2
        # dot between b4 (between d3 and d4) -> around digit boundary b4
        # We'll constrain within [b1,b3] for colon, [b3,b5] for dot.
        b1, b2, b3 = digit_bounds[1], digit_bounds[2], digit_bounds[3]
        b3_, b4, b5 = digit_bounds[3], digit_bounds[4], digit_bounds[5]

        # colon
        c0, c1 = seps["colon"]
        c0 = clamp(c0, b1 + eps, b3 - eps)
        c1 = clamp(c1, b1 + eps, b3 - eps)
        if c1 <= c0 + eps:
            c1 = min(b3 - eps, c0 + eps)
        seps["colon"] = [c0, c1]

        # dot
        d0, d1 = seps["dot"]
        d0 = clamp(d0, b3_ + eps, b5 - eps)
        d1 = clamp(d1, b3_ + eps, b5 - eps)
        if d1 <= d0 + eps:
            d1 = min(b5 - eps, d0 + eps)
        seps["dot"] = [d0, d1]

    def on_mouse(event, x, y, flags, param):
        nonlocal selected, dragging
        h, w = timer_big.shape[:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = nearest_key(x, w, digit_bounds, seps)
            dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            bx = clamp01(x / float(w))
            eps = 0.002

            if selected.startswith("b"):
                i = int(selected[1:])
                if i == 0:
                    digit_bounds[0] = 0.0
                elif i == len(digit_bounds) - 1:
                    digit_bounds[-1] = 1.0
                else:
                    lo = digit_bounds[i - 1] + eps
                    hi = digit_bounds[i + 1] - eps
                    digit_bounds[i] = clamp(bx, lo, hi)

            elif selected in ("colon0", "colon1", "dot0", "dot1"):
                name = "colon" if selected.startswith("colon") else "dot"
                side = 0 if selected.endswith("0") else 1
                seps[name][side] = bx

            enforce_constraints()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        enforce_constraints()
        vis = draw_overlay(timer_big, digit_bounds, seps, selected, show_grid=show_grid)

        y = 18
        for line in help_lines:
            cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

        cv2.imshow(window, vis)
        key = cv2.waitKey(16) & 0xFFFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key in (ord("g"), ord("G")):
            show_grid = not show_grid

        if key == 9:  # TAB
            selected = key_next(selected)

        if key in (ord("r"), ord("R")):
            digit_bounds = make_default_digit_bounds()
            seps = make_default_separators()
            selected = "b1"

        if key in (ord("n"), ord("N")):
            # new random frame
            cap2 = cv2.VideoCapture(args.video)
            if cap2.isOpened():
                idx = random.randrange(0, total_frames)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok2, frame2 = cap2.read()
                cap2.release()
                if ok2 and frame2 is not None:
                    roi2, _ = crop_norm(frame2, x0n, y0n, x1n, y1n)
                    if roi2 is not None:
                        frame_idx = idx
                        timer_big = cv2.resize(
                            roi2,
                            (int(roi2.shape[1] * args.scale), int(roi2.shape[0] * args.scale)),
                            interpolation=cv2.INTER_NEAREST
                        )

        if key in (ord("s"), ord("S")):
            cfg = {
                "timer_roi_norm": {"x0": x0n, "y0": y0n, "x1": x1n, "y1": y1n},
                "frame_used": int(frame_idx),
                "digit_bounds_x_norm_within_timer": [float(b) for b in digit_bounds],  # len=8
                "separators_x_norm_within_timer": {
                    "colon": [float(seps["colon"][0]), float(seps["colon"][1])],
                    "dot":   [float(seps["dot"][0]),   float(seps["dot"][1])],
                },
                "digit_slots": [
                    {"name": "m_tens",  "i": 0},
                    {"name": "m_ones",  "i": 1},
                    {"name": "s_tens",  "i": 2},
                    {"name": "s_ones",  "i": 3},
                    {"name": "ms_hund",  "i": 4},
                    {"name": "ms_tens",  "i": 5},
                    {"name": "ms_ones",  "i": 6},
                ],
                "notes": "Digits use bounds[i]..bounds[i+1] within timer ROI. Separators define bands to EXCLUDE from adjacent digit crops.",
            }
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"[saved] {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
