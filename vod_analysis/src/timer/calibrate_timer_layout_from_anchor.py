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


# ----------------------------
# Template anchor matching
# ----------------------------
def find_anchor_multiscale(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    top_frac: float = 0.30,
    right_frac: float = 0.50,
    scales=(0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
    method=cv2.TM_CCOEFF_NORMED,
):
    H, W = frame_bgr.shape[:2]
    sy0 = 0
    sy1 = int(round(top_frac * H))
    sx0 = int(round(right_frac * W))
    sx1 = W

    search = frame_bgr[sy0:sy1, sx0:sx1]
    if search.size == 0:
        return None

    search_g = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    tmpl_g0 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    best = None  # (score, ax, ay, aw, ah, scale)
    for s in scales:
        tw = int(round(tmpl_g0.shape[1] * s))
        th = int(round(tmpl_g0.shape[0] * s))
        if tw < 8 or th < 8:
            continue
        if th >= search_g.shape[0] or tw >= search_g.shape[1]:
            continue

        tmpl_g = cv2.resize(tmpl_g0, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_g, tmpl_g, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        score = float(max_val)
        loc = max_loc

        if best is None or score > best[0]:
            ax = sx0 + loc[0]
            ay = sy0 + loc[1]
            best = (score, ax, ay, tw, th, s)

    if best is None:
        return None

    score, ax, ay, aw, ah, s = best
    return score, (ax, ay, aw, ah), s, (sx0, sy0, sx1 - sx0, sy1 - sy0)


# ----------------------------
# Anchor-relative boxes in "anchor-height units"
# rel_x = (x - ax) / ah
# rel_y = (y - ay) / ah
# ----------------------------
def abs_box_from_rel(anchor_xywh, rel):
    ax, ay, aw, ah = anchor_xywh
    ah = float(max(1, ah))
    x0 = int(round(ax + rel["x0"] * ah))
    y0 = int(round(ay + rel["y0"] * ah))
    x1 = int(round(ax + rel["x1"] * ah))
    y1 = int(round(ay + rel["y1"] * ah))
    return x0, y0, x1, y1


def crop_xyxy(frame_bgr, box_xyxy):
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    x0 = clamp(x0, 0, W - 1)
    x1 = clamp(x1, 0, W)
    y0 = clamp(y0, 0, H - 1)
    y1 = clamp(y1, 0, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


# ----------------------------
# Default layout within timer ROI
# ----------------------------
def make_default_digit_bounds():
    return [0.00, 0.14, 0.28, 0.46, 0.60, 0.74, 0.87, 1.00]


def make_default_separators():
    return {"colon": [0.30, 0.34], "dot": [0.62, 0.65]}


def default_timer_roi_rel():
    # generous guess: digits to the right of "IGT:"
    return {"x0": 1.00, "y0": -0.25, "x1": 7.00, "y1": 1.30}


def enforce_timer_rel_constraints(timer_rel):
    # ensure x0<x1, y0<y1 with tiny eps
    eps = 0.05
    if timer_rel["x1"] <= timer_rel["x0"] + eps:
        timer_rel["x1"] = timer_rel["x0"] + eps
    if timer_rel["y1"] <= timer_rel["y0"] + eps:
        timer_rel["y1"] = timer_rel["y0"] + eps


def enforce_digit_constraints(digit_bounds, seps):
    eps = 0.002
    for i in range(1, len(digit_bounds)):
        if digit_bounds[i] <= digit_bounds[i - 1] + eps:
            digit_bounds[i] = min(1.0, digit_bounds[i - 1] + eps)

    b1, b3 = digit_bounds[1], digit_bounds[3]
    b3_, b5 = digit_bounds[3], digit_bounds[5]

    c0, c1 = seps["colon"]
    c0 = clamp(c0, b1 + eps, b3 - eps)
    c1 = clamp(c1, b1 + eps, b3 - eps)
    if c1 <= c0 + eps:
        c1 = min(b3 - eps, c0 + eps)
    seps["colon"] = [c0, c1]

    d0, d1 = seps["dot"]
    d0 = clamp(d0, b3_ + eps, b5 - eps)
    d1 = clamp(d1, b3_ + eps, b5 - eps)
    if d1 <= d0 + eps:
        d1 = min(b5 - eps, d0 + eps)
    seps["dot"] = [d0, d1]


# ----------------------------
# Drawing: timer ROI handles + internal digit overlay
# ----------------------------
def draw_timer_roi_preview(frame_bgr, anchor_xywh, timer_rel, scale=1.0):
    """
    Returns a copy of frame with anchor box + timer roi box drawn.
    """
    vis = frame_bgr.copy()
    ax, ay, aw, ah = anchor_xywh
    cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)
    cv2.putText(vis, "ANCHOR", (ax, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    x0, y0, x1, y1 = abs_box_from_rel(anchor_xywh, timer_rel)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)
    cv2.putText(vis, "TIMER_ROI", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    return vis


def draw_digit_overlay(timer_big, digit_bounds, seps, selected_key, show_grid=True):
    vis = timer_big.copy()
    h, w = vis.shape[:2]

    if show_grid:
        for gx in np.linspace(0, 1, 11):
            x = int(round(gx * (w - 1)))
            cv2.line(vis, (x, 0), (x, h - 1), (40, 40, 40), 1)

    # digit slots
    for i in range(len(digit_bounds) - 1):
        x0 = int(round(digit_bounds[i] * w))
        x1 = int(round(digit_bounds[i + 1] * w))
        cv2.rectangle(vis, (x0, 0), (x1, h - 1), (0, 180, 0), 1)
        cv2.putText(vis, f"d{i}", (x0 + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    # boundary lines (NOTE: b0 and b7 are still 0/1)
    for i, bx in enumerate(digit_bounds):
        x = int(round(bx * w))
        key = f"b{i}"
        is_sel = (key == selected_key)
        color = (0, 0, 255) if is_sel else (200, 200, 0)
        thick = 2 if is_sel else 1
        cv2.line(vis, (x, 0), (x, h - 1), color, thick)
        cv2.putText(vis, f"b{i}", (x + 2, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # separator bands
    for name, (a, b) in seps.items():
        xa = int(round(a * w))
        xb = int(round(b * w))
        overlay = vis.copy()
        cv2.rectangle(overlay, (xa, 0), (xb, h - 1), (0, 0, 255), -1)
        vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

        for side, x in [("0", xa), ("1", xb)]:
            key = f"{name}{side}"
            is_sel = (key == selected_key)
            color = (255, 255, 255) if is_sel else (0, 0, 255)
            cv2.line(vis, (x, 0), (x, h - 1), color, 2)
            cv2.putText(vis, f"{name}{side}", (x + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


def nearest_digit_key(x_px, w, digit_bounds, seps):
    candidates = []
    for i, bx in enumerate(digit_bounds):
        candidates.append((abs(int(round(bx * w)) - x_px), f"b{i}"))
    for name, (a, b) in seps.items():
        candidates.append((abs(int(round(a * w)) - x_px), f"{name}0"))
        candidates.append((abs(int(round(b * w)) - x_px), f"{name}1"))
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4")
    ap.add_argument("--anchor_png", required=True)
    ap.add_argument("--out", default="configs/timer_layout_from_anchor.json")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tries", type=int, default=300)
    ap.add_argument("--frame", type=int, default=-1)

    # anchor search
    ap.add_argument("--top_frac", type=float, default=0.30)
    ap.add_argument("--right_frac", type=float, default=0.50)
    ap.add_argument("--min_score", type=float, default=0.65)

    # visual scaling
    ap.add_argument("--timer_scale", type=float, default=6.0)

    args = ap.parse_args()
    random.seed(args.seed)

    template = cv2.imread(args.anchor_png, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(args.anchor_png)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count from video.")

    # pick best anchor frame
    chosen = None
    if args.frame >= 0:
        idx = clamp(args.frame, 0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx}")
        found = find_anchor_multiscale(frame, template, args.top_frac, args.right_frac)
        if found is None:
            cap.release()
            raise RuntimeError("Anchor match failed on specified frame.")
        score, anchor_xywh, _, _ = found
        if score < args.min_score:
            cap.release()
            raise RuntimeError(f"Anchor score too low: {score:.3f}")
        chosen = (idx, frame, score, anchor_xywh)
    else:
        best = None
        for _ in range(args.tries):
            idx = random.randrange(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            found = find_anchor_multiscale(frame, template, args.top_frac, args.right_frac)
            if found is None:
                continue
            score, anchor_xywh, _, _ = found
            if best is None or score > best[0]:
                best = (score, idx, frame, anchor_xywh)
        if best is None or best[0] < args.min_score:
            cap.release()
            raise RuntimeError("Could not find a good anchor match in sampled frames.")
        score, idx, frame, anchor_xywh = best
        chosen = (idx, frame, score, anchor_xywh)

    cap.release()
    frame_idx, frame, anchor_score, anchor_xywh = chosen

    timer_rel = default_timer_roi_rel()
    digit_bounds = make_default_digit_bounds()
    seps = make_default_separators()

    # UI state
    show_grid = True
    mode = "roi"  # "roi" or "digits"
    dragging = False
    selected = "roi_right"  # ROI handle or digit key
    mouse_x = 0

    # Precompute timer_roi / timer_big based on current timer_rel
    def refresh_timer_big():
        enforce_timer_rel_constraints(timer_rel)
        timer_box = abs_box_from_rel(anchor_xywh, timer_rel)
        roi, _ = crop_xyxy(frame, timer_box)
        if roi is None:
            return None, None, timer_box
        big = cv2.resize(
            roi,
            (int(roi.shape[1] * args.timer_scale), int(roi.shape[0] * args.timer_scale)),
            interpolation=cv2.INTER_NEAREST
        )
        return roi, big, timer_box

    timer_roi, timer_big, timer_box = refresh_timer_big()
    if timer_big is None:
        raise RuntimeError("Initial timer ROI crop failed. Adjust default_timer_roi_rel() constants.")

    # Window setup
    win = "calibrate_timer_layout_from_anchor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def roi_handles_px(timer_big, timer_rel):
        """
        Return handle x/y positions in timer_big pixels for roi_left/right/top/bottom.
        Since timer_big is derived from timer_rel crop, handles are just edges of that crop.
        We expose them for picking/dragging by interpreting mouse in the *frame preview* mode instead.
        So here we return None (kept for clarity).
        """
        return None

    def on_mouse(event, x, y, flags, param):
        nonlocal selected, dragging

        if mode != "digits":
            return

        # Convert window coords -> digits panel coords
        dx0 = panel["digits_x0"]
        dy0 = panel["digits_y0"]
        dw  = panel["digits_w"]
        dh  = panel["digits_h"]

        # Only respond if mouse is inside digits panel
        if not (dx0 <= x < dx0 + dw and dy0 <= y < dy0 + dh):
            if event == cv2.EVENT_LBUTTONUP:
                dragging = False
            return

        x_local = x - dx0
        y_local = y - dy0

        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = timer_big.shape[:2]
            selected = nearest_digit_key(x_local, w, digit_bounds, seps)
            dragging = True

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            h, w = timer_big.shape[:2]
            bx = clamp01(x_local / float(max(1, w)))
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

            enforce_digit_constraints(digit_bounds, seps)


    cv2.setMouseCallback(win, on_mouse)

    def save_cfg():
        ax, ay, aw, ah = anchor_xywh
        cfg = {
            "anchor": {
                "template_path": str(args.anchor_png),
                "match_region": {"top_frac": args.top_frac, "right_frac": args.right_frac},
                "min_score": float(args.min_score),
                "units": "anchor_height",
                "frame_used": int(frame_idx),
                "anchor_bbox_px_at_calibration": {"x": int(ax), "y": int(ay), "w": int(aw), "h": int(ah)},
                "anchor_score_at_calibration": float(anchor_score),
            },
            "timer_roi_rel_to_anchor": dict(timer_rel),
            "digit_bounds_x_norm_within_timer_roi": [float(b) for b in digit_bounds],
            "separators_x_norm_within_timer_roi": {
                "colon": [float(seps["colon"][0]), float(seps["colon"][1])],
                "dot":   [float(seps["dot"][0]),   float(seps["dot"][1])],
            },
            "digit_slots": [
                {"name": "m_tens", "i": 0},
                {"name": "m_ones", "i": 1},
                {"name": "s_tens", "i": 2},
                {"name": "s_ones", "i": 3},
                {"name": "ms_hund", "i": 4},
                {"name": "ms_tens", "i": 5},
                {"name": "ms_ones", "i": 6},
            ],
            "notes": "Edit ROI in ROI mode; edit digit bounds/separators in DIGITS mode. ROI is relative to anchor (anchor height units).",
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[saved] {out_path}")

    # ROI editing: via keyboard nudges (simple + reliable)
    # (Mouse ROI editing inside the frame preview is trickier; nudges are robust and fast.)
    # Use SHIFT to move faster.
    def roi_nudge(key, step):
        changed = False
        if key == ord("a"):  # left
            timer_rel["x0"] -= step; timer_rel["x1"] -= step; changed = True
        elif key == ord("d"):  # right
            timer_rel["x0"] += step; timer_rel["x1"] += step; changed = True
        elif key == ord("w"):  # up
            timer_rel["y0"] -= step; timer_rel["y1"] -= step; changed = True
        elif key == ord("s"):  # down
            timer_rel["y0"] += step; timer_rel["y1"] += step; changed = True
        elif key == ord("j"):  # shrink left edge
            timer_rel["x0"] += step; changed = True
        elif key == ord("l"):  # expand right edge
            timer_rel["x1"] += step; changed = True
        elif key == ord("i"):  # shrink top edge
            timer_rel["y0"] += step; changed = True
        elif key == ord("k"):  # expand bottom edge
            timer_rel["y1"] += step; changed = True
        return changed

    while True:

        # globals updated every frame (used by mouse callback)
        panel = {
            "digits_x0": 0,
            "digits_y0": 0,
            "digits_w": 0,
            "digits_h": 0,
        }

        enforce_timer_rel_constraints(timer_rel)
        enforce_digit_constraints(digit_bounds, seps)
        timer_roi, timer_big, timer_box = refresh_timer_big()
        if timer_big is None:
            # if user nudged into invalid region, keep going but warn
            timer_big = np.zeros((60, 300, 3), dtype=np.uint8)

        # Left panel: full frame with anchor + timer ROI drawn
        frame_vis = draw_timer_roi_preview(frame, anchor_xywh, timer_rel)
        frame_vis = cv2.resize(frame_vis, (int(frame_vis.shape[1] * 0.75), int(frame_vis.shape[0] * 0.75)),
                               interpolation=cv2.INTER_AREA)

        # Right panel: timer ROI big with digit overlay
        digits_vis = draw_digit_overlay(timer_big, digit_bounds, seps, selected_key=selected if mode == "digits" else "b1",
                                        show_grid=show_grid)
        

        gap = 10
        panel["digits_x0"] = frame_vis.shape[1] + gap
        panel["digits_y0"] = 0
        panel["digits_w"]  = digits_vis.shape[1]
        panel["digits_h"]  = digits_vis.shape[0]


        # Compose side-by-side
        H = max(frame_vis.shape[0], digits_vis.shape[0])
        W = frame_vis.shape[1] + 10 + digits_vis.shape[1]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:frame_vis.shape[0], :frame_vis.shape[1]] = frame_vis
        canvas[:digits_vis.shape[0], frame_vis.shape[1] + 10:frame_vis.shape[1] + 10 + digits_vis.shape[1]] = digits_vis

        # header text
        header = f"mode={mode.upper()} | ROI move: WASD (shift=fast) | ROI resize: J/L (left/right), I/K (top/bot) | DIGITS: click/drag lines | TAB next | G grid | R reset | P save | Q quit"
        cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(16) & 0xFFFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key in (ord("g"), ord("G")):
            show_grid = not show_grid

        if key in (ord("m"), ord("M")):
            mode = "digits" if mode == "roi" else "roi"

        if key == 9 and mode == "digits":  # TAB cycles only in digit mode
            keys = [f"b{i}" for i in range(len(digit_bounds))] + ["colon0", "colon1", "dot0", "dot1"]
            i = keys.index(selected) if selected in keys else 0
            selected = keys[(i + 1) % len(keys)]

        if key in (ord("r"), ord("R")):
            timer_rel = default_timer_roi_rel()
            digit_bounds = make_default_digit_bounds()
            seps = make_default_separators()
            selected = "b1"
            mode = "roi"

        if key in (ord("p"), ord("P")):
            save_cfg()

        # ROI nudging keys (only in ROI mode)
        if mode == "roi":
            fast = (cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 0) and (cv2.waitKey(1) & 0xFFFF)
            step = 0.05
            # If SHIFT held, OpenCV doesn't expose reliably; use uppercase as "fast"
            if key in (ord("A"), ord("D"), ord("W"), ord("S"), ord("J"), ord("L"), ord("I"), ord("K")):
                step = 0.20
                key = ord(chr(key).lower())

            changed = roi_nudge(key, step)
            if changed:
                # refresh happens next loop
                pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
