# src/timer/test_anchor_timer_rois.py
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


def load_layout_json(p: Path):
    cfg = json.loads(p.read_text(encoding="utf-8"))
    anchor = cfg["anchor"]
    timer_rel = cfg["timer_roi_rel_to_anchor"]
    bounds = cfg["digit_bounds_x_norm_within_timer_roi"]
    seps = cfg.get("separators_x_norm_within_timer_roi", {"colon": [0.0, 0.0], "dot": [0.0, 0.0]})

    # sanity
    bounds = [float(b) for b in bounds]
    if len(bounds) != 8:
        raise ValueError(f"Expected 8 digit bounds (7 digits), got {len(bounds)}")
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            raise ValueError("digit bounds must be strictly increasing")

    return {
        "anchor_template_path": anchor.get("template_path", None),
        "match_region": anchor.get("match_region", {"top_frac": 0.30, "right_frac": 0.50}),
        "min_score": float(anchor.get("min_score", 0.65)),
        "timer_rel": {k: float(timer_rel[k]) for k in ["x0", "y0", "x1", "y1"]},
        "digit_bounds": bounds,
        "seps": {
            "colon": [float(seps["colon"][0]), float(seps["colon"][1])],
            "dot":   [float(seps["dot"][0]),   float(seps["dot"][1])],
        }
    }


def find_anchor_multiscale(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    top_frac: float,
    right_frac: float,
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

        if best is None or float(max_val) > best[0]:
            ax = sx0 + max_loc[0]
            ay = sy0 + max_loc[1]
            best = (float(max_val), ax, ay, tw, th, float(s))

    if best is None:
        return None
    score, ax, ay, aw, ah, s = best
    return score, (ax, ay, aw, ah), s, (sx0, sy0, sx1 - sx0, sy1 - sy0)


def abs_timer_box_from_rel(anchor_xywh, timer_rel):
    ax, ay, aw, ah = anchor_xywh
    ah = float(max(1, ah))
    x0 = int(round(ax + timer_rel["x0"] * ah))
    y0 = int(round(ay + timer_rel["y0"] * ah))
    x1 = int(round(ax + timer_rel["x1"] * ah))
    y1 = int(round(ay + timer_rel["y1"] * ah))
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
    return frame_bgr[y0:y1, x0:x1].copy()


def digit_boxes_within_timer(timer_xyxy, bounds_x_norm, seps, pad_px=0):
    """
    timer_xyxy: absolute (x0,y0,x1,y1) of timer ROI in frame
    bounds_x_norm: len=8 boundaries within timer ROI for 7 digit slots
    seps: {"colon":[a,b], "dot":[a,b]} in [0,1] within timer ROI
          where [a,b] is the separator band to EXCLUDE
    pad_px: optional padding per digit (applied before exclusion, then clamped)
    Returns list of 7 absolute digit boxes (x0,y0,x1,y1)
    """
    tx0, ty0, tx1, ty1 = timer_xyxy
    W = max(1, tx1 - tx0)

    # separator bands in absolute x
    colon_a = tx0 + int(round(seps["colon"][0] * W))
    colon_b = tx0 + int(round(seps["colon"][1] * W))
    dot_a   = tx0 + int(round(seps["dot"][0]   * W))
    dot_b   = tx0 + int(round(seps["dot"][1]   * W))

    out = []
    for i in range(7):
        x0 = tx0 + int(round(bounds_x_norm[i] * W)) - pad_px
        x1 = tx0 + int(round(bounds_x_norm[i + 1] * W)) + pad_px

        # Exclude separators from adjacent digits
        # Format: d0 d1 : d2 d3 . d4 d5 d6
        if i == 1:  # d1 (minutes ones) must end before colon band
            x1 = min(x1, colon_a)
        elif i == 2:  # d2 (seconds tens) must start after colon band
            x0 = max(x0, colon_b)
        elif i == 3:  # d3 (seconds ones) must end before dot band
            x1 = min(x1, dot_a)
        elif i == 4:  # d4 (ms hundreds) must start after dot band
            x0 = max(x0, dot_b)

        # clamp to timer ROI
        x0 = clamp(x0, tx0, tx1 - 1)
        x1 = clamp(x1, tx0 + 1, tx1)

        # if exclusion collapses (bad config), fall back to original slot crop
        if x1 <= x0:
            x0 = tx0 + int(round(bounds_x_norm[i] * W))
            x1 = tx0 + int(round(bounds_x_norm[i + 1] * W))
            x0 = clamp(x0, tx0, tx1 - 1)
            x1 = clamp(x1, tx0 + 1, tx1)

        out.append((x0, ty0, x1, ty1))

    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4")
    ap.add_argument("--layout", default="configs/timer_layout_from_anchor.json")
    ap.add_argument("--anchor_png", default=None, help="Override anchor template path (else use layout JSON)")
    ap.add_argument("--out_dir", default="debug/anchor_timer_test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frame", type=int, default=None, help="Optional absolute frame idx (else random)")
    ap.add_argument("--tries", type=int, default=10, help="How many random frames to try to find anchor")
    ap.add_argument("--pad", type=int, default=0, help="Optional padding for digit crops (px)")
    ap.add_argument("--scales", default="0.7,0.8,0.9,1.0,1.1,1.2,1.3")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    layout = load_layout_json(Path(args.layout))
    top_frac = float(layout["match_region"].get("top_frac", 0.30))
    right_frac = float(layout["match_region"].get("right_frac", 0.50))
    min_score = float(layout["min_score"])
    timer_rel = layout["timer_rel"]
    bounds = layout["digit_bounds"]

    tmpl_path = args.anchor_png or layout["anchor_template_path"]
    if not tmpl_path:
        raise ValueError("No anchor template path provided. Set in JSON or pass --anchor_png.")
    template = cv2.imread(tmpl_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(tmpl_path)

    scales = tuple(float(s) for s in args.scales.split(",") if s.strip())

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count.")

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    chosen = None  # (idx, frame, score, anchor_xywh)
    if args.frame is not None:
        idx = clamp(int(args.frame), 0, total - 1)
        ok, frame = read_frame(idx)
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx}")
        found = find_anchor_multiscale(frame, template, top_frac, right_frac, scales=scales)
        if found is None:
            cap.release()
            raise RuntimeError("Anchor match failed on specified frame.")
        score, anchor_xywh, _, _ = found
        if score < min_score:
            cap.release()
            raise RuntimeError(f"Anchor score too low: {score:.3f} < {min_score:.3f}")
        chosen = (idx, frame, score, anchor_xywh)
    else:
        best = None
        for _ in range(args.tries):
            idx = random.randrange(0, total)
            ok, frame = read_frame(idx)
            if not ok or frame is None:
                continue
            found = find_anchor_multiscale(frame, template, top_frac, right_frac, scales=scales)
            if found is None:
                continue
            score, anchor_xywh, _, _ = found
            if best is None or score > best[0]:
                best = (score, idx, frame, anchor_xywh)
        if best is None or best[0] < min_score:
            cap.release()
            raise RuntimeError("Could not find a good anchor match in sampled frames.")
        score, idx, frame, anchor_xywh = best
        chosen = (idx, frame, score, anchor_xywh)

    cap.release()
    idx, frame, score, anchor_xywh = chosen

    ax, ay, aw, ah = anchor_xywh
    timer_xyxy = abs_timer_box_from_rel(anchor_xywh, timer_rel)
    timer_crop = crop_xyxy(frame, timer_xyxy)
    if timer_crop is None:
        raise RuntimeError("Timer ROI crop collapsed; check timer_rel_to_anchor values.")

    # Digit crops
    digit_boxes = digit_boxes_within_timer(timer_xyxy, bounds, layout["seps"], pad_px=args.pad)
    digit_crops = []
    for (x0, y0, x1, y1) in digit_boxes:
        digit_crops.append(frame[y0:y1, x0:x1].copy())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Draw overlay on frame
    vis = frame.copy()
    cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)
    cv2.putText(vis, f"ANCHOR score={score:.3f}", (ax, max(0, ay - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    tx0, ty0, tx1, ty1 = timer_xyxy
    cv2.rectangle(vis, (tx0, ty0), (tx1, ty1), (255, 0, 0), 2)
    cv2.putText(vis, "TIMER", (tx0, max(0, ty0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Draw digit boxes
    for i, (x0, y0, x1, y1) in enumerate(digit_boxes):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(vis, f"d{i}", (x0 + 2, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), vis)

    timer_path = out_dir / "timer_roi.png"
    cv2.imwrite(str(timer_path), timer_crop)

    # Save digit crops
    for i, c in enumerate(digit_crops):
        cv2.imwrite(str(out_dir / f"digit_{i}.png"), c)

    t = idx / fps if fps > 0 else None
    meta = {
        "video": str(args.video),
        "layout": str(args.layout),
        "anchor_template": str(tmpl_path),
        "frame_idx": int(idx),
        "t_seconds": float(t) if t is not None else None,
        "anchor_score": float(score),
        "anchor_xywh": {"x": int(ax), "y": int(ay), "w": int(aw), "h": int(ah)},
        "timer_xyxy": {"x0": int(tx0), "y0": int(ty0), "x1": int(tx1), "y1": int(ty1)},
        "digit_boxes_xyxy": [
            {"i": i, "x0": int(b[0]), "y0": int(b[1]), "x1": int(b[2]), "y1": int(b[3])}
            for i, b in enumerate(digit_boxes)
        ],
        "notes": "overlay.png shows anchor (yellow), timer ROI (blue), digit boxes (green). digit_0..digit_6.png are crops."
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {overlay_path}")
    print(f"Saved: {timer_path}")
    print(f"Saved digits: {out_dir}/digit_0.png .. digit_6.png")
    print(f"Frame: {idx}/{total-1}  fps={fps:.3f}  t={t}")
    print(f"Anchor score: {score:.3f}  anchor_xywh={anchor_xywh}  timer_xyxy={timer_xyxy}")


if __name__ == "__main__":
    main()
