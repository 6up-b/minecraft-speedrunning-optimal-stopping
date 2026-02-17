import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yellow_score_mask(bgr: np.ndarray, thr: int = 40) -> np.ndarray:
    """
    score = min(R,G) - B
    returns uint8 mask {0,255}
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    score = np.minimum(r, g) - b
    score = np.clip(score, 0, 255).astype(np.uint8)
    return (score >= thr).astype(np.uint8) * 255


def find_anchor_multiscale(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    top_frac: float = 0.30,
    right_frac: float = 0.50,
    scales=(0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.25),
    method=cv2.TM_CCOEFF_NORMED,
):
    """
    Searches in top-right region: y in [0, top_frac*H), x in [right_frac*W, W)
    Returns: best (score, (x,y,w,h), scale_used, search_roi_xywh)
    (x,y,w,h) are in full-frame coordinates
    """
    H, W = frame_bgr.shape[:2]
    y0 = 0
    y1 = int(round(top_frac * H))
    x0 = int(round(right_frac * W))
    x1 = W

    search = frame_bgr[y0:y1, x0:x1]
    if search.size == 0:
        return None

    # Use grayscale for matching (more robust)
    search_g = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    tmpl_g0 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    best = None  # (score, x, y, w, h, scale)
    for s in scales:
        tw = int(round(tmpl_g0.shape[1] * s))
        th = int(round(tmpl_g0.shape[0] * s))
        if tw < 8 or th < 8:
            continue
        if th >= search_g.shape[0] or tw >= search_g.shape[1]:
            continue

        tmpl_g = cv2.resize(tmpl_g0, (tw, th), interpolation=cv2.INTER_AREA)

        res = cv2.matchTemplate(search_g, tmpl_g, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = max_val if method in (cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED) else -min_val
        loc = max_loc if method in (cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED) else min_loc

        if best is None or score > best[0]:
            ax = x0 + loc[0]
            ay = y0 + loc[1]
            best = (score, ax, ay, tw, th, s)

    if best is None:
        return None

    score, ax, ay, aw, ah, s = best
    return score, (ax, ay, aw, ah), s, (x0, y0, x1 - x0, y1 - y0)


def bbox_from_mask(mask_u8: np.ndarray, pad: int = 2):
    """
    mask_u8: uint8 0/255
    Returns bbox (x0,y0,x1,y1) in mask coords, or None if empty
    """
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask_u8.shape[1], x1 + pad)
    y1 = min(mask_u8.shape[0], y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def segment_columns_to_runs(mask_u8: np.ndarray, min_on: int = 1):
    """
    Returns list of (x0,x1) runs where column-sum > 0.
    """
    col = (mask_u8 > 0).sum(axis=0)
    on = col >= min_on
    runs = []
    x = 0
    W = mask_u8.shape[1]
    while x < W:
        if not on[x]:
            x += 1
            continue
        x0 = x
        while x < W and on[x]:
            x += 1
        x1 = x
        runs.append((x0, x1))
    return runs


def merge_close_runs(runs, gap_thr: int = 2):
    if not runs:
        return []
    out = [list(runs[0])]
    for x0, x1 in runs[1:]:
        prev = out[-1]
        if x0 - prev[1] <= gap_thr:
            prev[1] = x1
        else:
            out.append([x0, x1])
    return [(a, b) for a, b in out]


def choose_9_segments(runs, target=9):
    """
    We expect 9 glyph segments: d0 d1 ':' d2 d3 '.' d4 d5 d6
    If we get more, merge smallest gaps.
    If fewer, fall back to uniform splitting later.
    """
    runs = runs[:]
    if len(runs) == target:
        return runs
    if len(runs) < target:
        return None

    # Merge closest neighbors until target count
    while len(runs) > target:
        gaps = []
        for i in range(len(runs) - 1):
            gap = runs[i + 1][0] - runs[i][1]
            gaps.append((gap, i))
        gaps.sort(key=lambda t: t[0])
        _, i = gaps[0]
        merged = (runs[i][0], runs[i + 1][1])
        runs = runs[:i] + [merged] + runs[i + 2 :]
    return runs


def runs_to_boxes(runs, y0, y1):
    """
    Convert x-runs into boxes spanning [y0,y1)
    Returns list of (x0,y0,x1,y1)
    """
    return [(x0, y0, x1, y1) for (x0, x1) in runs]


def rel_box_from_anchor(anchor_xywh, box_xyxy):
    """
    Store box relative to anchor top-left, scaled by anchor height.
    rel = ( (x - ax)/ah, (y - ay)/ah, (x2 - ax)/ah, (y2 - ay)/ah )
    """
    ax, ay, aw, ah = anchor_xywh
    x0, y0, x1, y1 = box_xyxy
    ah = float(max(1, ah))
    return {
        "x0": (x0 - ax) / ah,
        "y0": (y0 - ay) / ah,
        "x1": (x1 - ax) / ah,
        "y1": (y1 - ay) / ah,
    }


def abs_box_from_rel(anchor_xywh, rel):
    ax, ay, aw, ah = anchor_xywh
    ah = float(max(1, ah))
    x0 = int(round(ax + rel["x0"] * ah))
    y0 = int(round(ay + rel["y0"] * ah))
    x1 = int(round(ax + rel["x1"] * ah))
    y1 = int(round(ay + rel["y1"] * ah))
    return x0, y0, x1, y1


def draw_box(img, box_xyxy, color, thickness=2, label=None):
    x0, y0, x1, y1 = box_xyxy
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if label is not None:
        cv2.putText(img, label, (x0 + 2, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_png", required=True, help="Path to anchor template PNG, e.g. assets/igt_anchor.png")
    ap.add_argument("--out_json", default="configs/timer_from_igt_anchor.json", help="Where to write calibration JSON")

    # choose a calibration frame
    ap.add_argument("--image", default=None, help="Calibration frame image (png/jpg). If set, ignores --video.")
    ap.add_argument("--video", default=None, help="Calibration video path")
    ap.add_argument("--frame", type=int, default=None, help="Frame idx in video; if omitted random frame is used")
    ap.add_argument("--seed", type=int, default=123)

    # search + segmentation params
    ap.add_argument("--top_frac", type=float, default=0.30)
    ap.add_argument("--right_frac", type=float, default=0.50)
    ap.add_argument("--min_score", type=float, default=0.60, help="Min template match score to accept")
    ap.add_argument("--yellow_thr", type=int, default=40, help="Yellow score threshold")
    ap.add_argument("--pad", type=int, default=2, help="Pad around detected digit mask bbox")
    ap.add_argument("--gap_thr", type=int, default=2, help="Merge x-runs if gap <= this many px")

    ap.add_argument("--debug_out", default="debug/calib_anchor_digits.png")
    ap.add_argument("--show", action="store_true", default=False)

    args = ap.parse_args()
    random.seed(args.seed)

    template = cv2.imread(args.anchor_png, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(args.anchor_png)

    # Load calibration frame
    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(args.image)
        frame_idx = None
        fps = None
    else:
        if not args.video:
            raise ValueError("Provide --image or --video")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            cap.release()
            raise RuntimeError("Could not read frame count.")
        frame_idx = args.frame if args.frame is not None else random.randrange(0, total)
        frame_idx = max(0, min(total - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx}")

    # Find anchor
    found = find_anchor_multiscale(
        frame, template,
        top_frac=args.top_frac,
        right_frac=args.right_frac,
        scales=(0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
        method=cv2.TM_CCOEFF_NORMED,
    )
    if found is None:
        raise RuntimeError("Anchor template matching failed (no scales fit).")

    score, anchor_xywh, scale_used, search_xywh = found
    if score < args.min_score:
        raise RuntimeError(f"Anchor score too low: {score:.3f} < {args.min_score}")

    ax, ay, aw, ah = anchor_xywh

    # Define a search window to the right of anchor where digits should be
    # (heuristics; scale with anchor size)
    H, W = frame.shape[:2]
    sx0 = clamp(ax + aw, 0, W - 1)
    sx1 = clamp(ax + int(round(aw + 6.0 * ah)), 0, W)
    sy0 = clamp(ay - int(round(0.40 * ah)), 0, H - 1)
    sy1 = clamp(ay + int(round(1.40 * ah)), 0, H)

    digit_search = frame[sy0:sy1, sx0:sx1].copy()
    if digit_search.size == 0:
        raise RuntimeError("Digit search ROI collapsed; check heuristics.")

    # Yellow mask in digit search
    mask = yellow_score_mask(digit_search, thr=args.yellow_thr)

    # Clean up a little
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # Find bbox of yellow pixels
    bb = bbox_from_mask(mask, pad=args.pad)
    if bb is None:
        raise RuntimeError("No yellow pixels found near anchor; try lowering --yellow_thr or choose a clearer frame.")
    mx0, my0, mx1, my1 = bb

    # Absolute bbox of full timer text (digits+seps) in frame coords
    text_box = (sx0 + mx0, sy0 + my0, sx0 + mx1, sy0 + my1)

    # Segment glyphs within text box using x-runs
    mask_text = mask[my0:my1, mx0:mx1]
    runs = segment_columns_to_runs(mask_text, min_on=1)
    runs = merge_close_runs(runs, gap_thr=args.gap_thr)
    runs9 = choose_9_segments(runs, target=9)

    if runs9 is None:
        # Fallback: uniform split into 9 regions
        Wt = mask_text.shape[1]
        step = Wt / 9.0
        runs9 = []
        for i in range(9):
            x0 = int(round(i * step))
            x1 = int(round((i + 1) * step))
            if x1 <= x0:
                x1 = x0 + 1
            runs9.append((x0, x1))

    # Build boxes: 9 glyph boxes in text-box coords
    glyph_boxes = runs_to_boxes(runs9, 0, mask_text.shape[0])  # y spans full text height
    # Convert to frame coords
    glyph_boxes_frame = []
    for (gx0, gy0, gx1, gy1) in glyph_boxes:
        x0 = text_box[0] + gx0
        x1 = text_box[0] + gx1
        y0 = text_box[1] + gy0
        y1 = text_box[1] + gy1
        glyph_boxes_frame.append((x0, y0, x1, y1))

    # Map to semantics: d0 d1 ':' d2 d3 '.' d4 d5 d6
    d0 = glyph_boxes_frame[0]
    d1 = glyph_boxes_frame[1]
    colon = glyph_boxes_frame[2]
    d2 = glyph_boxes_frame[3]
    d3 = glyph_boxes_frame[4]
    dot = glyph_boxes_frame[5]
    d4 = glyph_boxes_frame[6]
    d5 = glyph_boxes_frame[7]
    d6 = glyph_boxes_frame[8]
    digit_boxes = [d0, d1, d2, d3, d4, d5, d6]

    # Write JSON relative to anchor
    out = {
        "anchor": {
            "template_path": str(args.anchor_png),
            "match_region": {"top_frac": args.top_frac, "right_frac": args.right_frac},
            "method": "TM_CCOEFF_NORMED",
            "min_score": args.min_score,
        },
        "digit_search_relative": {
            # for debugging/traceability; search window used during calibration (relative to anchor height)
            "sx0": (sx0 - ax) / float(max(1, ah)),
            "sy0": (sy0 - ay) / float(max(1, ah)),
            "sx1": (sx1 - ax) / float(max(1, ah)),
            "sy1": (sy1 - ay) / float(max(1, ah)),
        },
        "timer_text_box_rel": rel_box_from_anchor(anchor_xywh, text_box),
        "separators_rel": {
            "colon": rel_box_from_anchor(anchor_xywh, colon),
            "dot": rel_box_from_anchor(anchor_xywh, dot),
        },
        "digits_rel": [rel_box_from_anchor(anchor_xywh, b) for b in digit_boxes],
        "notes": "Boxes stored relative to anchor top-left using anchor height as the scale unit.",
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ Wrote calibration JSON: {out_path}")

    # Debug visualization
    dbg = frame.copy()
    # draw search region
    sx, sy, sw, sh = search_xywh
    draw_box(dbg, (sx, sy, sx + sw, sy + sh), (128, 128, 128), 2, "search")
    # draw anchor
    draw_box(dbg, (ax, ay, ax + aw, ay + ah), (0, 0, 255), 2, f"IGT score={score:.3f}")
    # draw digit_search
    draw_box(dbg, (sx0, sy0, sx1, sy1), (255, 0, 0), 2, "digit_search")
    # draw text box
    draw_box(dbg, text_box, (0, 255, 255), 2, "timer_text")
    # draw digits
    for i, b in enumerate(digit_boxes):
        draw_box(dbg, b, (0, 255, 0), 2, f"d{i}")
    # draw separators
    draw_box(dbg, colon, (255, 255, 0), 2, ":")
    draw_box(dbg, dot, (255, 255, 0), 2, ".")

    dbg_out = Path(args.debug_out)
    dbg_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg_out), dbg)
    print(f"Saved debug image: {dbg_out}")

    if args.show:
        cv2.imshow("calibration", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
