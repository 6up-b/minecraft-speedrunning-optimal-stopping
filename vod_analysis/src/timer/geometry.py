# src/timer/geometry.py
"""
Shared geometry utilities for anchor-based timer extraction.
Used by both inference (infer.py) and dataset building (label_timer_digits.py, etc).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TimerLayout:
    anchor_template_path: str
    match_top_frac: float
    match_right_frac: float
    min_score: float
    timer_rel_x0: float
    timer_rel_y0: float
    timer_rel_x1: float
    timer_rel_y1: float
    digit_bounds: List[float]
    colon_band: Tuple[float, float]
    dot_band: Tuple[float, float]
    yellow_thr: int = 40


def load_timer_layout(path: str) -> TimerLayout:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))

    anchor = cfg["anchor"]
    timer_rel = cfg["timer_roi_rel_to_anchor"]
    bounds = cfg["digit_bounds_x_norm_within_timer_roi"]
    seps = cfg.get("separators_x_norm_within_timer_roi", {"colon": [0.0, 0.0], "dot": [0.0, 0.0]})
    region = anchor.get("match_region", {"top_frac": 0.30, "right_frac": 0.50})

    bounds = [float(b) for b in bounds]
    if len(bounds) != 8:
        raise ValueError(f"Expected 8 boundaries for 7 digits, got {len(bounds)}")

    return TimerLayout(
        anchor_template_path=str(anchor.get("template_path")),
        match_top_frac=float(region.get("top_frac", 0.30)),
        match_right_frac=float(region.get("right_frac", 0.50)),
        min_score=float(anchor.get("min_score", 0.65)),
        timer_rel_x0=float(timer_rel["x0"]),
        timer_rel_y0=float(timer_rel["y0"]),
        timer_rel_x1=float(timer_rel["x1"]),
        timer_rel_y1=float(timer_rel["y1"]),
        digit_bounds=bounds,
        colon_band=(float(seps["colon"][0]), float(seps["colon"][1])),
        dot_band=(float(seps["dot"][0]), float(seps["dot"][1])),
        yellow_thr=int(cfg.get("yellow_thr", 40)),
    )


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def match_anchor_multiscale(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    top_frac: float,
    right_frac: float,
    scales=(0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
) -> Optional[Tuple[float, Tuple[int, int, int, int]]]:
    """
    Returns (score, (x,y,w,h)) in full-frame coords, or None.
    """
    H, W = frame_bgr.shape[:2]
    y0 = 0
    y1 = int(round(top_frac * H))
    x0 = int(round(right_frac * W))
    x1 = W

    search = frame_bgr[y0:y1, x0:x1]
    if search.size == 0:
        return None

    search_g = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    tmpl_g0 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    best = None
    for s in scales:
        tw = int(round(tmpl_g0.shape[1] * s))
        th = int(round(tmpl_g0.shape[0] * s))
        if tw < 8 or th < 8:
            continue
        if th >= search_g.shape[0] or tw >= search_g.shape[1]:
            continue

        tmpl_g = cv2.resize(tmpl_g0, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_g, tmpl_g, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best is None or float(max_val) > best[0]:
            ax = x0 + max_loc[0]
            ay = y0 + max_loc[1]
            best = (float(max_val), ax, ay, tw, th)

    if best is None:
        return None
    score, ax, ay, aw, ah = best
    return score, (int(ax), int(ay), int(aw), int(ah))


def timer_xyxy_from_anchor(anchor_xywh, layout: TimerLayout) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = anchor_xywh
    ah = float(max(1, ah))

    x0 = int(round(ax + layout.timer_rel_x0 * ah))
    y0 = int(round(ay + layout.timer_rel_y0 * ah))
    x1 = int(round(ax + layout.timer_rel_x1 * ah))
    y1 = int(round(ay + layout.timer_rel_y1 * ah))
    return x0, y0, x1, y1


def crop_xyxy(img, xyxy) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = xyxy
    x0 = clamp(x0, 0, W - 1)
    x1 = clamp(x1, 0, W)
    y0 = clamp(y0, 0, H - 1)
    y1 = clamp(y1, 0, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


def digit_boxes_excluding_seps(
    timer_xyxy: Tuple[int, int, int, int],
    bounds_x_norm: List[float],
    colon_band: Tuple[float, float],
    dot_band: Tuple[float, float],
    pad_px: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """
    Returns 7 digit boxes for format: d0 d1 : d2 d3 . d4 d5 d6
    Excludes colon from d1/d2 and dot from d3/d4.
    """
    tx0, ty0, tx1, ty1 = timer_xyxy
    W = max(1, tx1 - tx0)

    colon_a = tx0 + int(round(colon_band[0] * W))
    colon_b = tx0 + int(round(colon_band[1] * W))
    dot_a = tx0 + int(round(dot_band[0] * W))
    dot_b = tx0 + int(round(dot_band[1] * W))

    out = []
    for i in range(7):
        x0 = tx0 + int(round(bounds_x_norm[i] * W)) - pad_px
        x1 = tx0 + int(round(bounds_x_norm[i + 1] * W)) + pad_px

        if i == 1:
            x1 = min(x1, colon_a)
        elif i == 2:
            x0 = max(x0, colon_b)
        elif i == 3:
            x1 = min(x1, dot_a)
        elif i == 4:
            x0 = max(x0, dot_b)

        x0 = clamp(x0, tx0, tx1 - 1)
        x1 = clamp(x1, tx0 + 1, tx1)

        if x1 <= x0:
            x0 = tx0 + int(round(bounds_x_norm[i] * W))
            x1 = tx0 + int(round(bounds_x_norm[i + 1] * W))
            x0 = clamp(x0, tx0, tx1 - 1)
            x1 = clamp(x1, tx0 + 1, tx1)

        out.append((x0, ty0, x1, ty1))

    return out


def letterbox(img_bgr: np.ndarray, out_size: int = 32) -> np.ndarray:
    """Resize to fit within out_size x out_size, then center-pad with black."""
    h, w = img_bgr.shape[:2]
    scale = min(out_size / h, out_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    y0 = (out_size - new_h) // 2
    x0 = (out_size - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def prep_digit_for_model(bgr_crop: np.ndarray, out_size=32, yellow_thr: int = 40) -> np.ndarray:
    """
    Match TimerEdgeDataset preprocessing:
      ch0 = raw grayscale
      ch1 = edge magnitude on yellow-masked grayscale
      ch2 = yellow mask
    Returns [3,32,32] float32
    """
    bgr_crop = letterbox(bgr_crop, out_size)

    gray_raw = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    b = bgr_crop[:, :, 0].astype(np.int16)
    g = bgr_crop[:, :, 1].astype(np.int16)
    r = bgr_crop[:, :, 2].astype(np.int16)
    score = np.minimum(r, g) - b
    score = np.clip(score, 0, 255).astype(np.uint8)
    mask = (score >= yellow_thr).astype(np.uint8) * 255

    gray_masked = cv2.bitwise_and(gray_raw, gray_raw, mask=mask)
    gx = cv2.Sobel(gray_masked, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_masked, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    x0 = gray_raw.astype(np.float32) / 255.0
    x1 = mag.astype(np.float32) / 255.0
    x2 = mask.astype(np.float32) / 255.0
    return np.stack([x0, x1, x2], axis=0)
