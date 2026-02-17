# src/timer/infer.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import TinyCharCNN

DIGIT_LABELS = [str(i) for i in range(10)]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_time_str(s: str) -> Optional[float]:
    # expects "mm:ss.mmm" (or m:ss.mmm), returns seconds
    try:
        mm, rest = s.split(":")
        ss, ms = rest.split(".")
        return int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except Exception:
        return None


@dataclass
class TimerLayout:
    anchor_template_path: str
    match_top_frac: float
    match_right_frac: float
    min_score: float
    # timer roi relative to anchor, expressed in units of anchor height
    timer_rel_x0: float
    timer_rel_y0: float
    timer_rel_x1: float
    timer_rel_y1: float
    # 8 boundaries -> 7 digits
    digit_bounds: List[float]
    # separator bands -> exclude from adjacent digits
    colon_band: Tuple[float, float]
    dot_band: Tuple[float, float]


def load_timer_layout(path: str) -> TimerLayout:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))

    # You said you have a "roi template json with the timer position and digit position relative to anchor".
    # This loader expects these keys (adjust here if your JSON differs).
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
    )


def _match_anchor_multiscale(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    top_frac: float,
    right_frac: float,
    scales=(0.80, 0.90, 1.00, 1.10, 1.20),
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

    best = None  # (score, ax, ay, aw, ah)
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


def _timer_xyxy_from_anchor(anchor_xywh, timer_rel: TimerLayout) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = anchor_xywh
    ah = float(max(1, ah))  # scale using anchor height

    x0 = int(round(ax + timer_rel.timer_rel_x0 * ah))
    y0 = int(round(ay + timer_rel.timer_rel_y0 * ah))
    x1 = int(round(ax + timer_rel.timer_rel_x1 * ah))
    y1 = int(round(ay + timer_rel.timer_rel_y1 * ah))
    return x0, y0, x1, y1


def _crop_xyxy(img, xyxy) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = xyxy
    x0 = clamp(x0, 0, W - 1)
    x1 = clamp(x1, 0, W)
    y0 = clamp(y0, 0, H - 1)
    y1 = clamp(y1, 0, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


def _digit_boxes_excluding_seps(
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
    dot_a   = tx0 + int(round(dot_band[0] * W))
    dot_b   = tx0 + int(round(dot_band[1] * W))

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

        # fail-safe fallback (shouldn't happen with good config)
        if x1 <= x0:
            x0 = tx0 + int(round(bounds_x_norm[i] * W))
            x1 = tx0 + int(round(bounds_x_norm[i + 1] * W))
            x0 = clamp(x0, tx0, tx1 - 1)
            x1 = clamp(x1, tx0 + 1, tx1)

        out.append((x0, ty0, x1, ty1))

    return out


def _prep_digit_for_model(bgr_crop: np.ndarray, out_size=32) -> np.ndarray:
    """
    Model was trained on 32x32 RGB crops (TimerEdgeDataset reads BGR from disk then creates channels).
    Here we mimic that logic: create raw grayscale, yellow mask, edge channel, and yellow mask channel (3-ch input).
    """
    # Resize to 32x32 first (same as dataset)
    if bgr_crop.shape[0] != out_size or bgr_crop.shape[1] != out_size:
        bgr_crop = cv2.resize(bgr_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Channel 0: raw grayscale
    gray_raw = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    # Channel 1: edge magnitude on yellow-masked grayscale
    b = bgr_crop[:, :, 0].astype(np.int16)
    g = bgr_crop[:, :, 1].astype(np.int16)
    r = bgr_crop[:, :, 2].astype(np.int16)
    score = np.minimum(r, g) - b
    score = np.clip(score, 0, 255).astype(np.uint8)
    mask = (score >= 40).astype(np.uint8) * 255  # keep consistent with dataset default unless you expose thr

    gray_masked = cv2.bitwise_and(gray_raw, gray_raw, mask=mask)
    gx = cv2.Sobel(gray_masked, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_masked, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    # Channel 2: yellow mask itself (helps separate digits from background)
    mask_u8 = mask

    x0 = gray_raw.astype(np.float32) / 255.0
    x1 = mag.astype(np.float32) / 255.0
    x2 = mask_u8.astype(np.float32) / 255.0
    x = np.stack([x0, x1, x2], axis=0)  # [3,32,32]
    return x


@torch.no_grad()
def infer_timer(
    frame_bgr: np.ndarray,
    model: TinyCharCNN,
    layout: TimerLayout,
    device: str = "cpu",
    pad_px: int = 0,
) -> Dict:
    """
    Returns:
      {
        "text": "mm:ss.mmm" or None,
        "seconds": float or None,
        "conf": float,
        "anchor_score": float,
        "anchor_bbox": (x,y,w,h) or None,
        "timer_bbox": (x0,y0,x1,y1) or None,
        "digit_bboxes": [(x0,y0,x1,y1), ...] len=7,
        "digit_confs": [..] len=7,
        "digit_preds": [..] len=7,
      }
    """
    # load anchor template
    tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
    if tmpl is None:
        raise FileNotFoundError(layout.anchor_template_path)

    found = _match_anchor_multiscale(
        frame_bgr,
        tmpl,
        top_frac=layout.match_top_frac,
        right_frac=layout.match_right_frac,
    )
    if found is None:
        return {
            "text": None, "seconds": None, "conf": 0.0,
            "anchor_score": 0.0, "anchor_bbox": None,
            "timer_bbox": None, "digit_bboxes": [],
            "digit_confs": [], "digit_preds": [],
        }

    anchor_score, anchor_xywh = found
    if anchor_score < layout.min_score:
        return {
            "text": None, "seconds": None, "conf": float(anchor_score),
            "anchor_score": float(anchor_score), "anchor_bbox": anchor_xywh,
            "timer_bbox": None, "digit_bboxes": [],
            "digit_confs": [], "digit_preds": [],
        }

    timer_xyxy = _timer_xyxy_from_anchor(anchor_xywh, layout)
    timer_crop = _crop_xyxy(frame_bgr, timer_xyxy)
    if timer_crop is None:
        return {
            "text": None, "seconds": None, "conf": 0.0,
            "anchor_score": float(anchor_score), "anchor_bbox": anchor_xywh,
            "timer_bbox": timer_xyxy, "digit_bboxes": [],
            "digit_confs": [], "digit_preds": [],
        }

    digit_boxes = _digit_boxes_excluding_seps(
        timer_xyxy=timer_xyxy,
        bounds_x_norm=layout.digit_bounds,
        colon_band=layout.colon_band,
        dot_band=layout.dot_band,
        pad_px=pad_px,
    )

    # prepare model inputs
    xs = []
    valid_boxes = []
    for bb in digit_boxes:
        crop = _crop_xyxy(frame_bgr, bb)
        if crop is None:
            continue
        xs.append(_prep_digit_for_model(crop, out_size=32))
        valid_boxes.append(bb)

    if len(xs) != 7:
        return {
            "text": None, "seconds": None, "conf": 0.0,
            "anchor_score": float(anchor_score), "anchor_bbox": anchor_xywh,
            "timer_bbox": timer_xyxy, "digit_bboxes": digit_boxes,
            "digit_confs": [], "digit_preds": [],
        }

    x = torch.from_numpy(np.stack(xs, axis=0)).to(device=device, dtype=torch.float32)  # [7,3,32,32]

    model = model.to(device)
    model.eval()
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    confs, idxs = probs.max(dim=1)

    preds = [DIGIT_LABELS[int(i)] for i in idxs.cpu().numpy().tolist()]
    digit_confs = confs.cpu().numpy().astype(np.float32).tolist()

    # reconstruct mm:ss.mmm from 7 digits
    # d0 d1 : d2 d3 . d4 d5 d6
    text = f"{preds[0]}{preds[1]}:{preds[2]}{preds[3]}.{preds[4]}{preds[5]}{preds[6]}"

    # overall confidence: mean digit confidence (simple + stable)
    conf = float(np.mean(digit_confs))
    seconds = parse_time_str(text)

    return {
        "text": text,
        "seconds": seconds,
        "conf": conf,
        "anchor_score": float(anchor_score),
        "anchor_bbox": anchor_xywh,
        "timer_bbox": timer_xyxy,
        "digit_bboxes": digit_boxes,
        "digit_confs": digit_confs,
        "digit_preds": preds,
    }
