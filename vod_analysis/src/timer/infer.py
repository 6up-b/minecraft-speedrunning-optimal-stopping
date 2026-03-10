# src/timer/infer.py
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import TinyCharCNN
from .decode import topk_constrained_decode_timer
from .geometry import (
    TimerLayout,
    load_timer_layout,
    clamp,
    clamp01,
    match_anchor_multiscale,
    timer_xyxy_from_anchor,
    crop_xyxy,
    digit_boxes_excluding_seps,
    letterbox,
    prep_digit_for_model,
)

DIGIT_LABELS = [str(i) for i in range(10)]


def parse_time_str(s: str) -> Optional[float]:
    try:
        mm, rest = s.split(":")
        ss, ms = rest.split(".")
        return int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except Exception:
        return None


# Keep old private names as aliases for any external callers
_match_anchor_multiscale = match_anchor_multiscale
_timer_xyxy_from_anchor = timer_xyxy_from_anchor
_crop_xyxy = crop_xyxy
_digit_boxes_excluding_seps = digit_boxes_excluding_seps
_letterbox = letterbox
_prep_digit_for_model = prep_digit_for_model


@torch.no_grad()
def infer_timer(
    frame_bgr: np.ndarray,
    model: TinyCharCNN,
    layout: TimerLayout,
    device: str = "cpu",
    pad_px: int = 0,
    decode_k: int = 3,
    ambiguous_margin: float = 0.15,
    constrain_d2_0to5: bool = True,
    constrain_d0_0to5: bool = False,
    anchor_template: Optional[np.ndarray] = None,
) -> Dict:
    """
    Returns:
      {
        "text": "mm:ss.mmm" or None,
        "seconds": float or None,
        "conf": float,                  # mean greedy digit conf
        "decode_score": float,          # constrained decode score
        "decode_mode": "greedy"|"topk"|None,
        "anchor_score": float,
        "anchor_bbox": (x,y,w,h) or None,
        "timer_bbox": (x0,y0,x1,y1) or None,
        "digit_bboxes": [(x0,y0,x1,y1), ...] len=7,
        "digit_confs": [..] len=7,      # greedy top1 confs
        "digit_preds": [..] len=7,      # decoded digits as strings
        "greedy_text": str or None,
        "decode_info": dict,
      }
    """
    if anchor_template is not None:
        tmpl = anchor_template
    else:
        tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
        if tmpl is None:
            raise FileNotFoundError(layout.anchor_template_path)

    found = match_anchor_multiscale(
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

    timer_xyxy = timer_xyxy_from_anchor(anchor_xywh, layout)
    timer_crop = crop_xyxy(frame_bgr, timer_xyxy)
    if timer_crop is None:
        return {
            "text": None, "seconds": None, "conf": 0.0,
            "anchor_score": float(anchor_score), "anchor_bbox": anchor_xywh,
            "timer_bbox": timer_xyxy, "digit_bboxes": [],
            "digit_confs": [], "digit_preds": [],
        }

    digit_boxes = digit_boxes_excluding_seps(
        timer_xyxy=timer_xyxy,
        bounds_x_norm=layout.digit_bounds,
        colon_band=layout.colon_band,
        dot_band=layout.dot_band,
        pad_px=pad_px,
    )

    xs = []
    for bb in digit_boxes:
        crop = crop_xyxy(frame_bgr, bb)
        if crop is None:
            continue
        xs.append(prep_digit_for_model(crop, out_size=32))

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
