"""
Toast template matching via grayscale cv2.matchTemplate.

Loads all PNG templates from a folder (e.g. templates/toast_templates/)
and matches each against the toast ROI using multi-scale grayscale
TM_CCOEFF_NORMED.

True positive scores are ~0.93+, false positives are ~0.33, so a
threshold of 0.55 cleanly separates them.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Toast ROI from configs/roi.json (covers chat + advancement area)
TOAST_ROI_X0 = 0.2359375
TOAST_ROI_Y0 = 0.7064814814814815
TOAST_ROI_X1 = 0.653125
TOAST_ROI_Y1 = 0.9194444444444444


@dataclass
class ToastTemplate:
    name: str
    gray: np.ndarray
    scaled: Tuple[Tuple[float, np.ndarray], ...] = ()


@dataclass
class ToastMatchResult:
    name: str
    score: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) within ROI


def load_toast_templates(
    template_dir: str,
    scales: Tuple[float, ...] = (),
) -> List[ToastTemplate]:
    """Load all PNG templates from a directory.
    Template name is derived from the filename (stem without extension)."""
    d = Path(template_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    templates = []
    for p in sorted(d.glob("*.png")):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        scaled = tuple(
            (
                s,
                cv2.resize(
                    gray,
                    (int(round(gray.shape[1] * s)), int(round(gray.shape[0] * s))),
                    interpolation=cv2.INTER_AREA,
                ),
            )
            for s in scales
            if int(round(gray.shape[1] * s)) >= 4 and int(round(gray.shape[0] * s)) >= 4
        )
        templates.append(ToastTemplate(
            name=p.stem,
            gray=gray,
            scaled=scaled,
        ))

    if not templates:
        raise FileNotFoundError(f"No PNG templates found in {template_dir}")
    return templates


def crop_toast_roi(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop the toast ROI (from configs/roi.json calibration).
    Returns (roi_bgr, (x0, y0, w, h)) in full-frame coords."""
    h, w = frame_bgr.shape[:2]
    x0 = int(TOAST_ROI_X0 * w)
    y0 = int(TOAST_ROI_Y0 * h)
    x1 = int(TOAST_ROI_X1 * w)
    y1 = int(TOAST_ROI_Y1 * h)
    roi = frame_bgr[y0:y1, x0:x1]
    return roi, (x0, y0, x1 - x0, y1 - y0)


def match_single_template(
    roi_gray: np.ndarray,
    template: ToastTemplate,
    min_score: float = 0.55,
    scales: Tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20),
) -> Optional[ToastMatchResult]:
    if roi_gray is None or roi_gray.size == 0:
        return None

    best = None
    variants = template.scaled
    if not variants:
        variants = tuple((s, template.gray) for s in scales)

    for scale, tmpl_s in variants:
        if not template.scaled and scale != 1.0:
            tw = int(round(template.gray.shape[1] * scale))
            th = int(round(template.gray.shape[0] * scale))
            if tw < 4 or th < 4:
                continue
            tmpl_s = cv2.resize(template.gray, (tw, th), interpolation=cv2.INTER_AREA)

        th, tw = tmpl_s.shape[:2]
        if th >= roi_gray.shape[0] or tw >= roi_gray.shape[1]:
            continue

        res = cv2.matchTemplate(roi_gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or float(max_val) > best[0]:
            best = (float(max_val), max_loc[0], max_loc[1], tw, th)

    if best is None or best[0] < min_score:
        return None

    score, x, y, w, h = best
    return ToastMatchResult(
        name=template.name,
        score=score,
        bbox=(int(x), int(y), int(w), int(h)),
    )


def match_toast_templates(
    roi_bgr: np.ndarray,
    templates: List[ToastTemplate],
    min_score: float = 0.55,
    scales: Tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20),
) -> List[ToastMatchResult]:
    """
    Match all templates against the toast ROI.

    Returns a list of ToastMatchResult for every template that exceeds
    min_score (may be empty).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    for tmpl in templates:
        match = match_single_template(
            roi_gray=roi_gray,
            template=tmpl,
            min_score=min_score,
            scales=scales,
        )
        if match is not None:
            results.append(match)

    return results
