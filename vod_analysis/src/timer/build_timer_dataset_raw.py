import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision

DIGITS = [str(i) for i in range(10)]
TIME_RE = re.compile(r"^\d{2}:\d{2}\.\d{3}$")  # fixed format xx:yy.zzz


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
        raise ValueError("Invalid timer ROI in roi.json")
    return x0, y0, x1, y1


def load_timer_digit_layout(timer_digit_json_path: Path):
    """
    Expects:
      digit_bounds_x_norm_within_timer: len=8 (7 digit slots)
      separators_x_norm_within_timer: {"colon":[x0,x1], "dot":[x0,x1]}
    """
    cfg = json.loads(timer_digit_json_path.read_text(encoding="utf-8"))

    bounds = cfg.get("digit_bounds_x_norm_within_timer", None)
    if bounds is None:
        raise KeyError("Expected key 'digit_bounds_x_norm_within_timer'")
    bounds = [float(b) for b in bounds]
    if len(bounds) != 8:
        raise ValueError(f"Expected 8 boundaries, got {len(bounds)}")
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            raise ValueError("Digit bounds must be strictly increasing")

    seps = cfg.get("separators_x_norm_within_timer", None)
    if seps is None or "colon" not in seps or "dot" not in seps:
        raise KeyError("Expected 'separators_x_norm_within_timer' with keys 'colon' and 'dot'")

    colon = [float(seps["colon"][0]), float(seps["colon"][1])]
    dot = [float(seps["dot"][0]), float(seps["dot"][1])]
    if colon[1] <= colon[0] or dot[1] <= dot[0]:
        raise ValueError("Separator bands must have x1 > x0")

    return bounds, {"colon": colon, "dot": dot}


def crop_norm(frame_bgr, x0, y0, x1, y1):
    h, w = frame_bgr.shape[:2]
    xa = clamp(int(x0 * w), 0, w - 1)
    xb = clamp(int(x1 * w), 0, w)
    ya = clamp(int(y0 * h), 0, h - 1)
    yb = clamp(int(y1 * h), 0, h)
    if xb <= xa or yb <= ya:
        return None
    return frame_bgr[ya:yb, xa:xb].copy()


def crop_timer_digits_excluding_seps(timer_roi_bgr, bounds_x_norm, seps_x_norm, pad_px=0):
    """
    Returns 7 digit crops (d0..d6) for fixed timer format: d0 d1 : d2 d3 . d4 d5 d6
    Excludes separators by trimming adjacent crops using separator bands.
    """
    H, W = timer_roi_bgr.shape[:2]
    colon0, colon1 = seps_x_norm["colon"]
    dot0, dot1 = seps_x_norm["dot"]

    colon0_px = int(round(colon0 * W))
    colon1_px = int(round(colon1 * W))
    dot0_px = int(round(dot0 * W))
    dot1_px = int(round(dot1 * W))

    crops = []
    for i in range(7):
        x0 = int(round(bounds_x_norm[i] * W))
        x1 = int(round(bounds_x_norm[i + 1] * W))

        # pad
        x0 = max(0, min(W - 1, x0 - pad_px))
        x1 = max(0, min(W,     x1 + pad_px))

        # trim separators
        if i == 1:      # before ':'
            x1 = min(x1, colon0_px)
        elif i == 2:    # after ':'
            x0 = max(x0, colon1_px)
        elif i == 3:    # before '.'
            x1 = min(x1, dot0_px)
        elif i == 4:    # after '.'
            x0 = max(x0, dot1_px)

        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W, x1))

        if x1 <= x0:
            return None
        crops.append(timer_roi_bgr[:, x0:x1].copy())
    return crops


def pad_to_square(img_bgr):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    out = np.zeros((m, m, 3), dtype=np.uint8)
    y0 = (m - h) // 2
    x0 = (m - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img_bgr
    return out


def preprocess_raw_color(crop_bgr, out_size=32):
    sq = pad_to_square(crop_bgr)
    sq = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return sq


def call_vision_on_bgr(client, roi_bgr):
    ok, buf = cv2.imencode(".png", roi_bgr)
    if not ok:
        return None
    image = vision.Image(content=buf.tobytes())
    resp = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return vision.AnnotateImageResponse.to_dict(resp)


def extract_full_text(vision_json):
    fta = vision_json.get("full_text_annotation", {})
    full_text = fta.get("text", None)
    if not full_text:
        return None
    return full_text.strip().replace("\n", "").replace(" ", "")


def timer_text_to_7_digits(txt: str):
    """
    For fixed format xx:yy.zzz, return [d0..d6] digits only.
    """
    if txt is None:
        return None
    if not TIME_RE.match(txt):
        return None
    digits = [c for c in txt if c.isdigit()]
    if len(digits) != 7:
        return None
    return digits


def ensure_dirs_digits(root: Path):
    for split in ["train", "val"]:
        for cls in DIGITS:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def done(counts, target):
    return all(counts[c] >= target for c in DIGITS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4")
    ap.add_argument("--roi", default="configs/roi.json")
    ap.add_argument("--timer_digits", default="configs/timer_digit_bounds.json")
    ap.add_argument("--out_root", default="data/timer_digits_raw")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--pad", type=int, default=0, help="Extra px pad when cropping each digit slot")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_attempts", type=int, default=60000)
    ap.add_argument("--require_timer_regex", action="store_true", default=True)
    args = ap.parse_args()

    random.seed(args.seed)

    out_root = Path(args.out_root)
    ensure_dirs_digits(out_root)

    counts = {c: 0 for c in DIGITS}
    val_target = {c: int(round(args.target * args.val_ratio)) for c in DIGITS}
    val_counts = {c: 0 for c in DIGITS}

    def pick_split(cls):
        if val_counts[cls] < val_target[cls]:
            val_counts[cls] += 1
            return "val"
        return "train"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count.")

    x0, y0, x1, y1 = load_timer_roi(Path(args.roi))
    bounds, seps = load_timer_digit_layout(Path(args.timer_digits))
    client = vision.ImageAnnotatorClient()

    used = set()
    attempts = 0
    saved_total = 0

    while attempts < args.max_attempts and not done(counts, args.target):
        attempts += 1

        # random frame, prefer without replacement
        if len(used) < total_frames:
            while True:
                idx = random.randrange(0, total_frames)
                if idx not in used:
                    used.add(idx)
                    break
        else:
            idx = random.randrange(0, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        roi = crop_norm(frame, x0, y0, x1, y1)
        if roi is None:
            continue

        vjson = call_vision_on_bgr(client, roi)
        if vjson is None:
            continue

        txt = extract_full_text(vjson)
        if args.require_timer_regex:
            digits = timer_text_to_7_digits(txt)
            if digits is None:
                continue
        else:
            # salvage: accept any string with 7 digits total
            digs = [c for c in (txt or "") if c.isdigit()]
            if len(digs) != 7:
                continue
            digits = digs

        crops = crop_timer_digits_excluding_seps(roi, bounds_x_norm=bounds, seps_x_norm=seps, pad_px=args.pad)
        if crops is None or len(crops) != 7:
            continue

        for j, ch in enumerate(digits):
            cls = ch  # '0'..'9'
            if counts[cls] >= args.target:
                continue

            proc = preprocess_raw_color(crops[j], out_size=32)
            split = pick_split(cls)

            # store the recognized timer string as provenance
            safe_txt = txt.replace(":", "_").replace(".", "_")
            out_path = out_root / split / cls / f"f{idx}_a{attempts}_d{j}_{safe_txt}.png"
            cv2.imwrite(str(out_path), proc)

            counts[cls] += 1
            saved_total += 1

            if done(counts, args.target):
                break

        if attempts % 500 == 0:
            short = " ".join([f"{k}:{counts[k]}" for k in DIGITS])
            print(f"[attempts={attempts} saved={saved_total} used={len(used)}] {short}")

    cap.release()

    print("\n=== DONE ===")
    print(f"Attempts: {attempts}")
    print(f"Used unique frames: {len(used)} / {total_frames}")
    for c in DIGITS:
        print(f"{c:>2}: {counts[c]} (val {val_counts[c]}/{val_target[c]})")
    if not done(counts, args.target):
        print("\n⚠️ Did not reach target for all digits. Increase --max_attempts or disable --require_timer_regex.")
    else:
        print(f"\n✅ Raw digit dataset built at: {out_root}")


if __name__ == "__main__":
    main()
