import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from google.cloud import vision

from src.timer.model import TinyCharCNN

TIME_RE = re.compile(r"^\d{2}:\d{2}\.\d{3}$")


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
    cfg = json.loads(timer_digit_json_path.read_text(encoding="utf-8"))
    bounds = [float(b) for b in cfg["digit_bounds_x_norm_within_timer"]]
    seps = cfg["separators_x_norm_within_timer"]
    colon = [float(seps["colon"][0]), float(seps["colon"][1])]
    dot = [float(seps["dot"][0]), float(seps["dot"][1])]
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

        x0 = max(0, min(W - 1, x0 - pad_px))
        x1 = max(0, min(W,     x1 + pad_px))

        if i == 1:
            x1 = min(x1, colon0_px)
        elif i == 2:
            x0 = max(x0, colon1_px)
        elif i == 3:
            x1 = min(x1, dot0_px)
        elif i == 4:
            x0 = max(x0, dot1_px)

        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W, x1))
        if x1 <= x0:
            return None
        crops.append(timer_roi_bgr[:, x0:x1].copy())
    return crops


def call_vision_full_text(client, roi_bgr):
    ok, buf = cv2.imencode(".png", roi_bgr)
    if not ok:
        return None
    image = vision.Image(content=buf.tobytes())
    resp = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    vjson = vision.AnnotateImageResponse.to_dict(resp)
    fta = vjson.get("full_text_annotation", {})
    txt = fta.get("text", None)
    if not txt:
        return None
    txt = txt.strip().replace("\n", "").replace(" ", "")
    return txt


def timer_text_to_digits(txt):
    if txt is None:
        return None
    if not TIME_RE.match(txt):
        return None
    digs = [c for c in txt if c.isdigit()]
    return digs if len(digs) == 7 else None


def build_gray_edge_tensor(digit_bgr, out_size=32):
    # raw gray channel
    gray = cv2.cvtColor(digit_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # edge channel: use Canny on a "yellowish mask" proxy
    # (works even if digit includes some yellow hue)
    hsv = cv2.cvtColor(digit_bgr, cv2.COLOR_BGR2HSV)
    # wide yellow range
    lower = np.array([15, 40, 80], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    edges = cv2.Canny(mask, 50, 150)
    # normalize to 0..1
    g = gray.astype(np.float32) / 255.0
    e = edges.astype(np.float32) / 255.0

    x = np.stack([g, e], axis=0)  # [2,H,W]
    return torch.from_numpy(x)


def digits_to_timer_str(digs7):
    # digs7 = [d0..d6]
    return f"{digs7[0]}{digs7[1]}:{digs7[2]}{digs7[3]}.{digs7[4]}{digs7[5]}{digs7[6]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4")
    ap.add_argument("--roi", default="configs/roi.json")
    ap.add_argument("--timer_digits", default="configs/timer_digit_bounds.json")
    ap.add_argument("--weights", default="timer_digit_model.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--save_mismatches", default="debug/mismatches")
    args = ap.parse_args()

    random.seed(args.seed)

    x0, y0, x1, y1 = load_timer_roi(Path(args.roi))
    bounds, seps = load_timer_digit_layout(Path(args.timer_digits))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count.")

    # google vision client
    gclient = vision.ImageAnnotatorClient()

    # model
    model = TinyCharCNN(num_classes=10, in_channels=2).to(args.device)
    sd = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(sd)
    model.eval()

    save_dir = Path(args.save_mismatches)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_match = 0
    n_total = 0

    for t in range(args.tries):
        idx = random.randrange(0, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        roi = crop_norm(frame, x0, y0, x1, y1)
        if roi is None:
            continue

        # vision
        vision_txt = call_vision_full_text(gclient, roi)
        vision_digits = timer_text_to_digits(vision_txt)

        # cnn digits
        digit_crops = crop_timer_digits_excluding_seps(roi, bounds, seps, pad_px=args.pad)
        if digit_crops is None:
            continue

        xs = [build_gray_edge_tensor(c, out_size=32) for c in digit_crops]
        x = torch.stack(xs, dim=0).to(args.device)  # [7,2,32,32]
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).detach().cpu().numpy().tolist()
        cnn_digits = [str(p) for p in pred]
        cnn_str = digits_to_timer_str(cnn_digits)

        n_total += 1
        ok_match = (vision_txt == cnn_str) if (vision_txt is not None) else False
        if ok_match:
            n_match += 1

        tsec = idx / fps if fps > 0 else None
        print(f"[{t+1:03d}] frame={idx} t={tsec:.3f}s  vision={vision_txt}  cnn={cnn_str}  match={ok_match}")

        # save mismatch debug
        if (not ok_match) and vision_txt is not None:
            out_base = save_dir / f"f{idx}_vision_{vision_txt.replace(':','_').replace('.','_')}_cnn_{cnn_str.replace(':','_').replace('.','_')}"
            cv2.imwrite(str(out_base) + "_timer.png", roi)
            # save digit crops too
            for j, c in enumerate(digit_crops):
                cv2.imwrite(str(out_base) + f"_d{j}.png", c)

    cap.release()

    if n_total > 0:
        print(f"\nMatch rate: {n_match}/{n_total} = {n_match/n_total:.3f}")
    else:
        print("\nNo valid frames tested.")


if __name__ == "__main__":
    main()
