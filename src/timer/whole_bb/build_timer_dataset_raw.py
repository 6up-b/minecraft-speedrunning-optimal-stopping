import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision

CLASSES = [str(i) for i in range(10)] + ["colon", "dot"]
ALLOWED = set(list("0123456789") + [":", "."])
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\.\d{2,3}$")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def class_name(ch: str) -> str:
    if ch == ":":
        return "colon"
    if ch == ".":
        return "dot"
    return ch

def load_timer_roi(roi_json_path: Path):
    cfg = json.loads(roi_json_path.read_text(encoding="utf-8"))
    t = cfg["timer"]
    x0, y0, x1, y1 = map(float, [t["x0"], t["y0"], t["x1"], t["y1"]])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid timer ROI in roi.json")
    return x0, y0, x1, y1

def crop_norm(frame_bgr, x0, y0, x1, y1):
    h, w = frame_bgr.shape[:2]
    xa = clamp(int(x0 * w), 0, w - 1)
    xb = clamp(int(x1 * w), 0, w)
    ya = clamp(int(y0 * h), 0, h - 1)
    yb = clamp(int(y1 * h), 0, h)
    if xb <= xa or yb <= ya:
        return None
    return frame_bgr[ya:yb, xa:xb].copy()

def bbox_from_vertices(verts, W, H, pad=8):
    xs = [v.get("x", 0) for v in verts]
    ys = [v.get("y", 0) for v in verts]
    x0 = clamp(min(xs) - pad, 0, W - 1)
    y0 = clamp(min(ys) - pad, 0, H - 1)
    x1 = clamp(max(xs) + pad, 0, W - 1)
    y1 = clamp(max(ys) + pad, 0, H - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)

def pad_to_square(img_bgr):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    out = np.zeros((m, m, 3), dtype=np.uint8)
    y0 = (m - h) // 2
    x0 = (m - w) // 2
    out[y0:y0+h, x0:x0+w] = img_bgr
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

def extract_symbols_and_text(vision_json):
    fta = vision_json.get("full_text_annotation", {})
    full_text = fta.get("text", None)
    syms = []
    for page in fta.get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    for s in word.get("symbols", []):
                        syms.append({
                            "ch": s.get("text", ""),
                            "conf": float(s.get("confidence", 0.0)),
                            "verts": s.get("bounding_box", {}).get("vertices", []),
                        })
    return full_text, syms

def ensure_dirs(root: Path):
    for split in ["train", "val"]:
        for cls in CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)

def done(counts, target):
    return all(counts[c] >= target for c in CLASSES)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input.mp4")
    ap.add_argument("--roi", default="configs/roi.json")
    ap.add_argument("--out_root", default="data/timer_chars_raw")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--min_conf", type=float, default=0.90)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_attempts", type=int, default=60000)
    ap.add_argument("--require_timer_regex", action="store_true", default=True)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    counts = {c: 0 for c in CLASSES}
    val_target = {c: int(round(args.target * args.val_ratio)) for c in CLASSES}
    val_counts = {c: 0 for c in CLASSES}

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

        full_text, syms = extract_symbols_and_text(vjson)
        if args.require_timer_regex:
            if not full_text:
                continue
            txt = full_text.strip().replace("\n", "").replace(" ", "")
            if not TIME_RE.match(txt):
                continue

        H, W = roi.shape[:2]

        for s in syms:
            ch = s["ch"]
            conf = float(s["conf"])
            if ch not in ALLOWED or conf < args.min_conf:
                continue

            cls = class_name(ch)
            if counts[cls] >= args.target:
                continue

            bb = bbox_from_vertices(s["verts"], W, H, pad=args.pad)
            if bb is None:
                continue
            xA, yA, xB, yB = bb
            crop = roi[yA:yB, xA:xB]
            if crop.size == 0:
                continue

            proc = preprocess_raw_color(crop, out_size=32)

            split = pick_split(cls)
            out_path = out_root / split / cls / f"f{idx}_a{attempts}_c{conf:.3f}.png"
            cv2.imwrite(str(out_path), proc)

            counts[cls] += 1
            saved_total += 1

            if done(counts, args.target):
                break

        if attempts % 500 == 0:
            short = " ".join([f"{k}:{counts[k]}" for k in ["0","1","2","3","4","5","6","7","8","9","colon","dot"]])
            print(f"[attempts={attempts} saved={saved_total} used={len(used)}] {short}")

    cap.release()

    print("\n=== DONE ===")
    print(f"Attempts: {attempts}")
    print(f"Used unique frames: {len(used)} / {total_frames}")
    for c in CLASSES:
        print(f"{c:>5}: {counts[c]} (val {val_counts[c]}/{val_target[c]})")
    if not done(counts, args.target):
        print("\n⚠️ Did not reach target for all classes. Increase --max_attempts or lower --min_conf.")
    else:
        print(f"\n✅ Raw color dataset built at: {out_root}")

if __name__ == "__main__":
    main()
