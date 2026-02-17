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
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\.\d{2,3}$")  # e.g. 2:36.001 or 02:36.00


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def class_name(ch: str) -> str:
    if ch == ":":
        return "colon"
    if ch == ".":
        return "dot"
    return ch  # digit


def load_timer_roi(roi_json_path: Path):
    cfg = json.loads(roi_json_path.read_text(encoding="utf-8"))
    t = cfg["timer"]
    x0 = float(t["x0"]); y0 = float(t["y0"]); x1 = float(t["x1"]); y1 = float(t["y1"])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid timer ROI coords in roi.json")
    return x0, y0, x1, y1


def crop_norm(frame_bgr, x0, y0, x1, y1):
    h, w = frame_bgr.shape[:2]
    xa = int(x0 * w); xb = int(x1 * w)
    ya = int(y0 * h); yb = int(y1 * h)
    xa = clamp(xa, 0, w - 1)
    xb = clamp(xb, 0, w)
    ya = clamp(ya, 0, h - 1)
    yb = clamp(yb, 0, h)
    if xb <= xa or yb <= ya:
        return None
    return frame_bgr[ya:yb, xa:xb].copy()


def preprocess_char_crop(crop_bgr: np.ndarray, out_size=32):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # DO NOT threshold
    # DO NOT invert
    # it just makes the model DOGSHIT

    h, w = gray.shape[:2]
    m = max(h, w)
    sq = np.zeros((m, m), dtype=np.uint8)
    y0 = (m - h) // 2
    x0 = (m - w) // 2
    sq[y0:y0+h, x0:x0+w] = gray

    sq = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return sq


def bbox_from_vertices(verts, W, H, pad=2):
    xs = [v.get("x", 0) for v in verts]
    ys = [v.get("y", 0) for v in verts]
    x0 = clamp(min(xs) - pad, 0, W - 1)
    y0 = clamp(min(ys) - pad, 0, H - 1)
    x1 = clamp(max(xs) + pad, 0, W - 1)
    y1 = clamp(max(ys) + pad, 0, H - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def call_vision_on_bgr(client: vision.ImageAnnotatorClient, image_bgr: np.ndarray):
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    image = vision.Image(content=buf.tobytes())
    resp = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(f"Vision API error: {resp.error.message}")
    return vision.AnnotateImageResponse.to_dict(resp)


def extract_symbols_and_text(vision_json: dict):
    """
    Returns:
      - full_text (str or None)
      - symbols: list of {ch, conf, verts}
    """
    fta = vision_json.get("full_text_annotation", {})
    full_text = fta.get("text", None)

    symbols = []
    for page in fta.get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    for sym in word.get("symbols", []):
                        ch = sym.get("text", "")
                        conf = float(sym.get("confidence", 0.0))
                        verts = sym.get("bounding_box", {}).get("vertices", [])
                        symbols.append({"ch": ch, "conf": conf, "verts": verts})

    return full_text, symbols


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
    ap.add_argument("--out_root", default="data/timer_chars")
    ap.add_argument("--target", type=int, default=200, help="Target per class (total across train+val)")
    ap.add_argument("--val_ratio", type=float, default=0.10, help="Fraction of samples per class to send to val")
    ap.add_argument("--min_conf", type=float, default=0.90, help="Min Vision symbol confidence")
    ap.add_argument("--pad", type=int, default=2, help="Padding around Vision symbol box (pixels)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_attempts", type=int, default=50000, help="Max random frames to try")
    ap.add_argument("--sample_without_replacement", action="store_true", default=True)
    ap.add_argument("--require_full_timer_pattern", action="store_true", default=True,
                    help="Require full_text_annotation.text to look like a timer (recommended)")
    ap.add_argument("--dump_some_debug", type=int, default=0,
                    help="If >0, saves up to N debug ROI images + json into out_root/debug/")
    args = ap.parse_args()

    random.seed(args.seed)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count from video.")

    x0, y0, x1, y1 = load_timer_roi(Path(args.roi))
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    debug_dir = out_root / "debug"
    if args.dump_some_debug > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Per-class totals across train+val
    counts = {c: 0 for c in CLASSES}
    # Per-class val quotas
    val_target = {c: int(round(args.target * args.val_ratio)) for c in CLASSES}
    val_counts = {c: 0 for c in CLASSES}

    client = vision.ImageAnnotatorClient()

    used_frames = set()
    debug_saved = 0

    def pick_split(cls):
        # Fill val up to quota, then train
        if val_counts[cls] < val_target[cls]:
            val_counts[cls] += 1
            return "val"
        return "train"

    attempts = 0
    saved_total = 0

    while attempts < args.max_attempts and not done(counts, args.target):
        attempts += 1

        # Random frame index (prefer without replacement)
        if args.sample_without_replacement and len(used_frames) < total_frames:
            while True:
                idx = random.randrange(0, total_frames)
                if idx not in used_frames:
                    used_frames.add(idx)
                    break
        else:
            idx = random.randrange(0, total_frames)

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        roi = crop_norm(frame, x0, y0, x1, y1)
        if roi is None or roi.size == 0:
            continue

        # Vision OCR on ROI
        vision_json = call_vision_on_bgr(client, roi)
        if vision_json is None:
            continue

        full_text, symbols = extract_symbols_and_text(vision_json)

        if args.require_full_timer_pattern:
            if not full_text:
                continue
            txt = full_text.strip()
            # Strip common stray whitespace/newlines
            txt = txt.replace("\n", "").replace(" ", "")
            if not TIME_RE.match(txt):
                continue

        # Optional debug dump
        if args.dump_some_debug > 0 and debug_saved < args.dump_some_debug:
            t = (idx / fps) if fps > 0 else None
            cv2.imwrite(str(debug_dir / f"roi_f{idx}_t{t}.png"), roi)
            (debug_dir / f"roi_f{idx}.json").write_text(json.dumps(vision_json, indent=2), encoding="utf-8")
            debug_saved += 1

        H, W = roi.shape[:2]

        # Save any needed symbols from this frame
        frame_saved = 0
        for s in symbols:
            ch = s["ch"]
            conf = float(s["conf"])
            if ch not in ALLOWED:
                continue
            if conf < args.min_conf:
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

            proc = preprocess_char_crop(crop, out_size=32)

            split = pick_split(cls)
            out_path = out_root / split / cls / f"f{idx}_a{attempts}_c{conf:.3f}.png"
            cv2.imwrite(str(out_path), proc)

            counts[cls] += 1
            saved_total += 1
            frame_saved += 1

            if done(counts, args.target):
                break

        # Print occasional progress
        if attempts % 250 == 0:
            short = " ".join([f"{k}:{counts[k]}" for k in ["0","1","2","3","4","5","6","7","8","9","colon","dot"]])
            print(f"[attempts={attempts} saved={saved_total} used_frames={len(used_frames)}] {short}")

        # Enforce “informative frames” a bit: if we got nothing, we just continue.
        # (We already ensured timer-pattern match + min_conf filters.)

    cap.release()

    print("\n=== DONE ===")
    print(f"Attempts: {attempts}")
    print(f"Used unique frames: {len(used_frames)} / {total_frames}")
    print("Counts per class:")
    for c in CLASSES:
        print(f"  {c:>5}: {counts[c]}  (val {val_counts[c]}/{val_target[c]})")

    if not done(counts, args.target):
        print("\nDid not reach target for all classes.")
        print("Try: lowering --min_conf slightly (e.g. 0.85), raising --max_attempts, or disabling --require_full_timer_pattern.")
    else:
        print(f"\nDataset built at: {out_root}")
        print("Train/Val folders are ready for ImageFolder training.")


if __name__ == "__main__":
    main()
