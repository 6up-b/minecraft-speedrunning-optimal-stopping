# test_surya_ocr.py
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


def _extract_text_lines(pred_obj):
    """
    Surya outputs can vary a bit by version/wrapper.
    We try a few common shapes:
      - dict with 'text_lines'
      - list of dicts (pages) each with 'text_lines'
      - object with .text_lines
    Returns: list of line dicts/objects
    """
    if pred_obj is None:
        return []

    # dict page
    if isinstance(pred_obj, dict):
        if "text_lines" in pred_obj:
            return pred_obj["text_lines"]
        # sometimes nested like {"pages":[{...}]}
        for k in ("pages", "page", "results"):
            v = pred_obj.get(k, None)
            if isinstance(v, list) and v and isinstance(v[0], dict) and "text_lines" in v[0]:
                return v[0]["text_lines"]

    # list of pages
    if isinstance(pred_obj, list):
        # if it's already text_lines
        if pred_obj and isinstance(pred_obj[0], dict) and "text" in pred_obj[0] and ("bbox" in pred_obj[0] or "polygon" in pred_obj[0]):
            return pred_obj
        # list of pages dicts
        for item in pred_obj:
            if isinstance(item, dict) and "text_lines" in item:
                return item["text_lines"]
            if hasattr(item, "text_lines"):
                return list(item.text_lines)

    # object page
    if hasattr(pred_obj, "text_lines"):
        return list(pred_obj.text_lines)

    return []


def _get_bbox(line):
    """
    Returns (x1,y1,x2,y2) if available, else None.
    """
    if line is None:
        return None
    if isinstance(line, dict):
        bb = line.get("bbox", None)
        if bb and len(bb) == 4:
            return tuple(int(round(v)) for v in bb)
        # polygon -> bbox fallback
        poly = line.get("polygon", None)
        if poly and len(poly) == 4:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return None

    # object-like
    if hasattr(line, "bbox") and line.bbox is not None and len(line.bbox) == 4:
        return tuple(int(round(v)) for v in line.bbox)
    if hasattr(line, "polygon") and line.polygon is not None and len(line.polygon) == 4:
        xs = [p[0] for p in line.polygon]
        ys = [p[1] for p in line.polygon]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    return None


def _get_text_conf(line):
    if isinstance(line, dict):
        return line.get("text", ""), float(line.get("confidence", 0.0))
    txt = getattr(line, "text", "")
    conf = getattr(line, "confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    return txt, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to local image (png/jpg)")
    ap.add_argument("--out", default="surya_ocr_annotated.png", help="Where to save annotated output image")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Only draw/print lines with conf >= this")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    image = Image.open(img_path).convert("RGB")

    # --- Surya predictors (per docs) ---
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()

    # recognition_predictor expects a list of images
    preds = recognition_predictor([image], det_predictor=detection_predictor)

    # Try to peel down to the first image prediction
    # Common shapes: preds[0] or preds[0][0]
    pred0 = None
    if isinstance(preds, list) and preds:
        pred0 = preds[0]
        # sometimes it's list-of-pages
        if isinstance(pred0, list) and pred0:
            # pages list; take first page
            pred0_page0 = pred0[0]
            lines = _extract_text_lines(pred0_page0)
        else:
            lines = _extract_text_lines(pred0)
    else:
        lines = _extract_text_lines(preds)

    # --- Draw ---
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    print(f"Found {len(lines)} text lines\n")

    kept = 0
    for i, line in enumerate(lines):
        text, conf = _get_text_conf(line)
        if conf < args.min_conf:
            continue
        bbox = _get_bbox(line)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        kept += 1

        # print to console
        print(f"[{i:03d}] conf={conf:.3f} bbox=({x1},{y1},{x2},{y2}) text={text!r}")

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # label background + text
        label = f"{conf:.2f} {text}"
        tx, ty = x1, max(0, y1 - 16)
        # simple background
        tw, th = draw.textbbox((tx, ty), label, font=font)[2:]
        draw.rectangle([tx, ty, tx + tw + 6, ty + 16], fill=(0, 0, 0))
        draw.text((tx + 3, ty), label, fill=(255, 255, 255), font=font)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print(f"\nKept/drew {kept} lines (min_conf={args.min_conf}).")
    print(f"Saved annotated image to: {out_path}")


if __name__ == "__main__":
    main()