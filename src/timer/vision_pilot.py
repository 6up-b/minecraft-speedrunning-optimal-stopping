import argparse
import json
from pathlib import Path

from google.cloud import vision


def annotate_image(image_path: Path, mode: str):
    """
    mode:
      - 'document' => DOCUMENT_TEXT_DETECTION (usually best for symbol boxes)
      - 'text'     => TEXT_DETECTION
    """
    client = vision.ImageAnnotatorClient()

    content = image_path.read_bytes()
    image = vision.Image(content=content)

    if mode == "document":
        response = client.document_text_detection(image=image)
    elif mode == "text":
        response = client.text_detection(image=image)
    else:
        raise ValueError("mode must be 'document' or 'text'")

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    # Convert to a plain JSON-serializable dict
    # (protobuf -> dict)
    return vision.AnnotateImageResponse.to_dict(response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an input image (e.g. a timer ROI crop png)")
    ap.add_argument("--out", default="vision.json", help="Where to write the JSON response")
    ap.add_argument("--mode", choices=["document", "text"], default="document",
                    help="Use 'document' for fullTextAnnotation (best), or 'text' for basic OCR")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    data = annotate_image(image_path, args.mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Helpful quick peek
    print(f"✅ Wrote Vision JSON to: {out_path}")

    # Print top-level summary
    if "full_text_annotation" in data and data["full_text_annotation"]:
        txt = data["full_text_annotation"].get("text", "")
        print("\n--- fullTextAnnotation.text ---")
        print(txt.strip()[:500])
    elif "text_annotations" in data and data["text_annotations"]:
        print("\n--- textAnnotations[0].description ---")
        print(data["text_annotations"][0].get("description", "").strip()[:500])
    else:
        print("\n(No text found in response.)")


if __name__ == "__main__":
    main()
