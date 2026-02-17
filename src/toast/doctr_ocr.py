from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from doctr.models import ocr_predictor


@dataclass
class OCRResult:
    text: Optional[str]
    conf: float


class ToastOCRDoctr:
    """
    Doctr-based OCR for toasts/achievements in a fixed ROI.
    Uses PyTorch backend and returns joined text + average word confidence.
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        assume_straight_pages: bool = True,
        device: str = "cpu",
    ):
        # doctr auto-selects torch backend based on install extra.
        # device string is accepted by predictor for torch backend in most versions;
        # if not, it will default to current torch device.
        self.predictor = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True,
            assume_straight_pages=assume_straight_pages,
        )

        # Best-effort device placement (doctr versions differ slightly)
        try:
            import torch
            if device.startswith("cuda") and torch.cuda.is_available():
                self.predictor.det_predictor.model.to(device)
                self.predictor.reco_predictor.model.to(device)
        except Exception:
            # If doctr internals differ, we still run on default device.
            pass

    def infer(self, roi_bgr: np.ndarray) -> OCRResult:
        """
        roi_bgr: OpenCV BGR uint8 image
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return OCRResult(text=None, conf=0.0)

        # doctr expects RGB images as numpy arrays (H,W,3) uint8
        roi_rgb = roi_bgr[..., ::-1].copy()

        # predictor returns a Document object; we can export to a dict
        doc = self.predictor([roi_rgb])
        exported = doc.export()

        # exported["pages"][0]["blocks"][...]["lines"][...]["words"][...]
        try:
            page = exported["pages"][0]
            words: List[Tuple[str, float]] = []

            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    for w in line.get("words", []):
                        val = (w.get("value") or "").strip()
                        conf = float(w.get("confidence", 0.0))
                        if val:
                            words.append((val, conf))

            if not words:
                return OCRResult(text=None, conf=0.0)

            # Reconstruct readable text by lines (prefer line grouping if available)
            # We'll also compute conf as mean word conf.
            # Optional: Try to join by lines instead of flat words
            lines_text: List[str] = []
            lines_conf: List[float] = []

            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    line_words = [(ww.get("value") or "").strip() for ww in line.get("words", [])]
                    line_confs = [float(ww.get("confidence", 0.0)) for ww in line.get("words", [])]
                    line_words = [t for t in line_words if t]
                    if line_words:
                        lines_text.append(" ".join(line_words))
                        if line_confs:
                            lines_conf.append(sum(line_confs) / max(1, len(line_confs)))

            text = " | ".join(lines_text) if lines_text else " ".join([w[0] for w in words])
            conf = float(sum([w[1] for w in words]) / max(1, len(words)))

            return OCRResult(text=text, conf=conf)

        except Exception:
            # If export structure changes or something goes wrong
            return OCRResult(text=None, conf=0.0)
