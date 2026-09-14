"""
ocr_backends.py
================

OCR backends with a common interface: `get_text_boxes(image) -> list[dict]`,
each dict being `{"text": str, "bbox": [x1, y1, x2, y2]}` in the input
image's pixel space.

These backends are used in *two* places in the pipeline (diagram):

1. "Ocr(pytesseract, paddleocr, rapidocr)" — page-level OCR fallback when
   pdfplumber finds no extractable words (scanned pages).
2. Table-cell OCR inside TableFormerONNX's `ocr_anchor_cells` step (see
   table_extraction.py) — same backends, reused as-is.

Unchanged in substance from docling_.ipynb, just split out into its own
module and with the legacy crop-based `.read()` method dropped (it was a
no-op stub for every backend already — `get_text_boxes` was the only one
actually implemented and used).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image
import os
logger = logging.getLogger(__name__)






def _log_ocr_tokens(backend_name: str, image: Image.Image, tokens: list[dict]) -> None:
    logger.debug("OCR (%s) on image %dx%d", backend_name, image.width, image.height)
    logger.info("OCR (%s) returned %d text boxes", backend_name, len(tokens))
    if logger.isEnabledFor(logging.DEBUG):
        for t in tokens[:10]:
            logger.debug("  %r @ %s", t["text"], t["bbox"])


class OCRBackend(ABC):
    @abstractmethod
    def get_text_boxes(self, image: Image.Image) -> list[dict]:
        """Run OCR on the full image once; return all text tokens with
        their global pixel-space bounding boxes [x1, y1, x2, y2]."""
        raise NotImplementedError

    def clone_for_worker(self) -> "OCRBackend":
        """Return an independent instance safe for parallel worker threads."""
        return self


class RapidOCRBackend(OCRBackend):
    def __init__(self, det_model_path, rec_model_path, rec_keys_path):
        from rapidocr_onnxruntime import RapidOCR

        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.rec_keys_path = rec_keys_path
        self._engine = RapidOCR(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            rec_keys_path=rec_keys_path,
        )

    def clone_for_worker(self) -> "RapidOCRBackend":
        return RapidOCRBackend(
            self.det_model_path, self.rec_model_path, self.rec_keys_path
        )

    @staticmethod
    def _to_bgr(image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def get_text_boxes(self, image: Image.Image) -> list[dict]:
        result, _ = self._engine(self._to_bgr(image), use_det=True, use_cls=False)
        tokens = []
        if result:
            for item in result:
                pts = np.array(item[0])  # 4 corners
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                tokens.append(
                    {"text": str(item[1]).strip(), "bbox": [float(x1), float(y1), float(x2), float(y2)]}
                )
        _log_ocr_tokens("rapidocr", image, tokens)
        return tokens


class PytesseractBackend(OCRBackend):
    def __init__(self, lang: str = "fra+eng"):
        import pytesseract

        self._pt = pytesseract
        self.lang = lang

    def clone_for_worker(self) -> "PytesseractBackend":
        return PytesseractBackend(lang=self.lang)

    def get_text_boxes(self, image: Image.Image) -> list[dict]:
        data = self._pt.image_to_data(image, lang=self.lang, output_type=self._pt.Output.DICT)
        tokens = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if text:
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                tokens.append({"text": text, "bbox": [float(x), float(y), float(x + w), float(y + h)]})
        _log_ocr_tokens("pytesseract", image, tokens)
        return tokens




def get_ocr_backend(name: str, **kwargs) -> OCRBackend:
    if name == "rapidocr":
        return RapidOCRBackend(**kwargs)
    if name == "pytesseract":
        return PytesseractBackend(**kwargs)
   
    raise ValueError(f"unknown OCR backend: {name!r} (use 'rapidocr', 'pytesseract')")
