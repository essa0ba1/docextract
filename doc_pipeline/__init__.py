"""
pdf_pipeline
============

PDF -> markdown extraction pipeline:

    pdfplumber (words+bboxes, image)
        -> [empty words? -> OCR] -> match against pp-doclayout boxes
        -> per-region markdown, with table regions routed through TableFormerONNX -> OTSL -> markdown

See README.md for the full architecture diagram and a usage walkthrough.
"""

from .layout import (
    BBox,
    DocLayoutResult,
    DocLayoutV3,
    LayoutBox,
    PlacedWord,
    RegionText,
    align_ocr_to_layout,
    align_tokens_to_layout,
    align_words_to_layout,
    log_layout_result,
    pdfplumber_tokens_in_image_space,
    print_layout_result,
)
from .logging_config import setup_pipeline_logging
from .ocr_backends import (
    OCRBackend,
    PytesseractBackend,
    RapidOCRBackend,
    get_ocr_backend,
)
from .pipeline import (
    IMAGE_EXTENSIONS,
    available_cpu_count,
    compute_worker_count,
    crop_and_save_image,
    process_document,
    process_images,
    process_image_page,
    process_pdf,
    process_pdf_page,
    region_to_markdown,
    regions_to_markdown,
)
from .table_extraction import (
    TableFormerONNX,
    ocr_anchor_cells,
    otsl_to_markdown,
    seq_to_otsl,
    table_image_to_otsl,
)

__all__ = [
    "BBox",
    "DocLayoutResult",
    "DocLayoutV3",
    "LayoutBox",
    "PlacedWord",
    "RegionText",
    "align_ocr_to_layout",
    "align_tokens_to_layout",
    "align_words_to_layout",
    "pdfplumber_tokens_in_image_space",
    "log_layout_result",
    "print_layout_result",
    "OCRBackend",
    "PaddleOCRBackend",
    "PytesseractBackend",
    "RapidOCRBackend",
    "get_ocr_backend",
    "available_cpu_count",
    "compute_worker_count",
    "crop_and_save_image",
    "IMAGE_EXTENSIONS",
    "process_document",
    "process_images",
    "process_image_page",
    "process_pdf",
    "process_pdf_page",
    "region_to_markdown",
    "regions_to_markdown",
    "TableFormerONNX",
    "ocr_anchor_cells",
    "otsl_to_markdown",
    "seq_to_otsl",
    "table_image_to_otsl",
    "setup_pipeline_logging",
]
