"""
pipeline.py
===========

Wires together layout.py, ocr_backends.py, and table_extraction.py into the
exact flow drawn in the diagram:

    pdf -> pdfplumber -> words+bboxes, image
                            |                \\
                 if words empty?            pp-doclayout -> layout boxes
                  yes /        \\no                  |        |
                 OCR        Plumber+layout match     |   OCR+layout match
                  \\__________________________________/________/
                                    |
                         (class_name, bbox, reading_order, text) per region
                                    |
                        if class_name == "table" -> TableFormerONNX -> OTSL -> markdown
                        else                      -> markdown formatted by class_name

Two entry points sit on top of a shared per-image core (`_process_rendered_page`):

- `process_pdf_page` / `process_pdf` — PDF input via pdfplumber. May have
  native words (Plumber-path matching) or not (OCR fallback).
- `process_image_page` / `process_images` — standalone image input
  (.jpg/.png/etc., e.g. a photographed or scanned page with no PDF
  structure at all). These never have native words, so they always go
  through the OCR branch — the same code path a scanned PDF page uses.

`process_document` is a single dispatcher that picks the right one based on
the input (a .pdf path, an image path, or a list of image paths treated as
ordered pages of one document).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from .layout import (
    DocLayoutV3,
    PlacedWord,
    RegionText,
    align_ocr_to_layout,
    align_tokens_to_layout,
    log_layout_result,
    pdfplumber_tokens_in_image_space,
)
from .ocr_backends import OCRBackend
from .table_extraction import TableFormerONNX, otsl_to_markdown, table_image_to_otsl

logger = logging.getLogger(__name__)

# Layout classes that contribute nothing to the output markdown.
SKIP_CLASSES = {"header", "footer", "number", "seal"}

# Layout classes rendered as a cropped image link rather than as text.
IMAGE_LIKE_CLASSES = {"chart", "header_image", "footer_image", "vision_footnote"}

# Extensions routed through the standalone-image path by process_document.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# ORT sessions in layout/table modules typically use up to 4 intra-op threads.
_ORT_THREADS_PER_WORKER = 4
_MAX_WORKERS_CAP = 8


def available_cpu_count() -> int:
    """Return CPUs this process may use (cgroup/affinity-aware when supported)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError, OSError):
        return os.cpu_count() or 1


def compute_worker_count(
    num_pages: int,
    *,
    reserve_cpus: int = 1,
    ort_threads_per_worker: int = _ORT_THREADS_PER_WORKER,
    max_workers_cap: int = _MAX_WORKERS_CAP,
) -> int:
    """
    Choose a parallel worker count from CPU topology.

    Uses cgroup/affinity-aware ``available_cpu_count()``, reserves one CPU for
    the main thread / UI, then picks::

        min(max_cap, num_pages, ort_limited, half_of_usable)

    where ``ort_limited = usable // ort_threads_per_worker`` avoids launching
    more page workers than ONNX can usefully run when each session owns several
    intra-op threads. The ``half_of_usable`` term keeps light parallelism on
    laptops (e.g. 4 cores -> 2 workers for multi-page PDFs).
    """
    if num_pages <= 1:
        return 1

    usable = max(1, available_cpu_count() - reserve_cpus)
    half_pool = max(1, (usable + 1) // 2)
    if usable >= ort_threads_per_worker * 2:
        ort_limited = max(1, usable // ort_threads_per_worker)
        workers = min(ort_limited, half_pool)
    else:
        # Small machines: modest parallelism; ORT may oversubscribe slightly.
        workers = half_pool
    return min(workers, max_workers_cap, num_pages)


@dataclass(frozen=True)
class _RenderedPageJob:
    page_index: int
    order: int
    img: Image.Image
    page_tokens: Optional[List[PlacedWord]]


_thread_local = threading.local()


class _ThreadLocalOcrBackends:
    """One OCR backend clone per worker thread (RapidOCR is not thread-safe)."""

    def __init__(
        self,
        page_ocr_backend: Optional[OCRBackend],
        table_ocr_backend: Optional[OCRBackend],
    ):
        self._page_template = page_ocr_backend
        self._table_template = table_ocr_backend

    def page_ocr(self) -> Optional[OCRBackend]:
        if self._page_template is None:
            return None
        backend = getattr(_thread_local, "page_ocr_backend", None)
        if backend is None:
            backend = self._page_template.clone_for_worker()
            _thread_local.page_ocr_backend = backend
        return backend

    def table_ocr(self) -> Optional[OCRBackend]:
        if self._table_template is None:
            return None
        backend = getattr(_thread_local, "table_ocr_backend", None)
        if backend is None:
            backend = self._table_template.clone_for_worker()
            _thread_local.table_ocr_backend = backend
        return backend


def _process_rendered_page_jobs_parallel(
    jobs: Sequence[_RenderedPageJob],
    layout_detector: DocLayoutV3,
    doc_path: str,
    page_ocr_backend: Optional[OCRBackend],
    table_runner: Optional[TableFormerONNX],
    table_ocr_backend: Optional[OCRBackend],
    max_workers: int,
) -> list[str]:
    ocr_backends = _ThreadLocalOcrBackends(page_ocr_backend, table_ocr_backend)
    markdown_by_order: list[str | None] = [None] * len(jobs)

    def _run(job: _RenderedPageJob) -> None:
        result = _process_rendered_page(
            job.img,
            layout_detector,
            doc_path,
            job.page_index,
            job.page_tokens,
            ocr_backends.page_ocr(),
            table_runner,
            ocr_backends.table_ocr(),
        )
        markdown_by_order[job.order] = result["markdown"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run, job) for job in jobs]
        for future in as_completed(futures):
            future.result()

    if any(chunk is None for chunk in markdown_by_order):
        raise RuntimeError("Parallel page processing did not produce markdown for every page.")
    return [markdown_by_order[i] for i in range(len(jobs))]


def _to_bgr_array(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a BGR ndarray.

    `DocLayoutV3._preprocess` always applies a BGR->RGB channel swap, which
    is only correct if the array it receives is actually BGR (as
    `cv2.imread` produces). PIL images are RGB, so feeding one in directly
    — as the original page-rendering code did — silently double-swaps the
    channels and degrades detection. Converting explicitly here keeps every
    caller (PDF-rendered pages and standalone images alike) consistent with
    what `DocLayoutV3` actually expects from an in-memory array.
    """
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def crop_and_save_image(image: Image.Image, bbox: tuple, index: int, doc_path: str) -> str:
    """Crop `bbox` (xyxy, pixel space) out of `image` and save it under
    `<doc_stem>/images/<index>.png`, returning that path."""
    stem = doc_path.split(".")[0]
    images_dir = os.path.join(stem, "images")
    os.makedirs(images_dir, exist_ok=True)
    cropped = image.crop(bbox)
    out_path = os.path.join(images_dir, f"{index}.png")
    cropped.save(out_path)
    return out_path



def crop_to_base64(image: Image.Image, bbox: tuple, fmt: str = "PNG") -> str:
    """Crop `bbox` (xyxy, pixel space) out of `image` and return a base64
    data URI, ready to embed directly in markdown — no file written to disk."""
    cropped = image.crop(bbox)
    buffer = io.BytesIO()
    cropped.save(buffer, format=fmt)
    b64_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64_data}"


def _table_tokens_from_page(
    bbox: tuple, page_tokens: Optional[List[PlacedWord]]
) -> List[dict]:
    """
    Pull the pdfplumber word tokens whose centroid falls inside a table
    region's `bbox` (xyxy, full-page image pixel space — the same space
    `page_tokens` is already expressed in) and re-express them relative to
    the cropped table image's own origin (top-left of `bbox`), which is the
    coordinate convention `table_image_to_otsl`/`ocr_anchor_cells` expects.

    Returns `[]` if there's no text layer (`page_tokens` is None/empty) or
    no words happen to land inside this particular region — callers should
    fall back to OCR in that case.
    """
    if not page_tokens:
        return []
    xmin, ymin, xmax, ymax = bbox
    local_tokens = []
    for t in page_tokens:
        cx, cy = (t.x0 + t.x1) / 2.0, (t.top + t.bottom) / 2.0
        if xmin <= cx <= xmax and ymin <= cy <= ymax:
            local_tokens.append(
                {"text": t.text, "bbox": [t.x0 - xmin, t.top - ymin, t.x1 - xmin, t.bottom - ymin]}
            )
    return local_tokens


def region_to_markdown(
    region: RegionText,
    index: int,
    img: Image.Image,
    doc_path: str,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
    page_tokens: Optional[List[PlacedWord]] = None,
) -> str:
    label = region.class_name
    logger.debug("Rendering region %d: class=%s bbox=%s", index, label, region.bbox)

    if label in SKIP_CLASSES:
        logger.debug("Skipping region %d (%s)", index, label)
        return ""

    if label == "table":
        if table_runner is not None:
            table_tokens = _table_tokens_from_page(region.bbox, page_tokens)
            tmp_path = None
            try:
                import tempfile
                cropped = img.crop(region.bbox)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cropped.save(tmp, format="PNG")
                    tmp_path = tmp.name

                if table_tokens:
                    logger.info(
                        "Table region %d: using %d pdfplumber tokens (no OCR)",
                        index, len(table_tokens),
                    )
                    otsl = table_image_to_otsl(table_runner, tmp_path, tokens=table_tokens)
                elif table_ocr_backend is not None:
                    logger.info(
                        "Table region %d: no pdfplumber tokens — OCR via %s",
                        index, type(table_ocr_backend).__name__,
                    )
                    otsl = table_image_to_otsl(table_runner, tmp_path, backend=table_ocr_backend)
                else:
                    raise ValueError(
                        "no pdfplumber words found in this table region and no "
                        "table_ocr_backend was provided to fall back on"
                    )
                table_md, df = otsl_to_markdown(otsl)
                if df is not None:
                    logger.info(
                        "Table region %d: extracted %d rows x %d cols (Excel export disabled)",
                        index, df.shape[0], df.shape[1],
                    )
                    logger.debug("Table region %d dataframe:\n%s", index, df)
                    # Excel export intentionally disabled for now.

                return f"{table_md}\n\n"
            except Exception as exc:
                logger.warning(
                    "Table region %d: extraction failed (%s) — falling back to image embed",
                    index, exc,
                )
                image_b64 = crop_to_base64(img, region.bbox)
                return f"<!-- table extraction failed: {exc} -->\n![table]({image_b64})\n\n"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        logger.info("Table region %d: no TableFormer configured — embedding crop", index)
        image_b64 = crop_to_base64(img, region.bbox)
        return f"![table]({image_b64})\n\n"

    if label in IMAGE_LIKE_CLASSES:
        image_b64 = crop_to_base64(img, region.bbox)
        logger.info("Image-like region %d (%s): embedded as base64", index, label)
        return f"![{label}]({image_b64})\n\n"

    text = region.text
    if not text:
        return ""

    if label == "doc_title":
        return f"# {text}\n\n"
    if label == "paragraph_title":
        return f"## {text}\n\n"
    if label == "aside_text":
        return f"> {text}\n\n"
    if label == "figure_title":
        return f"### {text}\n\n"
    if label == "image":
        image_b64 = crop_to_base64(img, region.bbox)
        output = f"![{label}]({image_b64})\n\n"
        if text.strip():
            output += f"{text}\n\n"
        return output

    return f"{text}\n\n"

def regions_to_markdown(
    regions: List[RegionText],
    img: Image.Image,
    doc_path: str,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
    page_tokens: Optional[List[PlacedWord]] = None,
) -> str:
    """Render a full list of reading-order-sorted regions to one markdown string."""
    parts = []
    for i, region in enumerate(regions):
        parts.append(
            region_to_markdown(region, i, img, doc_path, table_runner, table_ocr_backend, page_tokens)
        )
    return "".join(parts)


def _process_rendered_page(
    img: Image.Image,
    layout_detector: DocLayoutV3,
    doc_path: str,
    page_index: int,
    page_tokens: Optional[List[PlacedWord]],
    page_ocr_backend: Optional[OCRBackend],
    table_runner: Optional[TableFormerONNX],
    table_ocr_backend: Optional[OCRBackend],
) -> dict:
    """
    Shared core of the diagram's bottom half, given an already-loaded page
    image: run layout detection, then route to either the pdfplumber-words
    matcher or the OCR-fallback matcher, then render per-region markdown.

    `page_tokens=None` (or `[]`) always forces the OCR path — this is what
    a standalone image (no native text layer at all) and a scanned PDF page
    (pdfplumber found no words) both end up doing identically.
    """
    logger.info("Page %d: running layout detection on %s", page_index, doc_path)
    layout_result = layout_detector.predict(_to_bgr_array(img))
    log_layout_result(layout_result)

    used_ocr = False
    if page_tokens:
        logger.info(
            "Page %d: matching path pdfplumber (%d tokens)",
            page_index,
            len(page_tokens),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for t in page_tokens[:5]:
                logger.debug("  sample token: %r @ (%.0f, %.0f)", t.text, t.x0, t.top)
        regions = align_tokens_to_layout(page_tokens, layout_result)
    else:
        used_ocr = True
        if page_ocr_backend is None:
            raise ValueError(
                "No native text/tokens available (scanned page or standalone "
                "image input) and no page_ocr_backend was provided to fall back on."
            )
        logger.warning(
            "Page %d: no text layer — OCR via %s",
            page_index,
            type(page_ocr_backend).__name__,
        )
        ocr_tokens = page_ocr_backend.get_text_boxes(img)
        regions = align_ocr_to_layout(layout_result, ocr_tokens)

    logger.info("Page %d: matched %d regions (used_ocr=%s)", page_index, len(regions), used_ocr)
    for r in regions:
        preview = (r.text[:80] + "...") if len(r.text) > 80 else r.text
        logger.info(
            "  region [%d] %s words=%d text=%r",
            r.reading_order,
            r.class_name,
            len(r.words),
            preview,
        )

    markdown_doc = regions_to_markdown(
        regions, img, doc_path, table_runner, table_ocr_backend, page_tokens=page_tokens or None
    )

    return {
        "page_index": page_index,
        "used_ocr": used_ocr,
        "layout_result": layout_result,
        "regions": regions,
        "markdown": markdown_doc,
        "image": img,
    }


def process_pdf_page(
    page,
    layout_detector: DocLayoutV3,
    doc_path: str,
    page_index: int = 0,
    resolution: int = 150,
    page_ocr_backend: Optional[OCRBackend] = None,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
) -> dict:
    """
    Run the full diagram for a single pdfplumber Page.

    - page_ocr_backend: used for the "Ocr(...)" box, only invoked if
      page.extract_words() comes back empty (scanned page / no text layer).
    - table_runner / table_ocr_backend: used for the "TableFormerONNX" box,
      only invoked for regions whose class_name == "table", and only as a
      fallback when no pdfplumber words land inside that region. If neither
      is usable, table regions fall back to an embedded cropped image instead.

    Returns a dict with the page's markdown plus the intermediate regions,
    so callers can inspect/debug a single page without re-running detection.
    """
    page_image = page.to_image(resolution=resolution)
    img = page_image.original.copy()

    # Computed once: reused both for matching regions (Plumber-path) and,
    # later, for pulling exact words inside any "table" region instead of
    # re-OCR-ing it. `[]` here means "no text layer" (e.g. a scanned page).
    img_w, img_h = img.size
    logger.info(
        "Processing PDF page %d of %s (resolution=%d, image=%dx%d)",
        page_index,
        doc_path,
        resolution,
        img_w,
        img_h,
    )
    page_tokens = pdfplumber_tokens_in_image_space(page, img_w, img_h)
    logger.info("Page %d: pdfplumber extracted %d word tokens", page_index, len(page_tokens))

    return _process_rendered_page(
        img, layout_detector, doc_path, page_index, page_tokens,
        page_ocr_backend, table_runner, table_ocr_backend,
    )


def process_pdf(
    doc_path: str,
    layout_detector: DocLayoutV3,
    pages: Optional[List[int]] = None,
    resolution: int = 150,
    page_ocr_backend: Optional[OCRBackend] = None,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
    max_workers: Optional[int] = None,
) -> str:
    """
    Run the full diagram over an entire PDF (or a subset of page indices)
    and return one concatenated markdown document.

    Pages are rendered sequentially (pdfplumber is not thread-safe), then
    layout/OCR/table extraction runs in parallel when ``max_workers > 1``.
    Pass ``max_workers=None`` (default) to pick a count from ``available_cpu_count()``.
    Pass ``max_workers=1`` to force sequential processing.
    """
    import pdfplumber

    logger.info("Processing PDF: %s (pages=%s)", doc_path, pages if pages is not None else "all")
    jobs: list[_RenderedPageJob] = []
    with pdfplumber.open(doc_path) as pdf:
        page_indices = pages if pages is not None else range(len(pdf.pages))
        for order, i in enumerate(page_indices):
            page = pdf.pages[i]
            page_image = page.to_image(resolution=resolution)
            img = page_image.original.copy()
            img_w, img_h = img.size
            logger.info(
                "Rendering PDF page %d of %s (resolution=%d, image=%dx%d)",
                i,
                doc_path,
                resolution,
                img_w,
                img_h,
            )
            page_tokens = pdfplumber_tokens_in_image_space(page, img_w, img_h)
            logger.info("Page %d: pdfplumber extracted %d word tokens", i, len(page_tokens))
            jobs.append(_RenderedPageJob(page_index=i, order=order, img=img, page_tokens=page_tokens))

    workers = compute_worker_count(len(jobs)) if max_workers is None else max(1, min(max_workers, len(jobs)))
    if workers <= 1:
        markdown_chunks = [
            _process_rendered_page(
                job.img,
                layout_detector,
                doc_path,
                job.page_index,
                job.page_tokens,
                page_ocr_backend,
                table_runner,
                table_ocr_backend,
            )["markdown"]
            for job in jobs
        ]
    else:
        logger.info(
            "Parallel PDF processing: %d workers (%d CPUs available, %d pages)",
            workers,
            available_cpu_count(),
            len(jobs),
        )
        markdown_chunks = _process_rendered_page_jobs_parallel(
            jobs,
            layout_detector,
            doc_path,
            page_ocr_backend,
            table_runner,
            table_ocr_backend,
            workers,
        )

    logger.info("Finished PDF: %s (%d pages)", doc_path, len(markdown_chunks))
    return "\n".join(markdown_chunks)


def process_image_page(
    image_path: Union[str, Path],
    layout_detector: DocLayoutV3,
    doc_path: Optional[str] = None,
    page_index: int = 0,
    page_ocr_backend: Optional[OCRBackend] = None,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
) -> dict:
    """
    Run the diagram on a single standalone image (a photo or scan, not a
    PDF page). There's no PDF text layer to check, so this always takes the
    OCR branch — equivalent to a PDF page where `extract_words()` came back
    empty. `page_ocr_backend` is therefore required, not optional, here.

    `doc_path` controls where cropped table/image regions get saved
    (`<doc_stem>/images/<n>.png`); defaults to `image_path` itself.
    """
    if page_ocr_backend is None:
        raise ValueError("process_image_page requires page_ocr_backend — standalone images have no text layer.")

    logger.info("Processing standalone image page %d: %s", page_index, image_path)
    img = Image.open(image_path).convert("RGB")
    doc_path = doc_path or str(image_path)

    return _process_rendered_page(
        img, layout_detector, doc_path, page_index, None,
        page_ocr_backend, table_runner, table_ocr_backend,
    )


def process_images(
    image_paths: Sequence[Union[str, Path]],
    layout_detector: DocLayoutV3,
    page_ocr_backend: Optional[OCRBackend] = None,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
    max_workers: Optional[int] = None,
) -> str:
    """
    Run the diagram over a batch of standalone images (e.g. photographed
    pages of one physical document) in the given order, and return one
    concatenated markdown document — the image-input analog of `process_pdf`.
    """
    if page_ocr_backend is None:
        raise ValueError("process_images requires page_ocr_backend — standalone images have no text layer.")

    logger.info("Processing %d standalone images", len(image_paths))
    jobs: list[_RenderedPageJob] = []
    for order, path in enumerate(image_paths):
        logger.info("Loading standalone image page %d: %s", order, path)
        img = Image.open(path).convert("RGB")
        jobs.append(_RenderedPageJob(page_index=order, order=order, img=img, page_tokens=None))

    workers = compute_worker_count(len(jobs)) if max_workers is None else max(1, min(max_workers, len(jobs)))
    if workers <= 1:
        chunks = [
            _process_rendered_page(
                job.img,
                layout_detector,
                str(image_paths[job.order]),
                job.page_index,
                job.page_tokens,
                page_ocr_backend,
                table_runner,
                table_ocr_backend,
            )["markdown"]
            for job in jobs
        ]
    else:
        logger.info(
            "Parallel image batch: %d workers (%d CPUs available, %d pages)",
            workers,
            available_cpu_count(),
            len(jobs),
        )
        chunks = _process_rendered_page_jobs_parallel(
            jobs,
            layout_detector,
            doc_path=str(image_paths[0]),
            page_ocr_backend=page_ocr_backend,
            table_runner=table_runner,
            table_ocr_backend=table_ocr_backend,
            max_workers=workers,
        )

    logger.info("Finished image batch (%d pages)", len(chunks))
    return "\n".join(chunks)


def process_document(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    layout_detector: DocLayoutV3,
    pages: Optional[List[int]] = None,
    resolution: int = 150,
    page_ocr_backend: Optional[OCRBackend] = None,
    table_runner: Optional[TableFormerONNX] = None,
    table_ocr_backend: Optional[OCRBackend] = None,
    max_workers: Optional[int] = None,
) -> str:
    """
    Single entry point for the whole pipeline — dispatches to the PDF path
    or the image path based on the input, so callers don't need to care
    which kind of file they have:

    - a path to a `.pdf` -> `process_pdf`
    - a path to a single image (.jpg/.png/etc.) -> `process_image_page`
    - a list of image paths -> `process_images` (treated as ordered pages
      of one document)
    """
    if isinstance(path, (list, tuple)):
        logger.info("process_document: dispatching to process_images (%d paths)", len(path))
        return process_images(
            path,
            layout_detector,
            page_ocr_backend=page_ocr_backend,
            table_runner=table_runner,
            table_ocr_backend=table_ocr_backend,
            max_workers=max_workers,
        )

    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        logger.info("process_document: dispatching to process_pdf (%s)", path)
        return process_pdf(
            str(path),
            layout_detector,
            pages=pages,
            resolution=resolution,
            page_ocr_backend=page_ocr_backend,
            table_runner=table_runner,
            table_ocr_backend=table_ocr_backend,
            max_workers=max_workers,
        )
    if suffix in IMAGE_EXTENSIONS:
        logger.info("process_document: dispatching to process_image_page (%s)", path)
        result = process_image_page(
            path, layout_detector, page_ocr_backend=page_ocr_backend,
            table_runner=table_runner, table_ocr_backend=table_ocr_backend,
        )
        return result["markdown"]

    raise ValueError(
        f"Unsupported file type {suffix!r} for {path!r}; expected a .pdf or "
        f"an image ({sorted(IMAGE_EXTENSIONS)})."
    )
