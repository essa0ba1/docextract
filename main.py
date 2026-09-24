from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from huggingface_hub import snapshot_download

from doc_pipeline import (
    DocLayoutV3,
    TableFormerONNX,
    available_cpu_count,
    compute_worker_count,
    get_ocr_backend,
    process_document,
    setup_pipeline_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

REC_KEYS_FILENAME = "ch_en_dict.txt"


# --------------------------------------------------------------------------- #
# Model path resolution — pulled out of module scope so it can be imported
# and unit-tested without triggering real downloads on `import main`.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelPaths:
    layout: str
    table_artifacts: str
    ocr_det_small: str
    ocr_rec_small: str
    ocr_det_medium: str
    ocr_rec_medium: str
    ocr_rec_keys: str


def _resolve_local_or_download(
    local_path: str,
    repo_id: str,
    local_dir: str,
    filename: str = "inference.onnx",
    downloader=snapshot_download,
    progress_callback=None,
) -> str:
    """
    Return `local_path` if it already exists on disk; otherwise download the
    repo via `downloader` (defaults to `huggingface_hub.snapshot_download`,
    swappable in tests) and return the resolved path to `filename` inside it.
    `progress_callback`, if given, is called once with a short human-readable
    message after this model is resolved (whether it was already local or
    freshly downloaded) — used to drive a live status indicator in the UI
    without making this function depend on Gradio.
    """
    if os.path.exists(local_path):
        if progress_callback:
            progress_callback(f"{repo_id}: already downloaded ✓")
        return local_path
    downloaded_dir = downloader(repo_id=repo_id, local_dir=local_dir)
    if progress_callback:
        progress_callback(f"{repo_id}: downloaded ✓")
    return os.path.join(downloaded_dir, filename)


def resolve_model_paths(downloader=snapshot_download, progress_callback=None) -> ModelPaths:
    """
    Resolve local paths to every model weight the pipeline needs, downloading
    only what's missing. Pure function of the local filesystem + `downloader`
    (injectable for tests — pass a fake/mock instead of hitting the network).
    Safe to call more than once — every branch just checks for an existing
    local file first.
    """
    layout = _resolve_local_or_download(
        "PP-DocLayout/inference.onnx",
        repo_id="PaddlePaddle/PP-DocLayoutV3_onnx",
        local_dir="PP-DocLayout",
        downloader=downloader,
        progress_callback=progress_callback,
    )

    if os.path.exists("tableformerv1"):
        table_artifacts = "tableformerv1"
        if progress_callback:
            progress_callback("bakhil-aissa/tableformerv1: already downloaded ✓")
    else:
        table_artifacts = downloader(
            repo_id="bakhil-aissa/tableformerv1", local_dir="tableformerv1"
        )
        if progress_callback:
            progress_callback("bakhil-aissa/tableformerv1: downloaded ✓")

    ocr_det_medium = _resolve_local_or_download(
        "pp_ocr_medium/det/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_medium_det_onnx",
        local_dir="pp_ocr_medium/det",
        downloader=downloader,
        progress_callback=progress_callback,
    )
    ocr_rec_medium = _resolve_local_or_download(
        "pp_ocr_medium/rec/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        local_dir="pp_ocr_medium/rec",
        downloader=downloader,
        progress_callback=progress_callback,
    )
    ocr_det_small = _resolve_local_or_download(
        "pp_ocr_small/det/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_small_det_onnx",
        local_dir="pp_ocr_small/det",
        downloader=downloader,
        progress_callback=progress_callback,
    )
    ocr_rec_small = _resolve_local_or_download(
        "pp_ocr_small/rec/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_small_rec_onnx",
        local_dir="pp_ocr_small/rec",
        downloader=downloader,
        progress_callback=progress_callback,
    )

    return ModelPaths(
        layout=layout,
        table_artifacts=table_artifacts,
        ocr_det_small=ocr_det_small,
        ocr_rec_small=ocr_rec_small,
        ocr_det_medium=ocr_det_medium,
        ocr_rec_medium=ocr_rec_medium,
        ocr_rec_keys=REC_KEYS_FILENAME,
    )


# --------------------------------------------------------------------------- #
# Pipeline loading (module-level cache; replaces st.cache_resource)
# --------------------------------------------------------------------------- #
_pipeline_cache: dict = {}


def load_pipeline_cached(
    layout_model: str,
    table_artifact_root: str,
    table_variant: str,
    ocr_backend_name: str,
    rec_path: str,
    det_path: str,
    rec_keys_path: str,
):
    key = (
        layout_model,
        table_artifact_root,
        table_variant,
        ocr_backend_name,
        rec_path,
        det_path,
        rec_keys_path,
    )
    if key not in _pipeline_cache:
        setup_pipeline_logging(level="INFO")
        layout_detector = DocLayoutV3(layout_model)

        table_runner = TableFormerONNX(
            artifact_root=table_artifact_root,
            variant=table_variant,
        )
        if ocr_backend_name == "rapidocr":
            ocr_backend = get_ocr_backend(
                ocr_backend_name,
                det_model_path=det_path,
                rec_model_path=rec_path,
                rec_keys_path=rec_keys_path,
            )
        else:
            ocr_backend = get_ocr_backend(ocr_backend_name)
        _pipeline_cache[key] = (
            layout_detector,
            ocr_backend,
            table_runner,
            ocr_backend,
        )
    return _pipeline_cache[key]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_pages(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    pages: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        page = int(part)
        if page < 1:
            raise ValueError("Page numbers must be 1-based (1, 2, 3, …).")
        pages.append(page)
    return pages or None


def resolve_markdown_images(markdown: str, base_dir: Path) -> str:
    """Turn relative image links into absolute paths so Gradio can render them."""

    def _replace(match: re.Match[str]) -> str:
        alt, path = match.group(1), match.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return match.group(0)
        candidate = Path(path)
        if not candidate.is_file():
            candidate = (base_dir / path).resolve()
        if candidate.is_file():
            return f"![{alt}]({candidate.as_posix()})"
        return match.group(0)

    return IMAGE_LINK_RE.sub(_replace, markdown)


def estimate_processing_seconds(
    num_pages: int, workers: int, seconds_per_page: float = 4.0
) -> float:
    """
    Rough wall-clock estimate used only to animate the progress bar smoothly
    while `process_document` runs in the background. It is not a precise
    measurement — real completion is still detected by the worker thread
    finishing, so a bad estimate only makes the bar move faster/slower than
    reality, never wrong about whether the job is actually done.
    """
    workers = max(workers, 1)
    return max(seconds_per_page, (num_pages / workers) * seconds_per_page + 1.5)


def render_progress_bar(fraction: float, desc: str) -> str:
    """A plain HTML progress bar. Rendered directly into a gr.HTML component
    so it's visible regardless of Gradio version/queue quirks — no reliance
    on gr.Progress()'s internal streaming."""
    pct = max(0, min(100, int(fraction * 100)))
    return f"""
<div style="margin:10px 0;font-family:inherit;">
  <div style="font-size:0.9em;margin-bottom:6px;">{desc}</div>
  <div style="background:#e5e7eb;border-radius:8px;overflow:hidden;height:18px;width:100%;">
    <div style="background:#6366f1;height:100%;width:{pct}%;
                transition:width 0.3s ease;"></div>
  </div>
  <div style="font-size:0.8em;color:#6b7280;margin-top:4px;">{pct}%</div>
</div>
"""


# --------------------------------------------------------------------------- #
# Gradio app
# --------------------------------------------------------------------------- #
def build_app(model_paths: ModelPaths) -> gr.Blocks:
    mp = model_paths

    with gr.Blocks(title="PDF Pipeline", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 PDF Pipeline")
        gr.Markdown("Extract structured markdown from PDFs and scanned images.")

        # ---- Sidebar: settings ----
        with gr.Sidebar():
            gr.Markdown("## Settings")
            layout_model_dd = gr.Dropdown(
                choices=[mp.layout],
                value=mp.layout,
                label="Layout model",
            )
            table_artifact_dd = gr.Dropdown(
                choices=[mp.table_artifacts],
                value=mp.table_artifacts,
                label="TableFormer artifacts",
            )
            table_variant_dd = gr.Dropdown(
                choices=["accurate"],
                value="accurate",
                label="TableFormer variant",
            )
            ocr_backend_dd = gr.Dropdown(
                choices=["rapidocr", "pytesseract"],
                value="rapidocr",
                label="OCR backend",
            )
            with gr.Group(visible=True) as rapidocr_group:
                det_model_dd = gr.Dropdown(
                    choices=[mp.ocr_det_small, mp.ocr_det_medium],
                    value=mp.ocr_det_small,
                    label="RapidOCR detector model",
                )
                rec_model_dd = gr.Dropdown(
                    choices=[mp.ocr_rec_small, mp.ocr_rec_medium],
                    value=mp.ocr_rec_small,
                    label="RapidOCR recognizer model",
                )
                keys_model_dd = gr.Dropdown(
                    choices=[mp.ocr_rec_keys],
                    value=mp.ocr_rec_keys,
                    label="RapidOCR keys model",
                )
            resolution_slider = gr.Slider(
                minimum=72,
                maximum=300,
                value=150,
                step=1,
                label="PDF render DPI",
            )
            pages_textbox = gr.Textbox(
                placeholder="1, 2, 5 — leave empty for all pages",
                label="PDF pages (optional)",
            )
            gr.Markdown(
                f"**Parallel pages:** auto "
                f"({available_cpu_count()} CPUs detected, up to "
                f"{compute_worker_count(99)} workers for multi-page docs)"
            )

        # ---- Main: upload ----
        file_upload = gr.File(
            label="Upload a PDF or image",
            file_types=[
                ".pdf", ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff", ".webp", ".gif",
            ],
        )
        upload_status_html = gr.HTML("")

        with gr.Row():
            extract_btn = gr.Button("Extract markdown", variant="primary", scale=3)
            render_preview_cb = gr.Checkbox(
                value=True,
                label="Render markdown preview",
                info="Uncheck to skip pretty-rendering — faster for large/table-heavy docs",
                scale=2,
            )

        progress_html = gr.HTML("")

        # ---- Results ----
        with gr.Tabs():
            with gr.Tab("Preview"):
                markdown_preview = gr.Markdown("")
            with gr.Tab("Markdown source"):
                markdown_source = gr.Code(language="markdown", show_label=False)
            with gr.Tab("Download"):
                download_btn = gr.DownloadButton(
                    label="Download .md file",
                    variant="primary",
                )

        # ---- Event wiring ----

        # Toggle RapidOCR-specific controls
        def toggle_rapidocr(backend: str):
            return gr.update(visible=(backend == "rapidocr"))

        ocr_backend_dd.change(
            toggle_rapidocr, inputs=ocr_backend_dd, outputs=rapidocr_group
        )

        # File upload → just confirm receipt with a full progress bar. No
        # document preview is rendered (that meant base64-encoding the whole
        # PDF into an iframe on every upload, which is what was slow).
        def handle_upload(file):
            if file is None:
                return ""

            file_path = Path(file)
            size_kb = file_path.stat().st_size / 1024
            size_label = f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
            return render_progress_bar(1.0, f"✓ Uploaded: {file_path.name} ({size_label})")

        file_upload.change(
            handle_upload,
            inputs=file_upload,
            outputs=upload_status_html,
        )

        # Extract button → run pipeline, with a visible progress bar.
        #
        # `process_document` is a single blocking call and we don't control
        # its internals, so we can't get true per-page progress out of it.
        # Instead: run it in a background thread, and while it's alive,
        # repeatedly `yield` an updated HTML progress bar against a rough
        # time estimate. This is a generator function — every `yield` pushes
        # a real UI update immediately, so the bar is guaranteed to actually
        # move on screen (unlike gr.Progress(), whose streaming depends on
        # queue/version details we can't verify here).
        def handle_extract(
            file,
            layout,
            table_root,
            table_var,
            ocr,
            det,
            rec,
            keys,
            dpi,
            pages,
            render_preview,
        ):
            if file is None:
                gr.Warning("Please upload a document first.")
                yield "", "", None, ""
                return

            try:
                page_list = parse_pages(pages) if pages.strip() else None
            except ValueError as exc:
                gr.Warning(str(exc))
                yield "", "", None, ""
                return

            yield "", "", None, render_progress_bar(0.0, "Preparing document…")

            # Copy uploaded file to a work directory
            src_path = Path(file)
            orig_name = src_path.name
            work_dir = PROJECT_ROOT / ".gradio_output" / Path(orig_name).stem
            work_dir.mkdir(parents=True, exist_ok=True)
            doc_path = work_dir / orig_name
            doc_path.write_bytes(src_path.read_bytes())

            yield "", "", None, render_progress_bar(
                0.08, "Loading pipeline (first run may download models)…"
            )

            # Load (cached) pipeline
            layout_detector, page_ocr_backend, table_runner, table_ocr_backend = (
                load_pipeline_cached(
                    layout,
                    table_root,
                    table_var,
                    ocr,
                    rec_path=rec,
                    det_path=det,
                    rec_keys_path=keys,
                )
            )

            kwargs: dict = {"resolution": dpi}
            if page_list is not None and doc_path.suffix.lower() == ".pdf":
                kwargs["pages"] = page_list

            num_pages = 1
            if doc_path.suffix.lower() == ".pdf":
                import pdfplumber

                with pdfplumber.open(doc_path) as pdf:
                    num_pages = len(page_list) if page_list is not None else len(pdf.pages)
            workers = compute_worker_count(num_pages)

            yield "", "", None, render_progress_bar(
                0.15, f"Processing {num_pages} page(s) with {workers} worker(s)…"
            )
            if workers > 1:
                gr.Info(
                    f"Processing {num_pages} pages with {workers} parallel workers "
                    f"({available_cpu_count()} CPUs)."
                )

            result_holder: dict = {}
            error_holder: dict = {}

            def _worker() -> None:
                try:
                    result_holder["markdown"] = process_document(
                        str(doc_path),
                        layout_detector,
                        page_ocr_backend=page_ocr_backend,
                        table_runner=table_runner,
                        table_ocr_backend=table_ocr_backend,
                        **kwargs,
                    )
                except Exception as exc:  # noqa: BLE001
                    error_holder["exc"] = exc

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            estimated_total = estimate_processing_seconds(num_pages, workers)
            start = time.time()
            while thread.is_alive():
                elapsed = time.time() - start
                frac = 0.15 + 0.75 * min(elapsed / estimated_total, 1.0)
                yield "", "", None, render_progress_bar(
                    frac, f"Processing pages… (~{int(frac * 100)}%)"
                )
                time.sleep(0.3)

            thread.join()

            if "exc" in error_holder:
                gr.Warning(f"Pipeline error: {error_holder['exc']}")
                yield "", "", None, ""
                return

            markdown_doc = result_holder.get("markdown", "")

            yield "", "", None, render_progress_bar(0.92, "Writing output file…")
            output_path = work_dir / (doc_path.stem + ".md")
            output_path.write_text(markdown_doc, encoding="utf-8")

            if render_preview:
                yield "", "", None, render_progress_bar(0.97, "Rendering preview…")
                preview_md = resolve_markdown_images(markdown_doc, work_dir)
            else:
                preview_md = (
                    "*Preview rendering skipped — check "
                    '"Render markdown preview" to enable it, or see the '
                    '"Markdown source" tab for the raw output.*'
                )

            gr.Info("Extraction complete.")
            yield preview_md, markdown_doc, str(output_path), render_progress_bar(
                1.0, "Done ✓"
            )

        extract_btn.click(
            handle_extract,
            inputs=[
                file_upload,
                layout_model_dd,
                table_artifact_dd,
                table_variant_dd,
                ocr_backend_dd,
                det_model_dd,
                rec_model_dd,
                keys_model_dd,
                resolution_slider,
                pages_textbox,
                render_preview_cb,
            ],
            outputs=[markdown_preview, markdown_source, download_btn, progress_html],
        )

    return app


def main() -> None:
    print("Preparing models…")
    model_paths = resolve_model_paths(progress_callback=print)
    print("✅ All models ready")

    app = build_app(model_paths)
    app.queue()  # required for the generator-based progress updates to stream live
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
    )


if __name__ == "__main__":
    main()