
from __future__ import annotations

import base64
import os
import re
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

        # ---- Main: upload + preview ----
        file_upload = gr.File(
            label="Upload a PDF or image",
            file_types=[
                ".pdf", ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff", ".webp", ".gif",
            ],
        )

        with gr.Row():
            with gr.Column(scale=2):
                pdf_preview = gr.HTML(visible=False, label="Document preview")
                image_preview = gr.Image(
                    visible=False, label="Document preview", interactive=False
                )
            with gr.Column(scale=1):
                file_info = gr.Markdown("")

        extract_btn = gr.Button("Extract markdown", variant="primary")

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

        # File upload → preview + metadata
        def handle_upload(file):
            if file is None:
                return (
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=None),
                    "",
                )

            file_path = Path(file)
            suffix = file_path.suffix.lower()
            file_size = file_path.stat().st_size / 1024
            info = f"**File:** `{file_path.name}`\n\n**Size:** {file_size:.1f} KB"

            if suffix == ".pdf":
                with open(file_path, "rb") as fh:
                    pdf_b64 = base64.b64encode(fh.read()).decode()
                html = (
                    f'<iframe src="data:application/pdf;base64,{pdf_b64}" '
                    'width="100%" height="640" style="border:none;"></iframe>'
                )
                return (
                    gr.update(visible=True, value=html),
                    gr.update(visible=False, value=None),
                    info,
                )
            elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                return (
                    gr.update(visible=False, value=""),
                    gr.update(visible=True, value=str(file_path)),
                    info,
                )
            else:
                return (
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=None),
                    info,
                )

        file_upload.change(
            handle_upload,
            inputs=file_upload,
            outputs=[pdf_preview, image_preview, file_info],
        )

        # Extract button → run pipeline
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
        ):
            if file is None:
                gr.Warning("Please upload a document first.")
                return "", "", None

            try:
                page_list = parse_pages(pages) if pages.strip() else None
            except ValueError as exc:
                gr.Warning(str(exc))
                return "", "", None

            # Copy uploaded file to a work directory
            src_path = Path(file)
            orig_name = src_path.name
            work_dir = PROJECT_ROOT / ".gradio_output" / Path(orig_name).stem
            work_dir.mkdir(parents=True, exist_ok=True)
            doc_path = work_dir / orig_name
            doc_path.write_bytes(src_path.read_bytes())

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
            if workers > 1:
                gr.Info(
                    f"Processing {num_pages} pages with {workers} parallel workers "
                    f"({available_cpu_count()} CPUs)."
                )

            try:
                markdown_doc = process_document(
                    str(doc_path),
                    layout_detector,
                    page_ocr_backend=page_ocr_backend,
                    table_runner=table_runner,
                    table_ocr_backend=table_ocr_backend,
                    **kwargs,
                )
            except Exception as exc:
                gr.Warning(f"Pipeline error: {exc}")
                return "", "", None

            preview_md = resolve_markdown_images(markdown_doc, work_dir)

            output_path = work_dir / (doc_path.stem + ".md")
            output_path.write_text(markdown_doc, encoding="utf-8")

            gr.Info("Extraction complete.")
            return preview_md, markdown_doc, str(output_path)

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
            ],
            outputs=[markdown_preview, markdown_source, download_btn],
        )

    return app


def main() -> None:
    print("Preparing models…")
    model_paths = resolve_model_paths(progress_callback=print)
    print("✅ All models ready")

    app = build_app(model_paths)
    app.launch()


if __name__ == "__main__":
    main()