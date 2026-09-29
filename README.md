# PDF Pipeline

A comprehensive PDF-to-Markdown extraction pipeline using state-of-the-art layout detection, OCR, and table extraction models.

## Features

- **Layout Detection**: Uses PP-DocLayoutV3 (ONNX) for accurate document layout analysis
- **OCR Support**: Multiple OCR backends - PaddleOCR, RapidOCR, Pytesseract
- **Table Extraction**: TableFormerONNX with OTSL (Object Table Structure Language) output
- **Dual Input Support**: Process both PDFs and standalone images (PNG, JPG, etc.)
- **Streamlit UI**: Interactive web interface for easy document processing
- **Chat with your document**: Upload a PDF, text file, or image and ask questions about it (BM25 retrieval + any OpenAI-compatible LLM)

## Architecture

```
PDF/Image Input
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layout Detection                          │
│              PP-DocLayoutV3 (ONNX)                          │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Extraction Strategy                        │
├─────────────────────────────────────────────────────────────┤
│  Native PDF Text │  OCR Fallback (Paddle/Rapid/Tesseract)   │
│  (pdfplumber)    │                                          │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│               Region Classification                          │
│  • Text Blocks  • Tables  • Figures  • Headers  • Footers     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Table Extraction (if applicable)                │
│         TableFormerONNX → OTSL → Markdown                   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Markdown Output                                 │
│  • Structured text  • Extracted tables  • Figure references   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
pdf_pipeline_project/
├── main.py                      # Streamlit web interface with HuggingFace Hub integration
├── example_usage.py             # Python API examples
├── html_to_table.py             # HTML table utilities
├── requirements.txt             # Python dependencies
├── PP-DocLayout/                # Layout detection models (282MB) - auto-downloaded
├── tableformerv1/               # Table extraction models (205MB) - auto-downloaded
├── pp_ocr_small/                # Small OCR models (~35MB) - auto-downloaded
├── pp_ocr_medium/               # Medium OCR models (~100MB) - auto-downloaded
├── pdf_pipeline/                # Core Python package
│   ├── __init__.py             # Public API exports
│   ├── layout.py               # DocLayoutV3 ONNX wrapper
│   ├── ocr_backends.py         # OCR implementations (RapidOCR, Pytesseract)
│   ├── pipeline.py             # Main processing pipeline
│   ├── table_extraction.py     # TableFormer integration
│   ├── logging_config.py       # Logging utilities
│   └── ch_en_dict.txt          # OCR character dictionary
├── doc_chat/                    # Retrieval-augmented chat over extracted documents
│   ├── loader.py               # PDF/image -> pipeline, text files read directly
│   ├── chunking.py             # Markdown -> chunks (page/heading aware)
│   ├── retrieval.py            # BM25 index over chunks
│   ├── rag.py                  # Prompt/message construction
│   └── llm.py                  # OpenAI-compatible chat client (streaming)
├── examples/                    # Sample PDFs and outputs
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.10+
- Tesseract OCR (for Pytesseract backend)
- HuggingFace account token (for model downloading)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd pdf_pipeline_project
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Tesseract** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
```

For macOS:
```bash
brew install tesseract
```

### Model Download (Automatic)

Models are automatically downloaded from HuggingFace Hub on first run:

| Model | Repository | Size |
|-------|------------|------|
| **PP-DocLayoutV3** | `PaddlePaddle/PP-DocLayoutV3_onnx` | ~282MB |
| **TableFormer** | `bakhil-aissa/tableformerv1` | ~205MB |
| **PaddleOCR Medium** | `PaddlePaddle/PP-OCRv6_medium_*_onnx` | ~100MB |
| **PaddleOCR Small** | `PaddlePaddle/PP-OCRv6_small_*_onnx` | ~35MB |

To pre-download models:
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="PaddlePaddle/PP-DocLayoutV3_onnx", local_dir="PP-DocLayout")
snapshot_download(repo_id="bakhil-aissa/tableformerv1", local_dir="tableformerv1")
```

## Usage

### Streamlit Web Interface

Run the interactive web UI:

```bash
streamlit run main.py
```

Then open your browser to `http://localhost:8501`

Features:
- Upload PDF or image files
- Configure OCR backend (PaddleOCR, RapidOCR, Pytesseract)
- Adjust rendering resolution
- Preview extracted markdown
- Download results

### Chat with a document

The **Chat** tab lets you upload a `.pdf`, `.txt`/`.md`, or image, index it, and
ask questions about its contents. PDFs and images go through the extraction
pipeline first; text files are read directly. The answer is generated by an
OpenAI-compatible chat endpoint, configured in the tab's *LLM settings*:

| Setting | Default | Notes |
|---------|---------|-------|
| API base URL | `$LLM_BASE_URL` or `https://api.openai.com/v1` | Works with OpenAI, Groq, Together, OpenRouter, vLLM, or Ollama (`http://localhost:11434/v1`) |
| Model | `$LLM_MODEL` or `gpt-4o-mini` | Any model the endpoint serves |
| API key | `$LLM_API_KEY` / `$OPENAI_API_KEY` | Can also be pasted in the UI |

```bash
export OPENAI_API_KEY=sk-...
python main.py
```

Retrieval is BM25 over markdown chunks (no embedding model, no vector store),
and answers cite the page/heading of the excerpts they used.

```python
from doc_chat import DocumentIndex, build_chat_messages, chat_completion

index = DocumentIndex.build("report.pdf", markdown_doc)
chunks = index.retrieve("What was the 2023 revenue?", top_k=4)
answer = chat_completion(
    build_chat_messages("What was the 2023 revenue?", chunks, document_name=index.name),
    model="gpt-4o-mini",
)
```

### Python API

#### Process a PDF:

```python
from pdf_pipeline import (
    DocLayoutV3,
    TableFormerONNX,
    get_ocr_backend,
    process_document,
)
from huggingface_hub import snapshot_download
import os

# Model paths (auto-download from HuggingFace)
PP_DOCLAYOUT_PATH = ("PP-DocLayout/inference.onnx" if os.path.exists("PP-DocLayout/inference.onnx") 
    else os.path.join(snapshot_download(repo_id="PaddlePaddle/PP-DocLayoutV3_onnx", local_dir="PP-DocLayout"), "inference.onnx")
)

DET_PATH_MEDIUM = (
    "pp_ocr_medium/det/inference.onnx" if os.path.exists("pp_ocr_medium/det/inference.onnx") 
    else os.path.join(snapshot_download(repo_id="PaddlePaddle/PP-OCRv6_medium_det_onnx", local_dir="pp_ocr_medium", subfolder="det"), "inference.onnx")
)

REC_PATH_MEDIUM = (
    "pp_ocr_medium/rec/inference.onnx" if os.path.exists("pp_ocr_medium/rec/inference.onnx") 
    else snapshot_download(repo_id="PaddlePaddle/PP-OCRv6_medium_rec_onnx", local_dir="pp_ocr_medium", subfolder="rec")
)

REC_KEYS_PATH = "ch_en_dict.txt"

# Initialize components
layout_detector = DocLayoutV3(PP_DOCLAYOUT_PATH)
page_ocr_backend = get_ocr_backend("rapidocr", det_model_path=DET_PATH_MEDIUM, rec_model_path=REC_PATH_MEDIUM, rec_keys_path=REC_KEYS_PATH)
table_runner = TableFormerONNX(artifact_root="tableformerv1", variant="accurate")
table_ocr_backend = get_ocr_backend("rapidocr", det_model_path=DET_PATH_MEDIUM, rec_model_path=REC_PATH_MEDIUM, rec_keys_path=REC_KEYS_PATH)

# Process document
markdown_doc = process_document(
    "document.pdf",
    layout_detector,
    page_ocr_backend=page_ocr_backend,
    table_runner=table_runner,
    table_ocr_backend=table_ocr_backend,
)

print(markdown_doc)
```

#### Process an Image:

```python
# Same setup as above...

# Process standalone image (always uses OCR)
markdown_doc = process_document(
    "scanned_page.png",
    layout_detector,
    page_ocr_backend=page_ocr_backend,
    table_runner=table_runner,
    table_ocr_backend=table_ocr_backend,
)
```

## OCR Backends

Choose the best OCR backend for your needs:

| Backend | Speed | Accuracy | Languages | Notes |
|---------|-------|----------|-----------|-------|
| **RapidOCR** | Very Fast | Good | 10+ | ONNX-based, lightweight |
| **Pytesseract** | Medium | Good | 100+ | Tesseract wrapper, configurable |

### Model Sizes

RapidOCR supports two model sizes:

| Model | Detection | Recognition | Size | Speed | Accuracy |
|-------|-----------|-------------|------|-------|----------|
| **Small** | PP-OCRv6_small | PP-OCRv6_small | ~35MB | Fast | Good |
| **Medium** | PP-OCRv6_medium | PP-OCRv6_medium | ~100MB | Medium | Better |

Switch backends and models:
```python
# RapidOCR with small model (faster)
ocr = get_ocr_backend("rapidocr", 
    det_model_path=DET_PATH_SMALL,
    rec_model_path=REC_PATH_SMALL,
    rec_keys_path=REC_KEYS_PATH)

# RapidOCR with medium model (more accurate)
ocr = get_ocr_backend("rapidocr",
    det_model_path=DET_PATH_MEDIUM,
    rec_model_path=REC_PATH_MEDIUM,
    rec_keys_path=REC_KEYS_PATH)

# Pytesseract (French + English)
ocr = get_ocr_backend("pytesseract", lang="fra+eng")
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pdf_pipeline'`

**Solution**: Run from project root, or install as editable package:
```bash
pip install -e .
```

### Issue: `TesseractNotFoundError: tesseract is not installed`

**Solution**: Install Tesseract system binary:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: `onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument`

**Solution**: Check model files are not corrupted. Re-download if needed:
```bash
# Check file sizes match expected
ls -lh PP-DocLayout/*.onnx
ls -lh tableformerv1/onnx/accurate/*.onnx
```

### Issue: Out of Memory (OOM) on large PDFs

**Solution**: Reduce rendering resolution:
```python
# Lower DPI for memory-constrained environments
process_document(
    "large.pdf",
    layout_detector,
    resolution=100,  # Default is 150, try 100 or 72
)
```

### Issue: Slow OCR on CPU

**Solution**: Use RapidOCR for faster CPU inference:
```python
ocr = get_ocr_backend("rapidocr")  # ~2-3x faster than PaddleOCR on CPU
```

## Performance Benchmarks

Typical processing times (single page, Intel i7, 16GB RAM):

| Stage | Time | Notes |
|-------|------|-------|
| PDF Rendering | 200-500ms | Depends on resolution |
| Layout Detection | 300-800ms | ONNXRuntime, CPU |
| Text Extraction | 100-300ms | Native PDF text |
| OCR (fallback) | 1-3s | Only if no native text |
| Table Extraction | 2-5s | Per table region |

**Total**: ~1-5 seconds per page (depending on content density)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black pdf_pipeline/ main.py

# Type checking
mypy pdf_pipeline/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PaddleOCR** - For OCR model implementations
- **PP-DocLayout** - For document layout detection
- **TableFormer** - For table structure recognition
- **Docling** - For OTSL to Markdown conversion

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for document processing automation**