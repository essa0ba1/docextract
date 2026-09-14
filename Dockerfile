# PDF Pipeline — Docker / Hugging Face Spaces (Gradio UI)
#
# Models (PP-DocLayout, TableFormer, OCR weights) are downloaded on first
# start via huggingface_hub and cached under /app/.cache/huggingface.

FROM python:3.10-slim

WORKDIR /app

# System deps:
# - tesseract-ocr*: optional Pytesseract backend
# - libgl1 / libglib2.0-0: OpenCV headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PORT=7860

RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers

EXPOSE 7860

CMD ["python", "main.py"]
