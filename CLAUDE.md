# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask HTTP server that accepts images and returns AI-generated tags, a description, and extracted text. Analysis pipelines run concurrently per request.

## Running

```bash
python src/server.py
```

Production (gunicorn):
```bash
gunicorn -w 1 --worker-class gthread --threads 16 -b 0.0.0.0:9100 --timeout 300 server:app
```

> Do **not** use `--preload` — models fail to load in forked workers due to PyTorch fork safety. Models load at import time per worker.
> The `gthread` worker is required — the default `sync` worker processes one request at a time and won't allow requests to queue.

Docker (CPU):
```bash
docker compose up --build
```

## Setup (first time)

```bash
# Python deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r src/requirements.txt
pip install git+https://github.com/xinyu1205/recognize-anything.git

# Download RAM++ checkpoint (~2 GB) — place in the working directory
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='xinyu1205/recognize-anything-plus-model', filename='ram_plus_swin_large_14m.pth', local_dir='.')
"
```

Florence-2-large (~1.5 GB) and SigLIP (~3.4 GB) download automatically from Hugging Face on first run and are cached in `~/.cache/huggingface/`. RAM++ requires the checkpoint file to be present at the path given by `RAM_CHECKPOINT` (default: `ram_plus_swin_large_14m.pth` in the working directory).

## Architecture

Logic is split across four modules in `src/`. Containerisation files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) remain at the repo root.

| Module | Purpose |
|---|---|
| `config.py` | Env vars, constants, device detection, import guards |
| `models.py` | Model singleton classes (Florence2Model, SigLIPModel, RAMModel, OCRCorrectionModel, SpacyModel) |
| `controller.py` | AnalysisController — image decode, orchestration, tag merging |
| `server.py` | Flask app and routes only |

**Request flow for `POST /analyse`:**

1. Decode image from multipart form-data or raw binary body
2. Downscale if longest edge exceeds `MAX_IMAGE_EDGE` (default 1600px)
3. Run Florence-2, SigLIP, and RAM++ concurrently; OCR correction and spaCy follow
4. Return `{"tags": [...], "description": "...", "text": "..."}` JSON

**Tagging pipeline (Florence-2 + RAM++ + SigLIP + spaCy → merge):**
- Florence-2 (`<OD>` task) returns deduplicated detected object labels
- RAM++ tags above `RAM_TAG_THRESHOLD` (default 0.68) are appended
- SigLIP scores the image against a 20-tag candidate list; tags above `SIGLIP_TAG_THRESHOLD` (default 0.05) are appended
- spaCy extracts noun chunks and individual nouns from the Florence-2 caption and quoted text within it, and words from the OCR output

**OCR pipeline (Florence-2 `<OCR>` task):**
- Runs as one of three sequential Florence-2 tasks
- Output is spell-corrected by the ByT5 OCR correction model
- Result is returned in the `text` response field

**Concurrency model:** Each model is a singleton class. Non-thread-safe models (Florence-2, OCR correction) own a private single-thread executor that queues jobs internally. Thread-safe models (SigLIP, RAM++, spaCy) are called directly from a shared thread pool. Requests are never turned away — they queue at each model's executor. gunicorn's `gthread` worker allows multiple requests to be in-flight simultaneously within the single process.

**Device selection:** CUDA is used automatically if available, otherwise CPU. All models load at import time.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `9100` | Listening port |
| `FLORENCE_MODEL` | `florence-community/Florence-2-large` | Florence-2 model ID or local path |
| `SIGLIP_MODEL` | `google/siglip2-so400m-patch14-384` | SigLIP model ID or local path |
| `RAM_CHECKPOINT` | `ram_plus_swin_large_14m.pth` | Path to RAM++ weights file |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model name |
| `OCR_CORRECTION_MODEL` | `yelpfeast/byt5-base-english-ocr-correction` | ByT5 OCR correction model ID |
| `MAX_IMAGE_EDGE` | `1600` | Downscale threshold (px) |

## API

- `GET /health` — model load status and device
- `POST /analyse` — accepts multipart `image` field or raw binary body