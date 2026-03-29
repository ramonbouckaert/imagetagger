# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Flask HTTP server (`server.py`) that accepts images and returns AI-generated tags and extracted text. The two analysis pipelines (tagging and OCR) run concurrently per request.

## Running

```bash
python src/server.py
```

Production (gunicorn):
```bash
gunicorn -w 1 -b 0.0.0.0:9100 --timeout 120 server:app
```

> Do **not** use `--preload` — RAM++ fails to load in forked workers due to PyTorch fork safety. Models load at import time per worker.

Docker (CPU):
```bash
docker compose up --build
```

## Setup (first time)

```bash
# System dependency
sudo apt install -y tesseract-ocr   # Ubuntu/Debian

# Python deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r src/requirements.txt
pip install git+https://github.com/xinyu1205/recognize-anything.git

# Download RAM++ model checkpoint (~2GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='xinyu1205/recognize-anything-plus-model', filename='ram_plus_swin_large_14m.pth', local_dir='.')
"
```

## Architecture

All logic lives in `src/server.py`. There are no tests, no linter config, and no other Python modules. Containerisation files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) remain at the repo root.

**Request flow for `POST /analyse`:**

1. Decode image from multipart form-data or raw binary body
2. Downscale if longest edge exceeds `MAX_IMAGE_EDGE` (default 1600px)
3. Run tagging and OCR **concurrently** via a 2-worker `ThreadPoolExecutor`
4. Return `{"tags": [...], "text": "..."}` JSON

**Tagging pipeline (RAM++ → CLIP → merge):**
- RAM++ runs on a 384×384 resize; supports `tag_confidence` query param to filter by logit score
- CLIP (ViT-B-32) scores the image against a hardcoded 30-tag candidate list; tags above `CLIP_TAG_THRESHOLD` (default 0.15) are appended without duplicating RAM++ results

**OCR pipeline (Tesseract):**
- Preprocesses: greyscale → upscale if <1000px → invert dark backgrounds → adaptive threshold
- Probes both greyscale and thresholded variants; picks the one with higher Tesseract confidence
- Detects script via OSD → maps to language pack → re-runs with that language
- Tries PSM modes 3, 6, 11 and picks highest-confidence result
- Post-processes tokens: min 2 chars, ≥40% alphanumeric, then normalises to lowercase (preserving decimals, times, domains)

**Concurrency control:** A semaphore limits to `MAX_CONCURRENCY` (default 2) simultaneous inference requests; excess requests get HTTP 429.

**Device selection:** CUDA is used automatically if available, otherwise CPU. Both models are loaded at module import time.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `9100` | Listening port |
| `RAM_CHECKPOINT` | `ram_plus_swin_large_14m.pth` | Path to RAM++ weights |
| `MAX_CONCURRENCY` | `2` | Max simultaneous requests |
| `MAX_IMAGE_EDGE` | `1600` | Downscale threshold (px) |
| `CLIP_TAG_THRESHOLD` | `0.15` | Min CLIP probability for a tag |

## API

- `GET /health` — model load status, device, concurrency config
- `POST /analyse` — accepts multipart `image` field or raw binary body; optional `?tag_confidence=0.0–1.0`