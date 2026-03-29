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
gunicorn -w 1 -b 0.0.0.0:9100 --timeout 300 server:app
```

> Do **not** use `--preload` — models fail to load in forked workers due to PyTorch fork safety. Models load at import time per worker.

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

All logic lives in `src/server.py`. There are no tests, no linter config, and no other Python modules. Containerisation files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) remain at the repo root.

**Request flow for `POST /analyse`:**

1. Decode image from multipart form-data or raw binary body
2. Downscale if longest edge exceeds `MAX_IMAGE_EDGE` (default 1600px)
3. Run Florence-2, SigLIP, and OCR **concurrently** via a `ThreadPoolExecutor`
4. Return `{"tags": [...], "text": "..."}` JSON

**Tagging pipeline (Florence-2 + RAM++ + SigLIP → merge):**
- Florence-2 (`<OD>` task) returns deduplicated detected object labels; runs concurrently with RAM++ and SigLIP
- RAM++ runs on a 384×384 resize; tags above `RAM_TAG_THRESHOLD` (default 0.68) are appended without duplicating Florence-2 results
- SigLIP scores the image against a hardcoded 32-tag candidate list; tags above `SIGLIP_TAG_THRESHOLD` (default 0.1) are appended without duplicating prior results
- `tag_confidence` query param overrides both `RAM_TAG_THRESHOLD` and `SIGLIP_TAG_THRESHOLD` for that request (must be 0.0–1.0)

**OCR pipeline (Florence-2 `<OCR>` task):**
- Runs as the third sequential Florence-2 task in `get_florence_results`
- Returns extracted text directly; result is returned in the `text` response field

**Concurrency control:** A semaphore limits to `MAX_CONCURRENCY` (default 2) simultaneous inference requests; excess requests get HTTP 429.

**Device selection:** CUDA is used automatically if available, otherwise CPU. Both models are loaded at module import time.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `9100` | Listening port |
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | Florence-2 model ID or local path |
| `SIGLIP_MODEL` | `google/siglip-so400m-patch14-384` | SigLIP model ID or local path |
| `RAM_CHECKPOINT` | `ram_plus_swin_large_14m.pth` | Path to RAM++ weights file |
| `MAX_CONCURRENCY` | `2` | Max simultaneous requests |
| `MAX_IMAGE_EDGE` | `1600` | Downscale threshold (px) |
| `SIGLIP_TAG_THRESHOLD` | `0.1` | Min SigLIP probability for a tag |
| `RAM_TAG_THRESHOLD` | `0.68` | Min RAM++ probability for a tag |

## API

- `GET /health` — model load status, device, concurrency config
- `POST /analyse` — accepts multipart `image` field or raw binary body; optional `?tag_confidence=0.0–1.0`