# Image Analysis Server

A vibe coded Flask server that accepts images over HTTP and returns:
- **Tags** from [RAM++](https://github.com/xinyu1205/recognize-anything) and [CLIP](https://github.com/mlfoundations/open_clip)
- **Text** extracted by [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## Response format

```json
{
  "tags": ["sofa", "furniture", "indoor", "room"],
  "text": "clearance sale 2290"
}
```

---

## Setup

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y tesseract-ocr

# macOS
brew install tesseract
```

### 2. Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r src/requirements.txt
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

### 3. Download the RAM++ checkpoint

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='xinyu1205/recognize-anything-plus-model',
    filename='ram_plus_swin_large_14m.pth',
    local_dir='.',
)
"
```

Or set `RAM_CHECKPOINT` to an existing path:

```bash
export RAM_CHECKPOINT=/path/to/ram_plus_swin_large_14m.pth
```

---

## Running

```bash
python src/server.py
```

### Production (gunicorn)

```bash
gunicorn -w 1 -b 0.0.0.0:9100 --timeout 120 server:app
```

> **Note:** Do not use `--preload` — it causes the RAM++ model to fail loading
> in worker processes due to PyTorch fork safety issues. The model loads at
> import time per worker instead.

---

## API

### `GET /health`

```json
{
  "status": "ok",
  "ram_model_loaded": true,
  "clip_model_loaded": true,
  "ram_available": true,
  "clip_available": true,
  "device": "cpu",
  "max_concurrency": 2
}
```

### `POST /analyse`

Send an image as multipart/form-data or raw binary body:

```bash
# multipart (recommended)
curl -X POST http://localhost:9100/analyse -F "image=@photo.jpg"

# raw binary
curl -X POST http://localhost:9100/analyse \
  -H "Content-Type: image/jpeg" --data-binary @photo.jpg
```

Optional query parameters:

| Parameter        | Type    | Default | Description                          |
|------------------|---------|---------|--------------------------------------|
| `tag_confidence` | float   | `0`     | Minimum RAM++ tag confidence (0–1)   |

Returns **429** if the server is at `MAX_CONCURRENCY` active requests.

---

## Environment variables

| Variable            | Default                            | Description                              |
|---------------------|------------------------------------|------------------------------------------|
| `PORT`              | `9100`                             | Port the server listens on               |
| `RAM_CHECKPOINT`    | `ram_plus_swin_large_14m.pth`      | Path to the RAM++ checkpoint             |
| `MAX_CONCURRENCY`   | `2`                                | Max simultaneous inference requests      |
| `MAX_IMAGE_EDGE`    | `1600`                             | Longest edge images are downscaled to    |
| `CLIP_TAG_THRESHOLD`| `0.15`                             | Min CLIP probability to include a tag    |

---

## GPU support

CUDA is used automatically if available. No configuration needed.

---

## Docker

```bash
docker compose up --build
```
