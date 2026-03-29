# Image Analysis Server

A single-file Flask server that accepts images over HTTP and returns AI-generated tags, a description, and extracted text.

## Response format

```json
{
  "tags": ["person", "phone", "selfie", "face"],
  "description": "A man taking a selfie in front of a bathroom mirror.",
  "text": "clearance sale 2290"
}
```

- **`tags`** — object labels from [Florence-2](https://huggingface.co/microsoft/Florence-2-large) (`<OD>`), [RAM++](https://github.com/xinyu1205/recognize-anything), and [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384), merged and deduplicated
- **`description`** — rich caption from Florence-2 (`<MORE_DETAILED_CAPTION>`)
- **`text`** — OCR output from Florence-2 (`<OCR>`), ASCII only

---

## Docker (recommended)

```bash
docker compose up --build
```

Models are downloaded at build time. The container needs no network access at runtime.

> Do not use `--preload` with gunicorn — models fail to load in forked workers due to PyTorch fork safety.

---

## Manual setup

### 1. Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r src/requirements.txt
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

### 2. Download the RAM++ checkpoint (~2 GB)

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

Florence-2-large (~1.5 GB) and SigLIP (~3.4 GB) download automatically on first run and are cached in `~/.cache/huggingface/`.

### 3. Run

```bash
python src/server.py
```

---

## API

### `GET /health`

```json
{
  "status": "ok",
  "florence_model_loaded": true,
  "siglip_model_loaded": true,
  "ram_model_loaded": true,
  "device": "cpu",
  "max_concurrency": 2
}
```

### `POST /analyse`

Send an image as multipart/form-data or raw binary body:

```bash
# multipart
curl -X POST http://localhost:9100/analyse -F "image=@photo.jpg"

# raw binary
curl -X POST http://localhost:9100/analyse \
  -H "Content-Type: image/jpeg" --data-binary @photo.jpg
```

Supported formats: JPEG, PNG, WebP, GIF, AVIF, HEIC, and anything else Pillow handles.

**Optional query parameters:**

| Parameter        | Type  | Default | Description                                           |
|------------------|-------|---------|-------------------------------------------------------|
| `tag_confidence` | float | —       | Override threshold for both RAM++ and SigLIP (0.0–1.0) |

**Error responses:**

| Status | Reason                                      |
|--------|---------------------------------------------|
| `400`  | Could not decode the uploaded image         |
| `429`  | Server at `MAX_CONCURRENCY` active requests |
| `503`  | One or more models not yet loaded           |

---

## Environment variables

| Variable              | Default                             | Description                              |
|-----------------------|-------------------------------------|------------------------------------------|
| `PORT`                | `9100`                              | Listening port                           |
| `FLORENCE_MODEL`      | `microsoft/Florence-2-large`        | Florence-2 model ID or local path        |
| `SIGLIP_MODEL`        | `google/siglip-so400m-patch14-384`  | SigLIP model ID or local path            |
| `RAM_CHECKPOINT`      | `ram_plus_swin_large_14m.pth`       | Path to RAM++ weights file               |
| `MAX_CONCURRENCY`     | `2`                                 | Max simultaneous inference requests      |
| `MAX_IMAGE_EDGE`      | `1600`                              | Longest edge images are downscaled to    |
| `SIGLIP_TAG_THRESHOLD`| `0.1`                               | Min SigLIP probability to include a tag  |
| `RAM_TAG_THRESHOLD`   | `0.68`                              | Min RAM++ probability to include a tag   |

---

## GPU support

CUDA is used automatically if available. No configuration needed.