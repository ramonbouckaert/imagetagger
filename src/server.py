"""
Image Analysis Server
- Uses Florence-2 for image tagging (OD), description, and OCR
- Uses RAM++ and SigLIP as additional tagging passes, results merged
- Request queue with configurable max concurrency to prevent OOM
- Exposes POST /analyse and GET /health
"""

import io
import os
import re
import sys
import logging
import traceback
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, wait

from PIL import Image
try:
    from pillow_heif import register_heif_opener, open_heif as _heif_open
    register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False
    _heif_open = None
    logging.warning("pillow-heif not installed — AVIF/HEIC images will not be supported")
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# ── Core imports ───────────────────────────────────────────────────────────────
_import_errors: list[str] = []

try:
    import torch
except ImportError as e:
    _import_errors.append(f"  • torch — {e}\n    Fix: {sys.executable} -m pip install torch torchvision")

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers import SiglipModel, SiglipProcessor
except ImportError as e:
    _import_errors.append(
        f"  • transformers — {e}\n"
        f"    Fix: {sys.executable} -m pip install 'transformers>=4.40'"
    )

try:
    from ram.models import ram_plus
    from ram import get_transform as ram_get_transform
    from ram import inference_ram
    _RAM_AVAILABLE = True
except Exception as _ram_import_err:
    _RAM_AVAILABLE = False
    logging.warning(
        "recognize-anything unavailable — RAM++ tags will be disabled.\n"
        "Import error: %s",
        _ram_import_err,
        exc_info=True,
    )

if _import_errors:
    logging.warning(
        "Some models will be unavailable. Failed imports:\n%s\n"
        "NOTE: Always use '%s -m pip install ...' (not bare 'pip install') "
        "to ensure packages install into the same Python that is running this server.",
        "\n".join(_import_errors),
        sys.executable,
    )

MODELS_AVAILABLE = len(_import_errors) == 0

def _open_image(data: bytes) -> Image.Image:
    """
    Open image bytes via PIL. Falls back to open_heif() directly for AVIF/HEIC
    files whose ftyp brand PIL's auto-detection rejects but libheif can decode.
    """
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        if _HEIF_AVAILABLE:
            heif = _heif_open(io.BytesIO(data))
            return heif[0].to_pillow()
        raise


# ── Config ─────────────────────────────────────────────────────────────────────
FLORENCE_MODEL       = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")
SIGLIP_MODEL_ID      = os.environ.get("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
RAM_CHECKPOINT       = os.environ.get("RAM_CHECKPOINT", "ram_plus_swin_large_14m.pth")
MAX_CONCURRENCY      = int(os.environ.get("MAX_CONCURRENCY", "2"))
MAX_IMAGE_EDGE       = int(os.environ.get("MAX_IMAGE_EDGE", "1600"))
SIGLIP_TAG_THRESHOLD = float(os.environ.get("SIGLIP_TAG_THRESHOLD", "0.1"))
RAM_TAG_THRESHOLD    = float(os.environ.get("RAM_TAG_THRESHOLD", "0.68"))

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except NameError:
    DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# ── Concurrency limiter ────────────────────────────────────────────────────────
# Caps how many requests run inference simultaneously. Callers that exceed the
# limit get a 429 immediately rather than queueing and silently exhausting RAM.
_concurrency_sem = threading.Semaphore(MAX_CONCURRENCY)

# ── Thread pool: inference runs off the Flask request thread ──────────────────
_inference_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY * 3, thread_name_prefix="inference")

# ── Florence-2 ─────────────────────────────────────────────────────────────────
florence_model     = None
florence_processor = None


def load_florence_model() -> None:
    global florence_model, florence_processor
    if not MODELS_AVAILABLE:
        return
    logger.info("Loading Florence-2 model (%s) on %s ...", FLORENCE_MODEL, DEVICE)
    florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    florence_model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL, trust_remote_code=True, torch_dtype=dtype,
    ).to(DEVICE)
    florence_model.eval()
    logger.info("Florence-2 model loaded.")


def _florence_generate(pil_image: Image.Image, task: str) -> str:
    """Run one Florence-2 task and return the post-processed text, special tokens stripped."""
    inputs = florence_processor(text=task, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = florence_processor.post_process_generation(
        generated_text, task=task, image_size=(pil_image.width, pil_image.height),
    )
    raw = parsed.get(task, "")
    # post_process_generation sometimes leaves Florence-2 special tokens
    # (e.g. <poly>, <loc_N>) in the output for tasks it doesn't fully handle.
    # A first pass removes complete <token> patterns; a second pass removes
    # any orphaned < or > characters left behind (e.g. "GENERATE_TAGS>").
    cleaned = re.sub(r"<[^>]*>", "", raw)
    return re.sub(r"[<>]", "", cleaned).strip()


def _florence_run_od(pil_image: Image.Image) -> list[str]:
    """Return deduplicated object labels from Florence-2 <OD>."""
    inputs = florence_processor(text="<OD>", images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = florence_processor.post_process_generation(
        generated_text, task="<OD>", image_size=(pil_image.width, pil_image.height),
    )
    labels = parsed.get("<OD>", {}).get("labels", [])
    seen: set[str] = set()
    tags: list[str] = []
    for label in labels:
        if label.lower() not in seen:
            seen.add(label.lower())
            tags.append(label)
    return tags


def get_florence_results(pil_image: Image.Image) -> tuple[list[str], str, str]:
    """
    Run all Florence-2 tasks sequentially in one thread (safe for shared model).
    Returns (tags, description, ocr_text).
    """
    if florence_model is None or florence_processor is None:
        logger.warning("Florence-2 model not loaded.")
        return [], "", ""

    tags: list[str] = []
    description = ""
    ocr_text = ""

    try:
        tags = _florence_run_od(pil_image)
    except Exception:
        logger.error("Florence-2 <OD> failed:\n%s", traceback.format_exc())

    try:
        raw_description = re.sub(r"\s+", " ", _florence_generate(pil_image, "<MORE_DETAILED_CAPTION>")).strip()
        description = re.sub(
            r"^The image \w+\s+(.)",
            lambda m: m.group(1).upper(),
            raw_description,
        )
    except Exception:
        logger.error("Florence-2 <MORE_DETAILED_CAPTION> failed:\n%s", traceback.format_exc())

    try:
        raw_ocr  = _florence_generate(pil_image, "<OCR>")
        ocr_text = re.sub(r"\s+", " ", raw_ocr.encode("ascii", errors="ignore").decode()).strip()
    except Exception:
        logger.error("Florence-2 <OCR> failed:\n%s", traceback.format_exc())

    return tags, description, ocr_text


# ── SigLIP ─────────────────────────────────────────────────────────────────────
siglip_model        = None
siglip_processor    = None
_siglip_text_inputs = None  # pre-tokenized candidate tags, kept on DEVICE

SIGLIP_CANDIDATE_TAGS = [
    "person", "cat", "dog", "face", "portrait", "selfie", "group photo",
    "screenshot", "meme", "photo", "water", "comic", "document", "map",
    "spreadsheet", "email", "chat", "electronics", "website", "chart",
    "code", "text", "sign", "receipt", "book", "car", "building", "room",
    "desk", "food", "outdoor", "nature",
]


def load_siglip_model() -> None:
    global siglip_model, siglip_processor, _siglip_text_inputs
    if not MODELS_AVAILABLE:
        return
    logger.info("Loading SigLIP model (%s) on %s ...", SIGLIP_MODEL_ID, DEVICE)
    siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_ID)
    siglip_model     = SiglipModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE)
    siglip_model.eval()
    with torch.no_grad():
        text_inputs         = siglip_processor(
            text=SIGLIP_CANDIDATE_TAGS, return_tensors="pt", padding="max_length",
        )
        _siglip_text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
    logger.info("SigLIP model loaded.")


def get_siglip_tags(pil_image: Image.Image, threshold: float) -> list[str]:
    if siglip_model is None or _siglip_text_inputs is None:
        return []
    try:
        img_inputs = siglip_processor(images=pil_image, return_tensors="pt")
        img_inputs = {k: v.to(DEVICE) for k, v in img_inputs.items()}
        with torch.no_grad():
            outputs = siglip_model(
                pixel_values=img_inputs["pixel_values"],
                **_siglip_text_inputs,
            )
            # SigLIP uses sigmoid (independent per-tag probabilities), not softmax
            probs = torch.sigmoid(outputs.logits_per_image).squeeze(0).cpu().numpy()
        return [tag for tag, p in zip(SIGLIP_CANDIDATE_TAGS, probs) if p >= threshold]
    except Exception:
        logger.error("SigLIP inference failed:\n%s", traceback.format_exc())
        return []


# ── RAM++ ──────────────────────────────────────────────────────────────────────
ram_model      = None
_ram_transform = None


def load_ram_model() -> None:
    global ram_model, _ram_transform
    if not _RAM_AVAILABLE or not MODELS_AVAILABLE:
        return
    if not os.path.exists(RAM_CHECKPOINT):
        logger.warning("RAM++ checkpoint not found at %s — skipping.", RAM_CHECKPOINT)
        return
    logger.info("Loading RAM++ model from %s on %s ...", RAM_CHECKPOINT, DEVICE)
    _ram_transform = ram_get_transform(image_size=384)
    ram_model = ram_plus(pretrained=RAM_CHECKPOINT, image_size=384, vit="swin_l")
    ram_model.eval()
    ram_model = ram_model.to(DEVICE)
    logger.info("RAM++ model loaded.")


def get_ram_tags(pil_image: Image.Image, threshold: float) -> list[str]:
    if ram_model is None or _ram_transform is None:
        return []
    try:
        image_tensor = _ram_transform(pil_image).unsqueeze(0).to(DEVICE)
        tags_str, _ = inference_ram(image_tensor, ram_model)
        return [t.strip() for t in tags_str.split("|") if t.strip()]
    except Exception:
        logger.error("RAM++ inference failed:\n%s", traceback.format_exc())
        return []


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":                 "ok",
        "florence_model_loaded":  florence_model is not None,
        "siglip_model_loaded":    siglip_model is not None,
        "ram_model_loaded":       ram_model is not None,
        "device":                 DEVICE,
        "max_concurrency":        MAX_CONCURRENCY,
    })


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Accepts an image via multipart/form-data (field: "image") or raw binary body.

    Optional query parameters:
      tag_confidence  float 0-1  minimum SigLIP confidence for secondary tags
                                 (default: SIGLIP_TAG_THRESHOLD env var, default 0.1)

    Returns 503 if any model has not finished loading.
    Returns 429 if MAX_CONCURRENCY active requests are already running.
    """
    not_loaded = [
        name for name, loaded in [
            ("Florence-2", florence_model is not None),
            ("SigLIP",     siglip_model is not None),
            ("RAM++",      ram_model is not None),
        ] if not loaded
    ]
    if not_loaded:
        return jsonify({
            "error":      "Service unavailable — models not yet loaded.",
            "not_loaded": not_loaded,
        }), 503

    tag_confidence_str = request.args.get("tag_confidence")
    if tag_confidence_str is not None:
        try:
            tag_confidence = float(tag_confidence_str)
        except ValueError:
            return jsonify({"error": "tag_confidence must be a float between 0 and 1"}), 400
        if not 0.0 <= tag_confidence <= 1.0:
            return jsonify({"error": "tag_confidence must be between 0 and 1"}), 400
        siglip_threshold = tag_confidence
        ram_threshold    = tag_confidence
    else:
        siglip_threshold = SIGLIP_TAG_THRESHOLD
        ram_threshold    = RAM_TAG_THRESHOLD

    # ── Decode image ──────────────────────────────────────────────────────────
    if request.files and "image" in request.files:
        try:
            stream = request.files["image"].stream
            stream.seek(0)
            pil_image = _open_image(stream.read())
        except Exception as e:
            return jsonify({"error": f"Could not decode uploaded file: {e}"}), 400
    elif request.data:
        try:
            pil_image = _open_image(request.data)
        except Exception as e:
            return jsonify({"error": f"Could not decode raw image body: {e}"}), 400
    else:
        return jsonify({"error": "No image provided. Send multipart field 'image' or a raw binary body."}), 400

    pil_image = pil_image.convert("RGB")

    # Downscale large images — accuracy doesn't meaningfully improve beyond
    # MAX_IMAGE_EDGE px on the longest edge, and cost scales with pixel count.
    if max(pil_image.size) > MAX_IMAGE_EDGE:
        pil_image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
        logger.debug("Downscaled image to %s", pil_image.size)

    # ── Acquire concurrency slot only for the inference phase ─────────────────
    if not _concurrency_sem.acquire(blocking=False):
        return jsonify({
            "error":           "Server is busy — try again shortly.",
            "max_concurrency": MAX_CONCURRENCY,
        }), 429

    try:
        future_florence = _inference_pool.submit(get_florence_results, pil_image)
        future_siglip   = _inference_pool.submit(get_siglip_tags, pil_image, siglip_threshold)
        future_ram      = _inference_pool.submit(get_ram_tags, pil_image, ram_threshold)
        wait([future_florence, future_siglip, future_ram])

        florence_tags, florence_description, florence_text = future_florence.result()
        tags = florence_tags
        seen = {t.lower() for t in tags}
        for tag in [*future_siglip.result(), *future_ram.result()]:
            if tag.lower() not in seen:
                tags.append(tag)
                seen.add(tag.lower())

        return jsonify({
            "tags":        tags,
            "description": florence_description,
            "text":        florence_text,
        })

    finally:
        _concurrency_sem.release()


# ── Load models on import ──────────────────────────────────────────────────────
load_florence_model()
load_siglip_model()
load_ram_model()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)