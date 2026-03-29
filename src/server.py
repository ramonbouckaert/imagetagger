"""
Image Analysis Server
- Uses Florence-2 for image tagging
- Uses SigLIP as a second tagging pass, results merged with Florence-2
- Uses Tesseract OCR for text extraction (multi-PSM)
- Runs tagging and OCR concurrently per request
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

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
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

if _import_errors:
    logging.warning(
        "Some models will be unavailable. Failed imports:\n%s\n"
        "NOTE: Always use '%s -m pip install ...' (not bare 'pip install') "
        "to ensure packages install into the same Python that is running this server.",
        "\n".join(_import_errors),
        sys.executable,
    )

MODELS_AVAILABLE = len(_import_errors) == 0

# ── Config ─────────────────────────────────────────────────────────────────────
FLORENCE_MODEL       = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")
SIGLIP_MODEL_ID      = os.environ.get("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
MAX_CONCURRENCY      = int(os.environ.get("MAX_CONCURRENCY", "2"))
MAX_IMAGE_EDGE       = int(os.environ.get("MAX_IMAGE_EDGE", "1600"))
SIGLIP_TAG_THRESHOLD = float(os.environ.get("SIGLIP_TAG_THRESHOLD", "0.1"))

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except NameError:
    DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Concurrency limiter ────────────────────────────────────────────────────────
# Caps how many requests run inference simultaneously. Callers that exceed the
# limit get a 429 immediately rather than queueing and silently exhausting RAM.
_concurrency_sem = threading.Semaphore(MAX_CONCURRENCY)

# ── Thread pool: tagging and OCR run concurrently per request ──────────────────
_inference_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY * 2, thread_name_prefix="inference")

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


def get_florence_tags(pil_image: Image.Image) -> list[str]:
    if florence_model is None or florence_processor is None:
        logger.warning("Florence-2 model not loaded; returning empty tags.")
        return []
    try:
        task   = "<GENERATE_TAGS>"
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
        parsed         = florence_processor.post_process_generation(
            generated_text, task=task, image_size=(pil_image.width, pil_image.height),
        )
        tags_str = parsed.get(task, "")
        return [t.strip() for t in tags_str.split(",") if t.strip()]
    except Exception:
        logger.error("Florence-2 inference failed:\n%s", traceback.format_exc())
        return []


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
                input_ids=_siglip_text_inputs["input_ids"],
                attention_mask=_siglip_text_inputs["attention_mask"],
            )
            # SigLIP uses sigmoid (independent per-tag probabilities), not softmax
            probs = torch.sigmoid(outputs.logits_per_image).squeeze(0).cpu().numpy()
        return [tag for tag, p in zip(SIGLIP_CANDIDATE_TAGS, probs) if p >= threshold]
    except Exception:
        logger.error("SigLIP inference failed:\n%s", traceback.format_exc())
        return []


def get_tags(pil_image: Image.Image, siglip_threshold: float = 0.0) -> list[str]:
    """Run Florence-2 and SigLIP, merge results (Florence-2 order first, no duplicates)."""
    effective_threshold = siglip_threshold if siglip_threshold > 0.0 else SIGLIP_TAG_THRESHOLD
    florence_tags = get_florence_tags(pil_image)
    siglip_tags   = get_siglip_tags(pil_image, effective_threshold)
    seen          = {t.lower() for t in florence_tags}
    merged        = list(florence_tags)
    for tag in siglip_tags:
        if tag.lower() not in seen:
            merged.append(tag)
            seen.add(tag.lower())
    return merged


# ── OCR ────────────────────────────────────────────────────────────────────────

# Tesseract PSM modes tried in order. The pass with the highest mean word
# confidence across words-with-text wins.
_PSM_MODES = [3, 6, 11]

# A PSM pass whose overall mean confidence falls below this is discarded
# entirely — its output is noise from a confused Tesseract run.
_MIN_VIABLE_CONF = 50.0


def _is_dark_background(grey: np.ndarray) -> bool:
    # Use 85 rather than 127 — midtone images (mean ~125) are not genuinely
    # dark-background and should not be inverted.
    return float(grey.mean()) < 85


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Prepare an image for Tesseract:
      1. Convert to greyscale; upscale if shorter than 1000px.
      2. Invert if the background is dark (e.g. dark-mode UI).
      3. Quick confidence probe — if Tesseract is already happy (≥70),
         return the greyscale image as-is.
      4. Otherwise apply adaptive threshold to clean up noisy photos.
    """
    grey = pil_image.convert("L")

    w, h = grey.size
    if max(w, h) < 1000:
        scale = 1000 / max(w, h)
        grey  = grey.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    if _is_dark_background(np.array(grey)):
        logger.debug("OCR: inverting dark background")
        grey = ImageOps.invert(grey)

    probe     = pytesseract.image_to_data(
        grey,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 3",
    )
    valid_c   = [c for c in probe["conf"] if c > 0]
    mean_conf = sum(valid_c) / len(valid_c) if valid_c else 0.0

    if mean_conf >= 70:
        logger.debug("OCR: greyscale sufficient (conf %.1f)", mean_conf)
        return grey

    logger.debug("OCR: applying adaptive threshold (conf %.1f)", mean_conf)
    grey_arr  = np.array(grey)
    blurred   = np.array(grey.filter(ImageFilter.GaussianBlur(radius=15)))
    processed = ((grey_arr.astype(np.int16) - blurred.astype(np.int16) + 15) > 0).astype(np.uint8) * 255
    return Image.fromarray(processed)


def _ocr_image(image: Image.Image, min_confidence: int, config: str) -> tuple[list[str], float]:
    """
    Run one Tesseract pass.

    Returns (confident_words, mean_confidence) where mean_confidence is computed
    only over rows that contain actual text — Tesseract sometimes returns a
    single empty-string word with high confidence (e.g. 95) when it finds
    nothing, which would otherwise win the PSM competition unfairly.
    """
    data      = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config=config,
    )
    texts     = [str(t) for t in data["text"]]
    confs     = data["conf"]
    valid_confs = [c for t, c in zip(texts, confs) if t.strip() and c > 0]
    mean_conf   = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
    confident   = [t.strip() for t, c in zip(texts, confs) if t.strip() and c >= min_confidence]
    return confident, mean_conf



def _normalise(text: str) -> str:
    """
    Lowercase and strip punctuation for search, while preserving:
      - Decimal numbers  e.g. 4.68
      - Times            e.g. 1:23
      - Domain names     e.g. example.com
      - Contractions     e.g. don't
    """
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"(?<=[0-9])\.(?=[0-9])",   "decpoint",  text)
    text = re.sub(r"(?<=[0-9]):(?=[0-9])",    "timecolon", text)
    text = re.sub(r"(?<=[a-z]{2})\.(?=[a-z]{2})", "dotdot", text)
    text = re.sub(r"[^a-z0-9 ']", "", text)
    text = text.replace("decpoint", ".").replace("timecolon", ":").replace("dotdot", ".")
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", "", text)
    return re.sub(r" +", " ", text).strip()


def _looks_real(token: str) -> bool:
    """Return True if a token has enough alphanumeric content to be real text."""
    alnum = sum(c.isalnum() for c in token)
    return len(token) >= 2 and (alnum / len(token)) >= 0.4


def get_ocr_text(pil_image: Image.Image, min_confidence: int = 60) -> str:
    """
    Multi-PSM, multi-language OCR.

    Runs PSM 3 / 6 / 11 on the preprocessed image, picks the pass with the
    highest mean word confidence (above _MIN_VIABLE_CONF), then filters and
    normalises. Language is auto-detected via Tesseract OSD.
    """
    try:
        preprocessed = preprocess_for_ocr(pil_image)
        best_words: list[str] = []
        best_conf             = -1.0

        for psm in _PSM_MODES:
            words, mean_conf = _ocr_image(
                preprocessed, min_confidence, f"--oem 3 --psm {psm} -l eng",
            )
            logger.debug("PSM %d -> %d words, conf %.1f", psm, len(words), mean_conf)
            if mean_conf > best_conf and mean_conf >= _MIN_VIABLE_CONF:
                best_conf  = mean_conf
                best_words = words

        if not best_words:
            return ""

        return _normalise(" ".join(w for w in best_words if _looks_real(w)))

    except Exception:
        logger.error("Tesseract OCR failed:\n%s", traceback.format_exc())
        return ""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":                 "ok",
        "florence_model_loaded":  florence_model is not None,
        "siglip_model_loaded":    siglip_model is not None,
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

    Returns 429 if MAX_CONCURRENCY active requests are already running.
    """
    try:
        tag_confidence = float(request.args.get("tag_confidence", 0.0))
    except ValueError:
        return jsonify({"error": "tag_confidence must be a float between 0 and 1"}), 400

    # ── Decode image ──────────────────────────────────────────────────────────
    if request.files and "image" in request.files:
        try:
            pil_image = Image.open(io.BytesIO(request.files["image"].stream.read()))
        except Exception as e:
            return jsonify({"error": f"Could not decode uploaded file: {e}"}), 400
    elif request.data:
        try:
            pil_image = Image.open(io.BytesIO(request.data))
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
        future_tags = _inference_pool.submit(get_tags, pil_image, tag_confidence)
        future_text = _inference_pool.submit(get_ocr_text, pil_image)
        wait([future_tags, future_text])

        return jsonify({
            "tags": future_tags.result(),
            "text": future_text.result(),
        })

    finally:
        _concurrency_sem.release()


# ── Load models on import ──────────────────────────────────────────────────────
load_florence_model()
load_siglip_model()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)