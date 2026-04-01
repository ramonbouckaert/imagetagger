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
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="fairscale")
warnings.filterwarnings("ignore", message="Asking to truncate to max_length but no maximum length is provided")

# ── Core imports ───────────────────────────────────────────────────────────────
_import_errors: list[str] = []

try:
    import torch
except ImportError as e:
    _import_errors.append(f"  • torch — {e}\n    Fix: {sys.executable} -m pip install torch torchvision")

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
    from transformers import SiglipModel, SiglipProcessor
    from transformers import pipeline as hf_pipeline
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

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except ImportError as _spacy_import_err:
    _SPACY_AVAILABLE = False
    logging.warning("spacy unavailable — noun chunk tags will be disabled. Install with: %s -m pip install spacy && python -m spacy download en_core_web_sm", sys.executable)

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
    if not data:
        raise ValueError("Empty body — 0 bytes received")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode; catches truncated files early
        return img
    except Exception as first_err:
        # Fallback: decode via ffmpeg, which is lenient about malformed EXIF metadata.
        # Pillow's libavif bails on files with a double-wrapped Exif offset box;
        # ffmpeg ignores the bad metadata and decodes the image anyway.
        logger.info("Pillow open failed (%s), retrying via pyvips", first_err)
        try:
            import pyvips
            vips_img = pyvips.Image.new_from_buffer(data, "")
            png_bytes = vips_img.write_to_buffer(".png")
            logger.info("pyvips fallback succeeded")
            return Image.open(io.BytesIO(png_bytes))
        except Exception as vips_err:
            header = data[:32].hex(" ")
            logger.error("pyvips fallback failed: %s", vips_err)
            raise ValueError(
                f"{first_err} — received {len(data):,} bytes, "
                f"first 32 bytes (hex): {header}"
            ) from first_err


_TYPO_CORRECTIONS = {
    "dinning table": "dining table",
}

# Noun chunks containing any of these words are dropped entirely.
_SPACY_CAPTION_BLOCKLIST = {
    "that", "they", "another", "foreground", "background", "left", "right", "top", "bottom", "something", "you",
    "overall", "which", "type", "them", "image", "him", "her", "he", "she", "this", "anything", "side", "who",
    "themself", "themselves", "other", "others", "atmosphere"
}
_SPACY_OCR_BLOCKLIST = {
    "that", "they", "another", "something", "you", "which", "them", "him", "her", "he", "she", "this",
    "anything", "who", "themself", "themselves", "other", "others"
}

def _normalise_tag(tag: str) -> list[str]:
    """Lowercase, strip leading articles, and fix known typos. Returns 0–n tags."""
    tag = tag.lower().strip()
    parts = tag.split(" or ")
    if len(parts) > 1:
        return [t for part in parts for t in _normalise_tag(part)]
    tag = tag.replace(" - ", "-")
    tag = re.sub(r'^(a|the)\s+', '', tag)
    tag = re.sub(r'^(one|same|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+', '', tag)
    tag = _TYPO_CORRECTIONS.get(tag, tag)
    tag = " ".join(t for t in tag.split() if re.search(r'[a-zA-Z0-9]{3}', t))
    if not tag or len(tag) < 3:
        return []
    if tag.startswith("human "):
        suffix = tag[len("human "):]
        return [tag, suffix] if len(suffix) >= 3 else [tag]
    return [tag]


def _compile(model):
    """Attempt torch.compile(); silently skip if unavailable (PyTorch < 2.0)."""
    try:
        return torch.compile(model)
    except Exception as e:
        # logger may not be initialised yet at module load time
        logging.warning("torch.compile() unavailable, running uncompiled: %s", e)
        return model


# ── Config ─────────────────────────────────────────────────────────────────────
FLORENCE_MODEL          = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")
SIGLIP_MODEL_ID         = os.environ.get("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
RAM_CHECKPOINT          = os.environ.get("RAM_CHECKPOINT", "ram_plus_swin_large_14m.pth")
SPACY_MODEL             = os.environ.get("SPACY_MODEL", "en_core_web_sm")
OCR_CORRECTION_MODEL_ID = os.environ.get("OCR_CORRECTION_MODEL", "yelpfeast/byt5-base-english-ocr-correction")
MAX_CONCURRENCY      = int(os.environ.get("MAX_CONCURRENCY", "2"))
MAX_IMAGE_EDGE       = int(os.environ.get("MAX_IMAGE_EDGE", "1600"))
SIGLIP_TAG_THRESHOLD = float(os.environ.get("SIGLIP_TAG_THRESHOLD", "0.1"))
RAM_TAG_THRESHOLD    = float(os.environ.get("RAM_TAG_THRESHOLD", "0.68"))

# ── Model enable flags ─────────────────────────────────────────────────────────
# Set any to False to skip loading and running that model entirely.
ENABLE_FLORENCE       = True  # Florence-2: OD tags, image description, OCR
ENABLE_SIGLIP         = True  # SigLIP: zero-shot tag classification
ENABLE_RAM            = True  # RAM++: open-set scene/object tagging
ENABLE_OCR_CORRECTION = True  # Spell-correct OCR output (ByT5 seq2seq)
ENABLE_SPACY          = True  # spaCy noun chunk extraction from Florence-2 description

try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        # Try to get a specific reason why CUDA isn't available
        try:
            torch.cuda.init()
        except Exception as _cuda_err:
            logging.warning("CUDA unavailable: %s", _cuda_err)
        logging.warning(
            "CUDA not available — running on CPU. "
            "torch.version.cuda=%s, torch.__version__=%s",
            torch.version.cuda, torch.__version__,
        )
except NameError:
    DEVICE = "cpu"

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# ── Concurrency limiter ────────────────────────────────────────────────────────
# Caps how many requests run inference simultaneously. Callers that exceed the
# limit get a 429 immediately rather than queueing and silently exhausting RAM.
_concurrency_sem = threading.Semaphore(MAX_CONCURRENCY)

# ── Thread pool: inference runs off the Flask request thread ──────────────────
# 3 tasks per request: Florence-2 (OD/CAP/OCR sequentially) + SigLIP + RAM++
_inference_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY * 3, thread_name_prefix="inference")

# ── Florence-2 ─────────────────────────────────────────────────────────────────
florence_processor = None
florence_model     = None  # single instance; OD/CAP/OCR tasks run sequentially


def load_florence_model() -> None:
    global florence_processor, florence_model
    if not MODELS_AVAILABLE or not ENABLE_FLORENCE:
        return
    logger.info("Loading Florence-2 model (%s) on %s ...", FLORENCE_MODEL, DEVICE)
    florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    florence_model = AutoModelForCausalLM.from_pretrained(
                         FLORENCE_MODEL, trust_remote_code=True, torch_dtype=dtype,
                     ).to(DEVICE)
    florence_model.eval()
    logger.info("Florence-2 model loaded.")


def _florence_generate(task: str, pil_image: Image.Image, *, max_new_tokens: int = 1024, num_beams: int = 3) -> str:
    """Run one Florence-2 task on the shared model instance and return cleaned text."""
    inputs = florence_processor(text=task, images=pil_image, return_tensors="pt")
    model_dtype = next(florence_model.parameters()).dtype
    inputs = {k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
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


def get_florence_tags(pil_image: Image.Image) -> list[str]:
    """Return deduplicated object labels from Florence-2 <OD>."""
    logger.debug("Florence OD started")
    if florence_model is None or florence_processor is None:
        logger.warning("Florence-2 model not loaded.")
        return []
    try:
        inputs = florence_processor(text="<OD>", images=pil_image, return_tensors="pt")
        model_dtype = next(florence_model.parameters()).dtype
        inputs = {k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE) for k, v in inputs.items()}
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
        logger.debug("Florence OD complete")
        return tags
    except Exception:
        logger.error("Florence-2 <OD> failed:\n%s", traceback.format_exc())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return []


def get_florence_description(pil_image: Image.Image) -> str:
    """Return a cleaned caption from Florence-2 <MORE_DETAILED_CAPTION>."""
    logger.debug("Florence CAP started")
    if florence_model is None or florence_processor is None:
        logger.warning("Florence-2 model not loaded.")
        return ""
    try:
        raw = re.sub(r"\s+", " ", _florence_generate("<MORE_DETAILED_CAPTION>", pil_image)).strip()
        logger.debug("Florence CAP complete")
        return re.sub(
            r"^The image \w+\s+(.)",
            lambda m: m.group(1).upper(),
            raw,
        )
    except Exception:
        logger.error("Florence-2 <MORE_DETAILED_CAPTION> failed:\n%s", traceback.format_exc())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return ""

def get_florence_ocr(pil_image: Image.Image) -> str:
    """Return ASCII-only OCR text from Florence-2 <OCR>."""
    logger.debug("Florence OCR started")
    if florence_model is None or florence_processor is None:
        logger.warning("Florence-2 model not loaded.")
        return ""
    try:
        raw = _florence_generate("<OCR>", pil_image, max_new_tokens=256, num_beams=3)
        logger.debug("Florence OCR complete")
        text = re.sub(r"\s+", " ", raw.encode("ascii", errors="ignore").decode()).strip()
        return text if re.search(r"[a-zA-Z0-9]{2}", text) else ""
    except Exception:
        logger.error("Florence-2 <OCR> failed:\n%s", traceback.format_exc())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return ""


def _run_florence(pil_image: Image.Image) -> dict:
    """Run all Florence-2 tasks sequentially on the single model instance."""
    return {
        "od":  get_florence_tags(pil_image),
        "cap": get_florence_description(pil_image),
        "ocr": get_florence_ocr(pil_image),
    }


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
    if not MODELS_AVAILABLE or not ENABLE_SIGLIP:
        return
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    logger.info("Loading SigLIP model (%s) on %s ...", SIGLIP_MODEL_ID, DEVICE)
    siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_ID)
    siglip_model     = SiglipModel.from_pretrained(SIGLIP_MODEL_ID, torch_dtype=dtype).to(DEVICE)
    siglip_model.eval()
    with torch.no_grad():
        text_inputs = siglip_processor(
            text=SIGLIP_CANDIDATE_TAGS, return_tensors="pt", padding="max_length",
        )
        _siglip_text_inputs = {
            k: v.to(DEVICE, dtype=dtype) if v.is_floating_point() else v.to(DEVICE)
            for k, v in text_inputs.items()
        }
    logger.info("SigLIP model loaded.")


def get_siglip_tags(pil_image: Image.Image, threshold: float) -> list[str]:
    logger.debug("Siglip tags started")
    if siglip_model is None or _siglip_text_inputs is None:
        return []
    try:
        img_inputs = siglip_processor(images=pil_image, return_tensors="pt")
        model_dtype = next(siglip_model.parameters()).dtype
        img_inputs = {
            k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
            for k, v in img_inputs.items()
        }
        with torch.no_grad():
            outputs = siglip_model(
                pixel_values=img_inputs["pixel_values"],
                **_siglip_text_inputs,
            )
            # SigLIP uses sigmoid (independent per-tag probabilities), not softmax
            probs = torch.sigmoid(outputs.logits_per_image).squeeze(0).cpu().numpy()
        logger.debug("Siglip tags complete")
        return [tag for tag, p in zip(SIGLIP_CANDIDATE_TAGS, probs) if p >= threshold]
    except Exception:
        logger.error("SigLIP inference failed:\n%s", traceback.format_exc())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return []


# ── RAM++ ──────────────────────────────────────────────────────────────────────
ram_model      = None
_ram_transform = None


def load_ram_model() -> None:
    global ram_model, _ram_transform
    if not _RAM_AVAILABLE or not MODELS_AVAILABLE or not ENABLE_RAM:
        return
    if not os.path.exists(RAM_CHECKPOINT):
        logger.warning("RAM++ checkpoint not found at %s — skipping.", RAM_CHECKPOINT)
        return
    logger.info("Loading RAM++ model from %s on %s ...", RAM_CHECKPOINT, DEVICE)
    _ram_transform = ram_get_transform(image_size=384)
    ram_model = ram_plus(pretrained=RAM_CHECKPOINT, image_size=384, vit="swin_l")
    ram_model.eval()
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    ram_model = ram_model.to(DEVICE, dtype=dtype)
    logger.info("RAM++ model loaded.")


def get_ram_tags(pil_image: Image.Image, threshold: float) -> list[str]:
    logger.debug("RAM++ tags started")
    if ram_model is None or _ram_transform is None:
        return []
    try:
        model_dtype = next(ram_model.parameters()).dtype
        image_tensor = _ram_transform(pil_image).unsqueeze(0).to(DEVICE, dtype=model_dtype)
        tags_str, _ = inference_ram(image_tensor, ram_model)
        logger.debug("RAM++ tags complete")
        return [t.strip() for t in tags_str.split("|") if t.strip()]
    except Exception:
        logger.error("RAM++ inference failed:\n%s", traceback.format_exc())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return []




# ── OCR spell correction ───────────────────────────────────────────────────────
_ocr_correction_pipeline = None


def load_ocr_correction_model() -> None:
    global _ocr_correction_pipeline
    if not MODELS_AVAILABLE or not ENABLE_OCR_CORRECTION:
        return
    logger.info("Loading OCR correction model (%s) on %s ...", OCR_CORRECTION_MODEL_ID, DEVICE)
    device_id = 0 if DEVICE == "cuda" else -1
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    _ocr_correction_pipeline = hf_pipeline(
        "text2text-generation",
        model=OCR_CORRECTION_MODEL_ID,
        device=device_id,
        model_kwargs={"torch_dtype": dtype},
    )
    logger.info("OCR correction model loaded.")


def correct_ocr_text(text: str) -> str:
    """Return spell-corrected version of OCR text; falls back to original on error."""
    logger.debug("OCR correction started")
    if _ocr_correction_pipeline is None or not text.strip():
        return text
    try:
        tok = _ocr_correction_pipeline.tokenizer
        token_count = len(tok(text)["input_ids"])
        result = _ocr_correction_pipeline(
            [text],
            max_new_tokens=int(token_count * 1.1),
            truncation=True,
        )
        corrected = result[0]["generated_text"].strip()
        logger.debug("OCR correction complete")
        return corrected
    except Exception:
        logger.error("OCR correction failed:\n%s", traceback.format_exc())
        return text


# ── spaCy noun chunks ──────────────────────────────────────────────────────────
spacy_nlp = None


def load_spacy_model() -> None:
    global spacy_nlp
    if not _SPACY_AVAILABLE or not ENABLE_SPACY:
        return
    logger.info("Loading spaCy model (%s) ...", SPACY_MODEL)
    spacy_nlp = _spacy.load(SPACY_MODEL)
    logger.info("spaCy model loaded.")


def get_noun_chunk_tags(description: str, blocklist: set[str] = _SPACY_CAPTION_BLOCKLIST) -> list[str]:
    """Extract lowercased noun chunks from description, stripping determiners."""
    logger.debug("spaCy noun chunks started")
    if spacy_nlp is None or not description:
        return []
    try:
        doc = spacy_nlp(description)
        tags: list[str] = []
        for chunk in doc.noun_chunks:
            text = " ".join(t.text for t in chunk if t.dep_ not in ("det", "poss")).strip().lower()
            if text and not blocklist.intersection(text.split()):
                tags.append(text)
        logger.debug("spaCy noun chunks complete: %s", tags)
        return tags
    except Exception:
        logger.error("spaCy noun chunk extraction failed:\n%s", traceback.format_exc())
        return []


def get_noun_tags(text: str, blocklist: set[str] = _SPACY_OCR_BLOCKLIST) -> list[str]:
    """Extract individual lowercased nouns from text."""
    logger.debug("spaCy noun extraction started")
    if spacy_nlp is None or not text:
        return []
    try:
        doc = spacy_nlp(text)
        tags: list[str] = []
        seen: set[str] = set()
        for token in doc:
            if token.pos_ == "NOUN":
                word = token.text.lower()
                if word not in blocklist and word not in seen:
                    tags.append(word)
                    seen.add(word)
        logger.debug("spaCy noun extraction complete: %s", tags)
        return tags
    except Exception:
        logger.error("spaCy noun extraction failed:\n%s", traceback.format_exc())
        return []


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    def _status(enabled, loaded):
        if not enabled:
            return "disabled"
        return "ok" if loaded else "loading"
    return jsonify({
        "status": "ok",
        "models": {
            "florence": _status(ENABLE_FLORENCE, florence_model   is not None),
            "siglip":   _status(ENABLE_SIGLIP,   siglip_model     is not None),
            "ram":      _status(ENABLE_RAM,       ram_model        is not None),
            "ocr_correction": _status(ENABLE_OCR_CORRECTION, _ocr_correction_pipeline is not None),
            "spacy":          _status(ENABLE_SPACY,          spacy_nlp                is not None),
        },
        "device":          DEVICE,
        "max_concurrency": MAX_CONCURRENCY,
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
        name for name, enabled, loaded in [
            ("Florence-2",       ENABLE_FLORENCE, florence_model   is not None),
            ("SigLIP",           ENABLE_SIGLIP,   siglip_model     is not None),
            ("RAM++",            ENABLE_RAM,      ram_model        is not None),
            ("OCR correction",       ENABLE_OCR_CORRECTION, _ocr_correction_pipeline is not None),
            ("spaCy",                ENABLE_SPACY,          spacy_nlp                is not None),
        ] if enabled and not loaded
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
        futures = {}
        if ENABLE_FLORENCE:
            futures["florence"] = _inference_pool.submit(_run_florence, pil_image)
        if ENABLE_SIGLIP:
            futures["siglip"]   = _inference_pool.submit(get_siglip_tags, pil_image, siglip_threshold)
        if ENABLE_RAM:
            futures["ram"]      = _inference_pool.submit(get_ram_tags, pil_image, ram_threshold)
        if futures:
            wait(futures.values())

        florence_results = futures["florence"].result() if "florence" in futures else {}

        tags: list[str] = []
        seen: set[str] = set()

        def _add_tag(raw: str) -> None:
            for norm in _normalise_tag(raw):
                if norm not in seen:
                    tags.append(norm)
                    seen.add(norm)

        for tag in florence_results.get("od", []):
            _add_tag(tag)
        for tag in [
            *(futures["siglip"].result() if "siglip" in futures else []),
            *(futures["ram"].result()    if "ram"    in futures else []),
        ]:
            _add_tag(tag)

        # OCR spell correction — runs after Florence OCR.
        ocr_text = florence_results.get("ocr", "")
        if ENABLE_OCR_CORRECTION and _ocr_correction_pipeline is not None and ocr_text:
            ocr_text = _inference_pool.submit(correct_ocr_text, ocr_text).result()

        cap = florence_results.get("cap", "")
        cap_chunks_future = cap_nouns_future = ocr_nouns_future = None
        if ENABLE_SPACY and spacy_nlp is not None:
            if cap:
                cap_chunks_future = _inference_pool.submit(get_noun_chunk_tags, cap)
                cap_nouns_future  = _inference_pool.submit(get_noun_tags, cap)
            if ocr_text:
                ocr_nouns_future  = _inference_pool.submit(get_noun_tags, ocr_text)
        for future in (cap_chunks_future, cap_nouns_future, ocr_nouns_future):
            if future:
                for tag in future.result():
                    _add_tag(tag)

        return jsonify({
            "tags":        tags,
            "description": florence_results.get("cap", ""),
            "text":        ocr_text,
        })

    finally:
        _concurrency_sem.release()


# ── Startup device report ──────────────────────────────────────────────────────

def _report_devices() -> None:
    """After all models load, log the actual device each one is running on."""
    def _device_of(enabled, model) -> str:
        if not enabled:
            return "disabled"
        if model is None:
            return "not loaded"
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return DEVICE

    logger.info("=" * 56)
    logger.info("Model device report")
    logger.info("  Florence-2 : %s", _device_of(ENABLE_FLORENCE, florence_model))
    logger.info("  SigLIP     : %s", _device_of(ENABLE_SIGLIP,   siglip_model))
    logger.info("  RAM++      : %s", _device_of(ENABLE_RAM,       ram_model))
    logger.info("  OCR-corr   : %s", _device_of(ENABLE_OCR_CORRECTION, _ocr_correction_pipeline.model if _ocr_correction_pipeline is not None else None))
    logger.info("  spaCy      : %s", "cpu" if spacy_nlp is not None else ("disabled" if not ENABLE_SPACY else "not loaded"))
    logger.info("=" * 56)


# ── Load models on import ──────────────────────────────────────────────────────
load_florence_model()
load_siglip_model()
load_ram_model()
load_ocr_correction_model()
load_spacy_model()
_report_devices()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)