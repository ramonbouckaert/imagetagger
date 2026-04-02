"""
Configuration — environment variables, constants, device detection.
All other modules import from here; nothing here imports from other app modules.
"""

import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="fairscale")
warnings.filterwarnings("ignore", message="Asking to truncate to max_length but no maximum length is provided")
warnings.filterwarnings("ignore", message="The new embeddings will be initialized from a multivariate normal distribution")

# Models are downloaded by the entrypoint before the server starts.
# Default to offline mode so every startup doesn't hit HuggingFace to
# validate cached files. Override with HF_HUB_OFFLINE=0 to allow downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Core imports ───────────────────────────────────────────────────────────────
_import_errors: list[str] = []

try:
    import torch
except ImportError as e:
    _import_errors.append(f"  • torch — {e}\n    Fix: {sys.executable} -m pip install torch torchvision")

try:
    from transformers import (  # noqa: F401 — imported for availability check
        AutoProcessor,
        AutoModel,
        Florence2ForConditionalGeneration,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
    )
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

MODELS_AVAILABLE: bool = len(_import_errors) == 0

# ── Model IDs / paths ──────────────────────────────────────────────────────────
FLORENCE_MODEL          = os.environ.get("FLORENCE_MODEL", "florence-community/Florence-2-large")
SIGLIP_MODEL_ID         = os.environ.get("SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")
RAM_CHECKPOINT          = os.environ.get("RAM_CHECKPOINT", "ram_plus_swin_large_14m.pth")
SPACY_MODEL             = os.environ.get("SPACY_MODEL", "en_core_web_sm")
OCR_CORRECTION_MODEL_ID = os.environ.get("OCR_CORRECTION_MODEL", "yelpfeast/byt5-base-english-ocr-correction")

# ── Runtime tuning ─────────────────────────────────────────────────────────────
MAX_IMAGE_EDGE  = int(os.environ.get("MAX_IMAGE_EDGE", "1600"))
RETRY_TIMEOUT   = int(os.environ.get("RETRY_TIMEOUT", "300"))  # seconds before a failing job gives up
SIGLIP_TAG_THRESHOLD = 0.05
RAM_TAG_THRESHOLD    = 0.68

# ── Model enable flags ─────────────────────────────────────────────────────────
ENABLE_FLORENCE       = True
ENABLE_SIGLIP         = True
ENABLE_RAM            = True
ENABLE_OCR_CORRECTION = True
ENABLE_SPACY          = True

# ── Device selection ───────────────────────────────────────────────────────────
try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
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
