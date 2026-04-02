"""
Image Analysis Server
- Uses Florence-2 for image tagging (OD), description, and OCR
- Uses RAM++ and SigLIP as additional tagging passes, results merged
- Request queue with configurable max concurrency to prevent OOM
- Exposes POST /analyse and GET /health
"""

import os
import logging

from flask import Flask, request, jsonify

from config import (
    DEVICE,
    FLORENCE_MODEL, SIGLIP_MODEL_ID, RAM_CHECKPOINT, SPACY_MODEL, OCR_CORRECTION_MODEL_ID,
    ENABLE_FLORENCE, ENABLE_SIGLIP, ENABLE_RAM, ENABLE_OCR_CORRECTION, ENABLE_SPACY,
)
from models import Florence2Model, SigLIPModel, RAMModel, OCRCorrectionModel, SpacyModel
from controller import AnalysisController

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# ── Singleton instantiation ────────────────────────────────────────────────────
# Models load synchronously here at import time. gunicorn's --timeout 300
# gives enough headroom. Do NOT use --preload with gunicorn.
_florence       = Florence2Model(FLORENCE_MODEL,          ENABLE_FLORENCE)
_siglip         = SigLIPModel(SIGLIP_MODEL_ID,            ENABLE_SIGLIP)
_ram            = RAMModel(RAM_CHECKPOINT,                 ENABLE_RAM)
_ocr_correction = OCRCorrectionModel(OCR_CORRECTION_MODEL_ID, ENABLE_OCR_CORRECTION)
_spacy          = SpacyModel(SPACY_MODEL,                  ENABLE_SPACY)

controller = AnalysisController(_florence, _siglip, _ram, _ocr_correction, _spacy)

# ── Startup device report ──────────────────────────────────────────────────────
report = controller.device_report()
logger.info("=" * 56)
logger.info("Model device report")
for name, device in report.items():
    logger.info("  %-16s: %s", name, device)
logger.info("=" * 56)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    def _status(model):
        if not model.is_enabled():
            return "disabled"
        return "ok" if model.is_ready() else "loading"

    return jsonify({
        "status": "ok",
        "models": {
            "florence":       _status(_florence),
            "siglip":         _status(_siglip),
            "ram":            _status(_ram),
            "ocr_correction": _status(_ocr_correction),
            "spacy":          _status(_spacy),
        },
        "device": DEVICE,
    })


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Accepts an image via multipart/form-data (field: "image") or raw binary body.
    Returns 503 if any model has not finished loading.
    """
    not_loaded = controller.not_ready()
    if not_loaded:
        return jsonify({
            "error":      "Service unavailable — models not yet loaded.",
            "not_loaded": not_loaded,
        }), 503

    try:
        if request.files and "image" in request.files:
            stream = request.files["image"].stream
            stream.seek(0)
            image = controller.decode_image(stream.read())
        elif request.data:
            image = controller.decode_image(request.data)
        else:
            return jsonify({"error": "No image provided. Send multipart field 'image' or a raw binary body."}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(controller.analyse(image))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
