import logging
import time
import traceback
from threading import Event

import torch
from PIL import Image

from config import DEVICE, MODELS_AVAILABLE, RETRY_TIMEOUT

logger = logging.getLogger(__name__)

SIGLIP_CANDIDATE_TAGS = [
    "screenshot", "meme", "selfie", "portrait", "document", "spreadsheet",
    "email", "chat", "website", "chart", "code", "receipt", "map", "landscape",
    "AI-generated", "photo", "text", "sign", "group", "face",
]


class SigLIPModel:
    """
    Singleton for SigLIP. Thread-safe — can be called concurrently.
    Pre-tokenizes candidate tags in __init__ to avoid repeated tokenisation cost.
    """

    def __init__(self, model_id: str, enabled: bool = True) -> None:
        self._enabled     = enabled
        self._model       = None
        self._processor   = None
        self._text_inputs = None
        if enabled and MODELS_AVAILABLE:
            self._load(model_id)

    def _load(self, model_id: str) -> None:
        from transformers import AutoProcessor, AutoModel
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        logger.info("Loading SigLIP model (%s) on %s ...", model_id, DEVICE)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model     = AutoModel.from_pretrained(model_id, dtype=dtype).to(DEVICE)
        self._model.eval()
        with torch.no_grad():
            text_inputs = self._processor(
                text=SIGLIP_CANDIDATE_TAGS, return_tensors="pt", padding="max_length",
            )
            self._text_inputs = {
                k: v.to(DEVICE, dtype=dtype) if v.is_floating_point() else v.to(DEVICE)
                for k, v in text_inputs.items()
            }
        logger.info("SigLIP model loaded.")

    def is_ready(self) -> bool:
        return self._model is not None and self._text_inputs is not None

    def is_enabled(self) -> bool:
        return self._enabled

    def classify(self, image: Image.Image, threshold: float, cancel: Event | None = None) -> list[str]:
        """Synchronous. Thread-safe — caller may submit to a pool for concurrency."""
        logger.debug("SigLIP classify started")
        if not self.is_ready():
            return []
        retry_since = None
        while True:
            try:
                img_inputs = self._processor(images=image, return_tensors="pt")
                model_dtype = next(self._model.parameters()).dtype
                img_inputs = {
                    k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
                    for k, v in img_inputs.items()
                }
                with torch.no_grad():
                    outputs = self._model(
                        pixel_values=img_inputs["pixel_values"],
                        **self._text_inputs,
                    )
                    probs = torch.sigmoid(outputs.logits_per_image).squeeze(0).cpu().numpy()
                logger.debug("SigLIP classify complete")
                return [tag for tag, p in zip(SIGLIP_CANDIDATE_TAGS, probs) if p >= threshold]
            except Exception:
                if cancel and cancel.is_set():
                    raise
                now = time.monotonic()
                if retry_since is None:
                    retry_since = now
                elif now - retry_since >= RETRY_TIMEOUT:
                    logger.error("SigLIP giving up after retrying for %ds", RETRY_TIMEOUT)
                    raise
                logger.error("SigLIP inference failed, retrying:\n%s", traceback.format_exc())
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(1)

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._model is None:
            return "not loaded"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return DEVICE
