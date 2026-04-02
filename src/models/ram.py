import logging
import time
import traceback
from threading import Event

import torch
from PIL import Image

from config import DEVICE, RETRY_TIMEOUT

logger = logging.getLogger(__name__)


class RAMModel:
    """
    Singleton for RAM++. Thread-safe for read-only inference.
    """

    def __init__(self, checkpoint_path: str, enabled: bool = True) -> None:
        self._enabled   = enabled
        self._model     = None
        self._transform = None
        if enabled:
            self._load(checkpoint_path)

    def _load(self, checkpoint_path: str) -> None:
        import os
        try:
            from ram.models import ram_plus
            from ram import get_transform as ram_get_transform
        except Exception as e:
            logger.warning(
                "recognize-anything unavailable — RAM++ tags will be disabled.\nImport error: %s",
                e, exc_info=True,
            )
            return
        if not os.path.exists(checkpoint_path):
            logger.warning("RAM++ checkpoint not found at %s — skipping.", checkpoint_path)
            return
        logger.info("Loading RAM++ model from %s on %s ...", checkpoint_path, DEVICE)
        self._transform = ram_get_transform(image_size=384)
        self._model = ram_plus(pretrained=checkpoint_path, image_size=384, vit="swin_l")
        self._model.eval()
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self._model = self._model.to(DEVICE, dtype=dtype)
        logger.info("RAM++ model loaded.")

    def is_ready(self) -> bool:
        return self._model is not None and self._transform is not None

    def is_enabled(self) -> bool:
        return self._enabled

    def classify(self, image: Image.Image, threshold: float, cancel: Event | None = None) -> list[str]:
        """Synchronous. Thread-safe — caller may submit to a pool for concurrency."""
        logger.debug("RAM++ classify started")
        if not self.is_ready():
            return []
        from ram import inference_ram
        retry_since = None
        while True:
            try:
                model_dtype = next(self._model.parameters()).dtype
                image_tensor = self._transform(image).unsqueeze(0).to(DEVICE, dtype=model_dtype)
                tags_str, _ = inference_ram(image_tensor, self._model)
                logger.debug("RAM++ classify complete")
                return [t.strip() for t in tags_str.split("|") if t.strip()]
            except Exception:
                if cancel and cancel.is_set():
                    raise
                now = time.monotonic()
                if retry_since is None:
                    retry_since = now
                elif now - retry_since >= RETRY_TIMEOUT:
                    logger.error("RAM++ giving up after retrying for %ds", RETRY_TIMEOUT)
                    raise
                logger.error("RAM++ inference failed, retrying:\n%s", traceback.format_exc())
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
