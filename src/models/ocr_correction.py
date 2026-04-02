import logging
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from config import DEVICE, MODELS_AVAILABLE

logger = logging.getLogger(__name__)


class OCRCorrectionModel:
    """
    Singleton for ByT5 OCR correction. NOT thread-safe — serialised through
    a dedicated single-thread executor.
    """

    def __init__(self, model_id: str, enabled: bool = True) -> None:
        self._enabled   = enabled
        self._model     = None
        self._tokenizer = None
        self._executor  = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-correction")
        if enabled and MODELS_AVAILABLE:
            self._load(model_id)

    def _load(self, model_id: str) -> None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        logger.info("Loading OCR correction model (%s) on %s ...", model_id, DEVICE)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            tie_word_embeddings=False,
            dtype=dtype,
        ).to(DEVICE)
        self._model.eval()
        logger.info("OCR correction model loaded.")

    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def is_enabled(self) -> bool:
        return self._enabled

    def correct(self, text: str) -> "Future[str]":
        """
        Submit OCR correction to the private serialising executor.
        Returns a Future so the controller can overlap it with spaCy work.
        If text is empty or model is not ready, returns a trivially-resolved Future.
        """
        if not text.strip() or not self.is_ready():
            f: Future[str] = Future()
            f.set_result(text)
            return f
        return self._executor.submit(self._correct_sync, text)

    def _correct_sync(self, text: str) -> str:
        logger.debug("OCR correction started")
        try:
            tok = self._tokenizer
            inputs = tok(text, return_tensors="pt", truncation=True).to(DEVICE)
            token_count = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=int(token_count * 1.1),
                    max_length=None,
                )
            corrected = tok.decode(output_ids[0], skip_special_tokens=True).strip()
            logger.debug("OCR correction complete")
            return corrected
        except Exception:
            logger.error("OCR correction failed:\n%s", traceback.format_exc())
            return text

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._model is None:
            return "not loaded"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return DEVICE
