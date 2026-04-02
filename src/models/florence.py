import logging
import re
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from PIL import Image

from config import DEVICE, MODELS_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class FlorenceResult:
    od_tags:     list[str] = field(default_factory=list)
    description: str = ""
    ocr_raw:     str = ""


class Florence2Model:
    """
    Singleton for Florence-2. NOT thread-safe — all inference is serialised
    through a dedicated single-thread executor.
    """

    def __init__(self, model_id: str, enabled: bool = True) -> None:
        self._enabled   = enabled
        self._processor = None
        self._model     = None
        self._executor  = ThreadPoolExecutor(max_workers=1, thread_name_prefix="florence")
        if enabled and MODELS_AVAILABLE:
            self._load(model_id)

    def _load(self, model_id: str) -> None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        logger.info("Loading Florence-2 model (%s) on %s ...", model_id, DEVICE)
        self._processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self._model = Florence2ForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype,
        ).to(DEVICE)
        self._model.eval()
        logger.info("Florence-2 model loaded.")

    def is_ready(self) -> bool:
        return self._model is not None and self._processor is not None

    def is_enabled(self) -> bool:
        return self._enabled

    def analyse(self, image: Image.Image) -> "Future[FlorenceResult]":
        """
        Submit OD + caption + OCR to the private single-thread executor.
        Returns a Future immediately; caller blocks on .result() when needed.
        """
        if not self.is_ready():
            f: Future[FlorenceResult] = Future()
            f.set_result(FlorenceResult())
            return f
        return self._executor.submit(self._run_all, image)

    def _run_all(self, image: Image.Image) -> FlorenceResult:
        return FlorenceResult(
            od_tags=self._od(image),
            description=self._caption(image),
            ocr_raw=self._ocr(image),
        )

    def _generate(self, task: str, image: Image.Image, *, max_new_tokens: int = 1024, num_beams: int = 3) -> str:
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        model_dtype = next(self._model.parameters()).dtype
        inputs = {
            k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                max_length=None,
                num_beams=num_beams,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height),
        )
        raw = parsed.get(task, "")
        cleaned = re.sub(r"<[^>]*>", "", raw)
        return re.sub(r"[<>]", "", cleaned).strip()

    def _od(self, image: Image.Image) -> list[str]:
        logger.debug("Florence OD started")
        try:
            inputs = self._processor(text="<OD>", images=image, return_tensors="pt")
            model_dtype = next(self._model.parameters()).dtype
            inputs = {
                k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
                for k, v in inputs.items()
            }
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                generated_text, task="<OD>", image_size=(image.width, image.height),
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

    def _caption(self, image: Image.Image) -> str:
        logger.debug("Florence CAP started")
        try:
            raw = re.sub(r"\s+", " ", self._generate("<MORE_DETAILED_CAPTION>", image)).strip()
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

    def _ocr(self, image: Image.Image) -> str:
        logger.debug("Florence OCR started")
        try:
            raw = self._generate("<OCR>", image, max_new_tokens=256, num_beams=3)
            logger.debug("Florence OCR complete")
            text = re.sub(r"\s+", " ", raw.encode("ascii", errors="ignore").decode()).strip()
            return text if re.search(r"[a-zA-Z0-9]{2}", text) else ""
        except Exception:
            logger.error("Florence-2 <OCR> failed:\n%s", traceback.format_exc())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            return ""

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._model is None:
            return "not loaded"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return DEVICE
