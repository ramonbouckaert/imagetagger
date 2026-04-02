import logging
import re
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from threading import Event

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
    Manages a pool of Florence-2 instances. Jobs are queued and dispatched to
    the next available instance, so up to num_instances tasks run in parallel.

    Each instance is NOT thread-safe internally; the pool ensures only one job
    runs on each instance at a time.
    """

    def __init__(self, model_id: str, enabled: bool = True, num_instances: int = 2) -> None:
        self._enabled  = enabled
        self._instances: list[tuple] = []   # (model, processor) — for introspection
        self._pool     = Queue()
        # Allow more queued jobs than instances so callers are never rejected.
        self._executor = ThreadPoolExecutor(max_workers=num_instances * 4, thread_name_prefix="florence")
        if enabled and MODELS_AVAILABLE:
            for i in range(num_instances):
                pair = self._load_instance(model_id, i)
                if pair is not None:
                    self._instances.append(pair)
                    self._pool.put(pair)

    def _load_instance(self, model_id: str, index: int) -> tuple | None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        try:
            logger.info("Loading Florence-2 instance %d (%s) on %s ...", index, model_id, DEVICE)
            processor = AutoProcessor.from_pretrained(model_id)
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            model = Florence2ForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype,
            ).to(DEVICE)
            model.eval()
            logger.info("Florence-2 instance %d loaded.", index)
            return (model, processor)
        except Exception:
            logger.error("Florence-2 instance %d failed to load:\n%s", index, traceback.format_exc())
            return None

    def is_ready(self) -> bool:
        return len(self._instances) > 0

    def is_enabled(self) -> bool:
        return self._enabled

    def analyse(self, image: Image.Image, cancel: Event | None = None) -> "Future[FlorenceResult]":
        """
        Submit OD + caption + OCR to the executor. The job will block until a
        model instance is free, then run all three tasks on that instance.
        Returns a Future immediately. Set cancel to abort retrying on failure.
        """
        if not self.is_ready():
            f: Future[FlorenceResult] = Future()
            f.set_result(FlorenceResult())
            return f
        return self._executor.submit(self._run_all, image, cancel)

    def _run_all(self, image: Image.Image, cancel: Event | None) -> FlorenceResult:
        model, processor = self._pool.get()
        try:
            while True:
                try:
                    return FlorenceResult(
                        od_tags=self._od(model, processor, image),
                        description=self._caption(model, processor, image),
                        ocr_raw=self._ocr(model, processor, image),
                    )
                except Exception:
                    if cancel and cancel.is_set():
                        raise
                    logger.error("Florence-2 inference failed, retrying:\n%s", traceback.format_exc())
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    time.sleep(1)
        finally:
            self._pool.put((model, processor))

    def _generate(
        self,
        model,
        processor,
        task: str,
        image: Image.Image,
        *,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
    ) -> str:
        inputs = processor(text=task, images=image, return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        inputs = {
            k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                max_length=None,
                num_beams=num_beams,
                do_sample=False,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height),
        )
        raw = parsed.get(task, "")
        cleaned = re.sub(r"<[^>]*>", "", raw)
        return re.sub(r"[<>]", "", cleaned).strip()

    def _od(self, model, processor, image: Image.Image) -> list[str]:
        logger.debug("Florence OD started")
        inputs = processor(text="<OD>", images=image, return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        inputs = {
            k: v.to(DEVICE, dtype=model_dtype) if v.is_floating_point() else v.to(DEVICE)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
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

    def _caption(self, model, processor, image: Image.Image) -> str:
        logger.debug("Florence CAP started")
        raw = re.sub(r"\s+", " ", self._generate(model, processor, "<MORE_DETAILED_CAPTION>", image)).strip()
        logger.debug("Florence CAP complete")
        return re.sub(
            r"^The image \w+\s+(.)",
            lambda m: m.group(1).upper(),
            raw,
        )

    def _ocr(self, model, processor, image: Image.Image) -> str:
        logger.debug("Florence OCR started")
        raw = self._generate(model, processor, "<OCR>", image, max_new_tokens=256, num_beams=3)
        logger.debug("Florence OCR complete")
        text = re.sub(r"\s+", " ", raw.encode("ascii", errors="ignore").decode()).strip()
        return text if re.search(r"[a-zA-Z0-9]{2}", text) else ""

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        if not self._instances:
            return "not loaded"
        try:
            return str(next(self._instances[0][0].parameters()).device)
        except StopIteration:
            return DEVICE
