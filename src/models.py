"""
Model singleton classes.

Each class loads its model in __init__ and exposes one public method.
Non-thread-safe models (Florence2Model, OCRCorrectionModel) own a private
single-thread executor that serialises inference calls; their public method
returns a concurrent.futures.Future so the caller can overlap it with other work.
Thread-safe models (SigLIPModel, RAMModel, SpacyModel) expose synchronous methods;
the caller submits them to a shared pool if concurrency is desired.
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field

import torch
from PIL import Image

from config import DEVICE, MODELS_AVAILABLE

logger = logging.getLogger(__name__)


# ── Florence-2 ─────────────────────────────────────────────────────────────────

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
        import re
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
        import re
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
        import re
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


# ── SigLIP ─────────────────────────────────────────────────────────────────────

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
        self._enabled       = enabled
        self._model         = None
        self._processor     = None
        self._text_inputs   = None
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

    def classify(self, image: Image.Image, threshold: float) -> list[str]:
        """Synchronous. Thread-safe — caller may submit to a pool for concurrency."""
        logger.debug("SigLIP classify started")
        if not self.is_ready():
            return []
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
            logger.error("SigLIP inference failed:\n%s", traceback.format_exc())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            return []

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._model is None:
            return "not loaded"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return DEVICE


# ── RAM++ ──────────────────────────────────────────────────────────────────────

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
            logger.warning("recognize-anything unavailable — RAM++ tags will be disabled.\nImport error: %s", e, exc_info=True)
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

    def classify(self, image: Image.Image, threshold: float) -> list[str]:
        """Synchronous. Thread-safe — caller may submit to a pool for concurrency."""
        logger.debug("RAM++ classify started")
        if not self.is_ready():
            return []
        try:
            from ram import inference_ram
            model_dtype = next(self._model.parameters()).dtype
            image_tensor = self._transform(image).unsqueeze(0).to(DEVICE, dtype=model_dtype)
            tags_str, _ = inference_ram(image_tensor, self._model)
            logger.debug("RAM++ classify complete")
            return [t.strip() for t in tags_str.split("|") if t.strip()]
        except Exception:
            logger.error("RAM++ inference failed:\n%s", traceback.format_exc())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            return []

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._model is None:
            return "not loaded"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return DEVICE


# ── OCR spell correction ───────────────────────────────────────────────────────

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


# ── spaCy noun chunks ──────────────────────────────────────────────────────────

_SPACY_CAPTION_BLOCKLIST = {
    "that", "they", "another", "foreground", "background", "left", "right", "top", "bottom", "something", "you",
    "overall", "which", "type", "them", "image", "him", "her", "he", "she", "this", "anything", "side", "who",
    "themself", "themselves", "other", "others", "atmosphere", "mood", "scene", "setting"
}
_SPACY_OCR_BLOCKLIST = {
    "that", "they", "another", "something", "you", "which", "them", "him", "her", "he", "she", "this",
    "anything", "who", "themself", "themselves", "other", "others"
}


class SpacyModel:
    """
    Singleton for spaCy. Thread-safe — the loaded nlp object is safe for
    concurrent inference calls.
    """

    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self._enabled = enabled
        self._nlp     = None
        if enabled:
            self._load(model_name)

    def _load(self, model_name: str) -> None:
        try:
            import spacy
        except ImportError:
            logger.warning(
                "spacy unavailable — noun chunk tags will be disabled. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            return
        logger.info("Loading spaCy model (%s) ...", model_name)
        self._nlp = spacy.load(model_name)
        logger.info("spaCy model loaded.")

    def is_ready(self) -> bool:
        return self._nlp is not None

    def is_enabled(self) -> bool:
        return self._enabled

    def noun_chunk_tags(self, text: str) -> list[str]:
        """Extract lowercased noun chunks from text, stripping determiners."""
        logger.debug("spaCy noun chunks started")
        if not self.is_ready() or not text:
            return []
        try:
            doc = self._nlp(text)
            tags: list[str] = []
            for chunk in doc.noun_chunks:
                t = " ".join(tok.text for tok in chunk if tok.dep_ not in ("det", "poss")).strip().lower()
                if t and not _SPACY_CAPTION_BLOCKLIST.intersection(t.split()):
                    tags.append(t)
            logger.debug("spaCy noun chunks complete: %s", tags)
            return tags
        except Exception:
            logger.error("spaCy noun chunk extraction failed:\n%s", traceback.format_exc())
            return []

    def noun_tags(self, text: str) -> list[str]:
        """Extract individual lowercased nouns and proper nouns from text."""
        logger.debug("spaCy noun extraction started")
        if not self.is_ready() or not text:
            return []
        try:
            doc = self._nlp(text)
            tags: list[str] = []
            seen: set[str] = set()
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    word = token.text.lower()
                    if word not in _SPACY_CAPTION_BLOCKLIST and word not in seen:
                        tags.append(word)
                        seen.add(word)
            logger.debug("spaCy noun extraction complete: %s", tags)
            return tags
        except Exception:
            logger.error("spaCy noun extraction failed:\n%s", traceback.format_exc())
            return []

    def word_tags(self, text: str) -> list[str]:
        """Extract lowercased nouns, proper nouns, adjectives, and verbs from text."""
        logger.debug("spaCy word extraction started")
        if not self.is_ready() or not text:
            return []
        try:
            doc = self._nlp(text)
            tags: list[str] = []
            seen: set[str] = set()
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN", "ADJ", "VERB"):
                    word = token.text.lower()
                    if word not in _SPACY_OCR_BLOCKLIST and word not in seen:
                        tags.append(word)
                        seen.add(word)
            logger.debug("spaCy word extraction complete: %s", tags)
            return tags
        except Exception:
            logger.error("spaCy word extraction failed:\n%s", traceback.format_exc())
            return []

    def device_str(self) -> str:
        if not self._enabled:
            return "disabled"
        return "cpu" if self._nlp is not None else "not loaded"
