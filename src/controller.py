"""
AnalysisController — orchestrates image decode, model inference, and tag merging.
Knows nothing about Flask; accepts bytes, returns a plain dict.
"""

import io
import re
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from threading import Event

from PIL import Image

from config import MAX_IMAGE_EDGE, SIGLIP_TAG_THRESHOLD, RAM_TAG_THRESHOLD
from models import Florence2Model, SigLIPModel, RAMModel, OCRCorrectionModel, SpacyModel

logger = logging.getLogger(__name__)


_TYPO_CORRECTIONS = {
    "dinning table": "dining table",
    "napskin": "napkins",
    "napekin": "napkin",
}


def _open_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Empty body — 0 bytes received")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception as first_err:
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


def _normalise_tag(tag: str) -> list[str]:
    """Lowercase, strip leading articles, strip undescriptive text, strip some punctuation, break apart some tags, fix known typos. Returns 0–n tags."""
    tag = tag.lower().strip().replace(',', '').replace(';', '').replace('\u2026', '').replace('...', '')
    tag = ' '.join(w for w in tag.split() if any(c.isalnum() for c in w) and not w.startswith("'"))
    parts = tag.split(" or ")
    if len(parts) > 1:
        return [t for part in parts for t in _normalise_tag(part)]
    tag = tag.replace(" - ", "-")
    tokens = tag.split()
    for i, token in enumerate(tokens):
        if "/" in token:
            return [t for alt in token.split("/") for t in _normalise_tag(" ".join(tokens[:i] + [alt] + tokens[i+1:]))]
    tag = re.sub(r'^(only|just|possibly|probably|likely|specifically)\s+', '', tag)
    tag = re.sub(r'^(a|the)\s+', '', tag)
    tag = re.sub(r'^(one|same|more|few|fewer|less|several|various|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+', '', tag)
    tag = _TYPO_CORRECTIONS.get(tag, tag)
    if not tag or len(tag) < 3:
        return []
    if tag.startswith("human "):
        suffix = tag[len("human "):]
        return [tag, suffix] if len(suffix) >= 3 else [tag]
    return [tag]

def _extract_quoted_strings(text: str, max_len: int = 1000) -> list[str]:
    def ns(i: int, step: int):
        i += step
        while 0 <= i < len(text) and text[i].isspace():
            i += step
        return text[i] if 0 <= i < len(text) else None

    def escaped(i: int) -> bool:
        n = 0
        i -= 1
        while i >= 0 and text[i] == "\\":
            n += 1
            i -= 1
        return n % 2 == 1

    out, stack = [], []

    for i, ch in enumerate(text):
        if ch == "\u201c":
            stack.append(("\u201c", i + 1))
        elif ch == "\u201d":
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == "\u201c":
                    _, start = stack.pop(j)
                    q = text[start:i].strip()
                    if q and len(q) <= max_len and re.match(r"\w", q):
                        out.append(q)
                    break
        elif ch == '"' and not escaped(i):
            prev_raw = text[i - 1] if i > 0 else None
            next_raw = text[i + 1] if i + 1 < len(text) else None
            prev_ns, next_ns = ns(i, -1), ns(i, 1)

            openish = (
                prev_raw is None
                or prev_raw.isspace()
                or prev_ns in '([{-–—/:;\u201c'
            )
            closeish = (
                next_raw is None
                or next_raw.isspace()
                or next_ns in '.,!?;:)]}>\u201d'
            )

            if openish and not closeish:
                stack.append(('"', i + 1))
            elif closeish or any(q == '"' for q, _ in stack):
                for j in range(len(stack) - 1, -1, -1):
                    if stack[j][0] == '"':
                        _, start = stack.pop(j)
                        q = text[start:i].strip()
                        if q and len(q) <= max_len and re.match(r"\w", q):
                            out.append(q)
                        break
            else:
                stack.append(('"', i + 1))

    return out

class AnalysisController:
    """
    Orchestrates image analysis across all model singletons.

    Concurrency model:
      - _sem caps simultaneous inference requests (429 on overflow).
      - Florence and OCRCorrection own single-thread executors; their public
        methods return Futures so they can run in parallel with other models.
      - SigLIP, RAM, and spaCy are thread-safe; submitted to _pool.

    Concurrency phases per request:
      Phase 1 (parallel): Florence (OD+caption+OCR), SigLIP, RAM
      Phase 2: OCR correction starts immediately after phase 1;
               caption spaCy tasks also start immediately (overlap with OCR correction)
      Phase 3: spaCy on corrected OCR text (after OCR correction resolves)

    Requests queue naturally at each model's own executor; there is no
    request-level concurrency cap — callers are never turned away with 429.
    """

    def __init__(
        self,
        florence:       Florence2Model,
        siglip:         SigLIPModel,
        ram:            RAMModel,
        ocr_correction: OCRCorrectionModel,
        spacy:          SpacyModel,
        max_image_edge: int = MAX_IMAGE_EDGE,
    ) -> None:
        self._florence       = florence
        self._siglip         = siglip
        self._ram            = ram
        self._ocr_correction = ocr_correction
        self._spacy          = spacy
        self._max_edge       = max_image_edge
        # Shared pool for thread-safe models (SigLIP, RAM, spaCy).
        # Non-thread-safe models own their own single-thread executors.
        self._pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="inference")

    def not_ready(self) -> list[str]:
        """Return names of models that are enabled but not yet loaded."""
        return [
            name for name, model in [
                ("Florence-2",    self._florence),
                ("SigLIP",        self._siglip),
                ("RAM++",         self._ram),
                ("OCR correction",self._ocr_correction),
                ("spaCy",         self._spacy),
            ]
            if model.is_enabled() and not model.is_ready()
        ]

    def decode_image(self, data: bytes) -> Image.Image:
        """
        Decode bytes → PIL Image, convert to RGB, downscale if needed.
        Raises ValueError with a descriptive message on failure.
        """
        img = _open_image(data)
        img = img.convert("RGB")
        if max(img.size) > self._max_edge:
            img.thumbnail((self._max_edge, self._max_edge), Image.LANCZOS)
            logger.debug("Downscaled image to %s", img.size)
        return img

    def analyse(self, image: Image.Image, cancel: Event | None = None) -> dict:
        """
        Run all models and return the merged result dict.
        Queued requests wait indefinitely. Pass a cancel Event to interrupt
        retry loops (e.g. on client disconnect).
        """
        if cancel is None:
            cancel = Event()

        # ── Phase 1: Florence + SigLIP + RAM in parallel ──────────────────────
        florence_future = self._florence.analyse(image, cancel)
        siglip_future = (
            self._pool.submit(self._siglip.classify, image, SIGLIP_TAG_THRESHOLD, cancel)
            if self._siglip.is_ready() else None
        )
        ram_future = (
            self._pool.submit(self._ram.classify, image, RAM_TAG_THRESHOLD, cancel)
            if self._ram.is_ready() else None
        )

        phase1 = [f for f in (florence_future, siglip_future, ram_future) if f is not None]
        futures_wait(phase1)

        florence_result = florence_future.result()
        cap = florence_result.description
        cap_quotes = _extract_quoted_strings(cap)

        # ── Phase 2: OCR correction + caption spaCy (overlapping) ─────────────
        ocr_future = self._ocr_correction.correct(florence_result.ocr_raw, cancel)

        cap_chunks_future = (
            self._pool.submit(self._spacy.noun_chunk_tags, [cap])
            if cap and self._spacy.is_ready() else None
        )
        cap_nouns_future = (
            self._pool.submit(self._spacy.noun_tags, [cap])
            if cap and self._spacy.is_ready() else None
        )
        cap_quotes_future = (
            self._pool.submit(self._spacy.sentence_tags, cap_quotes)
            if cap_quotes and self._spacy.is_ready() else None
        )
        cap_quotes_words_future = (
            self._pool.submit(self._spacy.word_tags, cap_quotes)
            if cap_quotes and self._spacy.is_ready() else None
        )

        ocr_text = ocr_future.result()

        # ── Phase 3: spaCy on corrected OCR text ──────────────────────────────
        ocr_words_future = (
            self._pool.submit(self._spacy.word_tags, [ocr_text])
            if ocr_text and self._spacy.is_ready() else None
        )

        spacy_futures = [
            f for f in (cap_chunks_future, cap_nouns_future, cap_quotes_future, cap_quotes_words_future, ocr_words_future)
            if f is not None
        ]
        if spacy_futures:
            futures_wait(spacy_futures)

        # ── Merge ──────────────────────────────────────────────────────────────
        tags: list[str] = []
        seen: set[str] = set()

        def _add(raw: str) -> None:
            for norm in _normalise_tag(raw):
                if norm not in seen:
                    tags.append(norm)
                    seen.add(norm)

        for tag in florence_result.od_tags:
            _add(tag)
        for tag in (siglip_future.result() if siglip_future else []):
            _add(tag)
        for tag in (ram_future.result() if ram_future else []):
            _add(tag)
        for future in (cap_chunks_future, cap_nouns_future, cap_quotes_future, cap_quotes_words_future, ocr_words_future):
            if future:
                for tag in future.result():
                    _add(tag)

        return {
            "tags":        tags,
            "description": cap,
            "text":        ocr_text,
        }

    def device_report(self) -> dict[str, str]:
        return {
            "Florence-2":    self._florence.device_str(),
            "SigLIP":        self._siglip.device_str(),
            "RAM++":         self._ram.device_str(),
            "OCR correction":self._ocr_correction.device_str(),
            "spaCy":         self._spacy.device_str(),
        }
