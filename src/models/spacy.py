import logging
import traceback

logger = logging.getLogger(__name__)

_CAPTION_BLOCKLIST = {
    "that", "they", "another", "foreground", "background", "left", "right", "top", "bottom", "something", "you",
    "overall", "which", "type", "them", "image", "him", "her", "he", "she", "this", "anything", "side", "who",
    "themself", "themselves", "other", "others", "atmosphere", "mood", "scene", "setting"
}
_OCR_BLOCKLIST = {
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
                if t and not _CAPTION_BLOCKLIST.intersection(t.split()):
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
                    if word not in _CAPTION_BLOCKLIST and word not in seen:
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
                    if word not in _OCR_BLOCKLIST and word not in seen:
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
