import logging
import traceback

logger = logging.getLogger(__name__)


_CAPTION_BLOCKLIST = {
    "that", "they", "another", "foreground", "background", "left", "right", "top", "bottom", "something", "you",
    "overall", "which", "type", "them", "image", "him", "her", "he", "she", "this", "anything", "side", "who",
    "themself", "themselves", "other", "others", "atmosphere", "mood", "scene", "setting", "some", "whom"
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

    def noun_chunk_tags(self, texts: list[str]) -> list[str]:
        """Extract lowercased noun chunks from texts, stripping determiners."""
        logger.debug("spaCy noun chunks started")
        if not self.is_ready() or not texts:
            return []
        try:
            tags: list[str] = []
            for doc in self._nlp.pipe(texts):
                for chunk in doc.noun_chunks:
                    t = " ".join(tok.text for tok in chunk if tok.dep_ not in ("det", "poss")).strip().lower()
                    if t and not _CAPTION_BLOCKLIST.intersection(t.split()):
                        tags.append(t)
            logger.debug("spaCy noun chunks complete: %s", tags)
            return tags
        except Exception:
            logger.error("spaCy noun chunk extraction failed:\n%s", traceback.format_exc())
            return []

    def noun_tags(self, texts: list[str]) -> list[str]:
        """Extract individual lowercased nouns and proper nouns from texts."""
        logger.debug("spaCy noun extraction started")
        if not self.is_ready() or not texts:
            return []
        try:
            tags: list[str] = []
            seen: set[str] = set()
            for doc in self._nlp.pipe(texts):
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

    _CLAUSE_DEPS = frozenset({'advcl', 'conj'})

    @staticmethod
    def _strip_connectives(clause):
        """Strip leading subordinating/coordinating conjunctions and trailing conjunctions from a clause span."""
        tokens = list(clause)
        while tokens and tokens[0].pos_ in ('CCONJ', 'SCONJ'):
            tokens = tokens[1:]
        while tokens and tokens[-1].pos_ in ('CCONJ',):
            tokens = tokens[:-1]
        if not tokens:
            return '', 0
        text = clause.doc[tokens[0].i:tokens[-1].i + 1].text
        return text, len(tokens)

    @staticmethod
    def _clause_spans(sent):
        """Split a spaCy sentence into clause-level spans using the dependency tree.
        Finds subordinate clause roots and splits at the start of each clause's subtree."""
        break_points = set()
        for token in sent:
            if token.dep_ in SpacyModel._CLAUSE_DEPS and token.pos_ in ('VERB', 'AUX'):
                start = min(t.i for t in token.subtree)
                if start > sent.start:
                    break_points.add(start)
            elif token.text == '(' and token.i > sent.start:
                break_points.add(token.i)
            elif token.text == ')' and token.i + 1 < sent.end:
                break_points.add(token.i + 1)
            elif token.text in ('\u2013', '\u2014') and token.i + 1 < sent.end:
                break_points.add(token.i + 1)
            elif token.text == ';' and token.i + 1 < sent.end:
                break_points.add(token.i + 1)
            elif token.text == ':' and token.i + 1 < sent.end:
                prev_tok = sent.doc[token.i - 1] if token.i > sent.start else None
                next_tok = sent.doc[token.i + 1]
                is_time = (
                    prev_tok is not None and prev_tok.text.isdigit()
                    and next_tok.text.isdigit()
                )
                if not is_time:
                    break_points.add(token.i + 1)

        boundaries = sorted([sent.start] + list(break_points) + [sent.end])
        for i in range(len(boundaries) - 1):
            yield sent.doc[boundaries[i]:boundaries[i + 1]]

    def sentence_tags(self, texts: list[str]) -> list[str]:
        """Split each text into sentences independently and return each as a lowercased tag.
        Sentences over 20 tokens are discarded."""
        logger.debug("spaCy sentence splitting started")
        if not self.is_ready() or not texts:
            return []
        try:
            tags: list[str] = []
            for doc in self._nlp.pipe(texts):
                for sent in doc.sents:
                    for clause in self._clause_spans(sent):
                        t, n = self._strip_connectives(clause)
                        t = t.strip().strip('.,!?;:\'"').strip()
                        if t and n <= 20:
                            tags.append(t.lower())
            logger.debug("spaCy sentence splitting complete: %s", tags)
            return tags
        except Exception:
            logger.error("spaCy sentence splitting failed:\n%s", traceback.format_exc())
            return []

    def word_tags(self, texts: list[str]) -> list[str]:
        """Extract lowercased nouns, proper nouns, adjectives, and verbs from texts."""
        logger.debug("spaCy word extraction started")
        if not self.is_ready() or not texts:
            return []
        try:
            tags: list[str] = []
            seen: set[str] = set()
            for doc in self._nlp.pipe(texts):
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
