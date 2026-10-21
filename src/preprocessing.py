"""Text preparation helpers that keep original punctuation available for analysis.

Tokenization uses a Unicode-aware regular expression, so it works offline and
does not require downloading NLTK data. A word may contain internal apostrophes;
hyphenated forms are treated as separate words. Sentence segmentation is a
lightweight punctuation heuristic, not a syntactic NLP model.
"""

from __future__ import annotations

import re
import unicodedata

WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[^.!?…]+[.!?…]+[\"'’”’)}\]]*|[^.!?…]+$", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without removing punctuation."""
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def tokenize_words(text: str) -> list[str]:
    """Return case-folded Unicode word tokens, retaining contractions."""
    return [word.casefold() for word in tokenize_words_preserve_case(text)]


def tokenize_words_preserve_case(text: str) -> list[str]:
    """Return NFC-normalized Unicode word tokens with source capitalization."""
    normalized = unicodedata.normalize("NFC", text or "")
    return [match.group(0) for match in WORD_PATTERN.finditer(normalized)]


def segment_sentences(text: str) -> list[str]:
    """Split on terminal punctuation and keep non-empty text fragments.

    Fragments containing only punctuation are excluded. Abbreviations and
    decimal numbers can be over-segmented; this transparent heuristic avoids
    external tokenizer downloads and is intended for descriptive statistics.
    """
    fragments = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(text or "")]
    return [fragment for fragment in fragments if tokenize_words(fragment)]


def count_paragraphs(text: str) -> int:
    """Count non-empty blocks separated by one or more blank lines."""
    return len([block for block in re.split(r"\n\s*\n+", text or "") if block.strip()])
