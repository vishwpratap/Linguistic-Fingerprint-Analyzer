"""Descriptive, length-aware-enough baseline stylometric feature extraction."""

from __future__ import annotations

import unicodedata
from collections import Counter
from math import log, sqrt
from statistics import pstdev
from typing import Any, Callable, Iterable, Mapping

from .preprocessing import count_paragraphs, normalize_text, segment_sentences, tokenize_words, tokenize_words_preserve_case

PUNCTUATION_MARKS = (".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "-", "—", "…")

# Small, transparent English stop-word set for the demonstration frequency chart.
# The lexical similarity calculation deliberately does not remove these terms.
STOP_WORDS = frozenset(
    "a an and are as at be been but by for from had has have he her hers him his "
    "i if in into is it its me my of on or our ours she so than that the their "
    "them then there these they this those to too us was we were what when where "
    "which who will with you your yours".split()
)

# This small, configurable list is illustrative rather than an exhaustive
# inventory of grammatical words or a complete author fingerprint.
FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "articles": ("a", "an", "the"),
    "pronouns": ("i", "you", "he", "she", "it", "we", "they"),
    "prepositions": ("in", "on", "at", "by", "for", "with", "from"),
    "conjunctions": ("and", "but", "or", "because", "although", "however"),
}

PENN_TAG_DESCRIPTIONS = {
    "CC": "coordinating conjunction", "CD": "cardinal number", "DT": "determiner",
    "EX": "existential there", "FW": "foreign word", "IN": "preposition or subordinating conjunction",
    "JJ": "adjective", "JJR": "comparative adjective", "JJS": "superlative adjective",
    "LS": "list item marker", "MD": "modal", "NN": "singular noun", "NNS": "plural noun",
    "NNP": "singular proper noun", "NNPS": "plural proper noun", "PDT": "predeterminer",
    "POS": "possessive ending", "PRP": "personal pronoun", "PRP$": "possessive pronoun",
    "RB": "adverb", "RBR": "comparative adverb", "RBS": "superlative adverb",
    "RP": "particle", "SYM": "symbol", "TO": "to", "UH": "interjection",
    "VB": "base-form verb", "VBD": "past-tense verb", "VBG": "gerund or present participle",
    "VBN": "past participle", "VBP": "non-third-person present verb", "VBZ": "third-person present verb",
    "WDT": "wh-determiner", "WP": "wh-pronoun", "WP$": "possessive wh-pronoun",
    "WRB": "wh-adverb", ".": "sentence-final punctuation", ",": "comma", ":": "colon or semicolon",
    "``": "opening quotation mark", "''": "closing quotation mark", "-LRB-": "left bracket", "-RRB-": "right bracket",
}


def punctuation_statistics(text: str, word_count: int | None = None) -> dict[str, dict[str, float | int]]:
    """Count common punctuation marks and their rate per 100 words.

    Punctuation is counted in the original text. Unicode punctuation glyphs
    outside the displayed common set remain in the overall text but not this
    focused comparison.
    """
    words = len(tokenize_words(text)) if word_count is None else word_count
    chars = len(text or "")
    sentences = len(segment_sentences(text or ""))
    output: dict[str, dict[str, float | int]] = {}
    for mark in PUNCTUATION_MARKS:
        count = (text or "").count(mark)
        output[mark] = {
            "count": count,
            "per_100_words": count / words * 100 if words else 0.0,
            "per_sentence": count / sentences if sentences else 0.0,
            "per_1000_characters": count / chars * 1000 if chars else 0.0,
        }
    return output


def extract_features(text: str) -> dict[str, Any]:
    """Extract consistent basic writing statistics from one document.

    Character count includes spaces and line breaks. Words are Unicode letter
    or number sequences with internal apostrophes, compared case-insensitively.
    TTR is 0 for an empty vocabulary and is sensitive to document length.
    Sentence length is total tokens divided by heuristic sentence count.
    """
    original = text or ""
    tokens = tokenize_words(original)
    sentences = segment_sentences(original)
    sentence_count = len(sentences)
    unique_count = len(set(tokens))
    word_count = len(tokens)
    token_lengths = [sum(1 for char in token if unicodedata.category(char)[0] in {"L", "N"}) for token in tokens]
    punctuation = punctuation_statistics(original, word_count)
    frequencies = Counter(tokens)
    lengths = Counter(token_lengths)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    hapax_count = sum(count == 1 for count in frequencies.values())
    dislegomena_count = sum(count == 2 for count in frequencies.values())
    result: dict[str, Any] = {
        "character_count": len(original),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": word_count / sentence_count if sentence_count else 0.0,
        "average_word_length": sum(token_lengths) / word_count if word_count else 0.0,
        "unique_word_count": unique_count,
        "type_token_ratio": unique_count / word_count if word_count else 0.0,
        "vocabulary_diversity": unique_count / word_count if word_count else 0.0,
        "root_ttr": unique_count / sqrt(word_count) if word_count else 0.0,
        "log_ttr": log(unique_count) / log(word_count) if word_count > 1 and unique_count else 0.0,
        "paragraph_count": count_paragraphs(original),
        "punctuation": punctuation,
        "hapax_count": hapax_count,
        "hapax_rate_per_1000": hapax_count / word_count * 1000 if word_count else 0.0,
        "dislegomena_count": dislegomena_count,
        "dislegomena_rate_per_1000": dislegomena_count / word_count * 1000 if word_count else 0.0,
        "word_length_distribution": {
            length: {"count": count, "per_1000_tokens": count / word_count * 1000 if word_count else 0.0}
            for length, count in sorted(lengths.items())
        },
        "sentence_length_distribution": sentence_lengths,
        "sentence_length_stddev": pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
        "sample_reliability": "short sample; interpret style statistics cautiously" if word_count < 50 else "descriptive sample",
    }
    return result


def word_frequencies(text: str, limit: int = 10, remove_stop_words: bool = True) -> list[tuple[str, int]]:
    """Return most frequent normalized words, optionally excluding STOP_WORDS."""
    if limit <= 0:
        return []
    words = tokenize_words(text)
    if remove_stop_words:
        words = [word for word in words if word not in STOP_WORDS]
    return Counter(words).most_common(limit)


def function_word_frequencies(
    text: str,
    function_words: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Count configurable grammatical words and normalize per 1,000 tokens."""
    words = tokenize_words(text)
    token_count = len(words)
    counts = Counter(words)
    result: dict[str, dict[str, dict[str, int | float]]] = {}
    for category, entries in (function_words or FUNCTION_WORDS).items():
        result[category] = {}
        for entry in entries:
            word = entry.casefold()
            count = counts[word]
            result[category][word] = {
                "count": count,
                "per_1000_tokens": count / token_count * 1000 if token_count else 0.0,
            }
    return result


def word_ngram_counts(
    text: str,
    ngram_range: tuple[int, int] = (1, 3),
    max_vocabulary: int = 1000,
    remove_stop_words: bool = False,
) -> dict[int, list[tuple[tuple[str, ...], int]]]:
    """Return top count-based word n-grams for each n in an inclusive range.

    Stop-word filtering is off by default. When enabled, filtered tokens are
    removed before windows are formed, so tokens separated in the source may
    become adjacent in the resulting phrase; interpret those phrases carefully.
    """
    start, end = ngram_range
    if start < 1 or end < start:
        raise ValueError("ngram_range must satisfy 1 <= minimum <= maximum")
    if max_vocabulary < 1:
        raise ValueError("max_vocabulary must be at least 1")
    tokens = tokenize_words(text)
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    results: dict[int, list[tuple[tuple[str, ...], int]]] = {}
    for n in range(start, end + 1):
        counts = Counter(tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1)))
        results[n] = counts.most_common(max_vocabulary)
    return results


def character_ngram_counts(
    text: str,
    ngram_range: tuple[int, int] = (3, 5),
    max_vocabulary: int = 1000,
) -> dict[int, list[tuple[str, int]]]:
    """Count overlapping character n-grams after NFC and whitespace collapse.

    Spaces and punctuation are retained as characters, so boundaries and
    orthographic patterns are inspectable; these patterns are not unique author
    identifiers. Short text naturally yields no grams for unavailable lengths.
    """
    start, end = ngram_range
    if start < 1 or end < start:
        raise ValueError("ngram_range must satisfy 1 <= minimum <= maximum")
    if max_vocabulary < 1:
        raise ValueError("max_vocabulary must be at least 1")
    normalized = normalize_text(text).casefold()
    results: dict[int, list[tuple[str, int]]] = {}
    for n in range(start, end + 1):
        counts = Counter(normalized[index:index + n] for index in range(max(0, len(normalized) - n + 1)))
        results[n] = counts.most_common(max_vocabulary)
    return results


def pos_tag_distribution(
    text: str,
    tagger: Callable[[list[str]], list[tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """Return normalized Penn Treebank POS frequencies or an explicit warning.

    NLTK's English averaged-perceptron tagger is used when available. No data is
    downloaded automatically; pass a tagger callable to test or replace it.
    """
    words = tokenize_words_preserve_case(text)
    if not words:
        return {"available": True, "tagged_token_count": 0, "distribution": {}, "warning": None}
    try:
        if tagger is None:
            import nltk

            tagged = nltk.pos_tag(words, lang="eng")
        else:
            tagged = tagger(words)
    except (ImportError, LookupError) as error:
        detail = next((line.strip() for line in str(error).splitlines() if line.strip()), type(error).__name__)
        return {
            "available": False,
            "tagged_token_count": 0,
            "distribution": {},
            "warning": "POS tagging is unavailable because NLTK or its English averaged-perceptron tagger data is missing. Install nltk and run `python -m nltk.downloader averaged_perceptron_tagger_eng` once. Details: " + detail,
        }
    counts = Counter(tag for _, tag in tagged)
    total = len(tagged)
    return {
        "available": True,
        "tagged_token_count": total,
        "distribution": {
            tag: {
                "count": count,
                "frequency": count / total if total else 0.0,
                "per_1000_tokens": count / total * 1000 if total else 0.0,
                "description": PENN_TAG_DESCRIPTIONS.get(tag, "Penn Treebank tag"),
            }
            for tag, count in sorted(counts.items())
        },
        "warning": None,
    }
