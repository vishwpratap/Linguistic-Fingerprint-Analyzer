"""Leakage-safe, reusable feature pipelines for authorship experiments."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

from .preprocessing import normalize_text, tokenize_words, tokenize_words_preserve_case
from .stylometry import FUNCTION_WORDS, PENN_TAG_DESCRIPTIONS, STOP_WORDS, extract_features, function_word_frequencies, pos_tag_distribution

REPRESENTATIONS = {
    "stylometric": "Traditional stylometric features",
    "word_unigram": "Word unigram TF-IDF",
    "word_unigram_bigram": "Word unigram + bigram TF-IDF",
    "character": "Character n-gram TF-IDF",
    "hybrid_word": "Hybrid: stylometry + word n-grams",
    "hybrid_character": "Hybrid: stylometry + character n-grams",
}


def configured_word_tokenizer(text: str, *, preserve_case: bool = False, remove_stop_words: bool = False) -> list[str]:
    """Picklable tokenizer shared by training, persistence, and inference."""
    tokens = tokenize_words_preserve_case(text) if preserve_case else tokenize_words(text)
    if remove_stop_words:
        tokens = [word for word in tokens if word.casefold() not in STOP_WORDS]
    return tokens


def configured_character_preprocessor(text: str, *, preserve_case: bool = False) -> str:
    normalized = normalize_text(text)
    return normalized if preserve_case else normalized.casefold()


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert raw texts to deterministic, nonnegative, interpretable numbers.

    POS features are optional and are never synthesized when an NLTK resource
    is missing. Scaling is intentionally applied later inside each training
    pipeline so statistics are learned from training documents only.
    """

    def __init__(self, include_pos: bool = False):
        self.include_pos = include_pos

    def fit(self, X: Any, y: Any = None):
        self.feature_names_in_ = self._names()
        if self.include_pos:
            for text in list(X):
                result = pos_tag_distribution(str(text))
                if not result["available"]:
                    raise ValueError(result["warning"] or "English POS tagger is unavailable.")
        return self

    def transform(self, X: Any) -> np.ndarray:
        rows = []
        for raw in list(X):
            text = str(raw or "")
            features = extract_features(text)
            function = function_word_frequencies(text)
            row = [
                float(function[group][word]["per_1000_tokens"])
                for group, words in FUNCTION_WORDS.items() for word in words
            ]
            row.extend([
                float(features["average_sentence_length"]),
                float(features["average_word_length"]),
                float(features["type_token_ratio"]),
                float(features["root_ttr"]),
                float(features["log_ttr"]),
                float(features["hapax_rate_per_1000"]),
                float(features["dislegomena_rate_per_1000"]),
                float(features["sentence_length_stddev"]),
                float(sum(item["count"] for item in features["punctuation"].values()) * 1000 / max(features["word_count"], 1)),
            ])
            for length in range(1, 21):
                row.append(float(features["word_length_distribution"].get(length, {}).get("per_1000_tokens", 0.0)))
            row.append(float(sum(data["count"] for length, data in features["word_length_distribution"].items() if length > 20) * 1000 / max(features["word_count"], 1)))
            if self.include_pos:
                tagged = pos_tag_distribution(text)
                if not tagged["available"]:
                    raise ValueError(tagged["warning"] or "English POS tagger is unavailable.")
                row.extend(float(tagged["distribution"].get(tag, {}).get("frequency", 0.0)) for tag in self._pos_tags())
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self._names(), dtype=object)

    @staticmethod
    def _pos_tags() -> list[str]:
        return sorted(PENN_TAG_DESCRIPTIONS)

    def _names(self) -> list[str]:
        names = [f"function_word:{group}:{word}" for group, words in FUNCTION_WORDS.items() for word in words]
        names += [
            "average_sentence_length", "average_word_length", "type_token_ratio", "root_ttr", "log_ttr",
            "hapax_rate_per_1000", "dislegomena_rate_per_1000", "sentence_length_stddev", "punctuation_per_1000_words",
        ]
        names += [f"word_length_{length}" for length in range(1, 21)] + ["word_length_21_plus"]
        if self.include_pos:
            names += [f"pos:{tag}" for tag in self._pos_tags()]
        return names


def _style_branch(include_pos: bool) -> Pipeline:
    return Pipeline([
        ("extract", StylometricFeatureExtractor(include_pos=include_pos)),
        ("scale", MinMaxScaler(clip=True)),
        ("normalize", Normalizer(norm="l2")),
    ])


def make_feature_transformer(
    representation: str,
    *,
    max_vocabulary: int = 3000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    include_pos: bool = False,
    style_weight: float = 1.0,
    text_weight: float = 1.0,
    word_case_sensitive: bool = False,
    char_case_sensitive: bool = False,
    remove_stop_words: bool = False,
) -> Any:
    """Build one representation; vectorizers/scalers fit only when pipeline.fit is called."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    if max_vocabulary < 1:
        raise ValueError("max_vocabulary must be at least 1")
    if style_weight < 0 or text_weight < 0 or (style_weight == 0 and text_weight == 0):
        raise ValueError("Hybrid block weights must be nonnegative and at least one must be positive")
    word = TfidfVectorizer(
        tokenizer=partial(configured_word_tokenizer, preserve_case=word_case_sensitive, remove_stop_words=remove_stop_words), token_pattern=None, preprocessor=None, lowercase=False,
        max_features=max_vocabulary, ngram_range=word_ngram_range, sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char", preprocessor=partial(configured_character_preprocessor, preserve_case=char_case_sensitive), lowercase=False,
        max_features=max_vocabulary, ngram_range=char_ngram_range, sublinear_tf=True,
    )
    if representation == "stylometric":
        return _style_branch(include_pos)
    if representation == "word_unigram":
        return word.set_params(ngram_range=(1, 1))
    if representation == "word_unigram_bigram":
        return word.set_params(ngram_range=(1, 2))
    if representation == "character":
        return char
    text_block = word.set_params(ngram_range=word_ngram_range) if representation == "hybrid_word" else char
    return FeatureUnion(
        [("style", _style_branch(include_pos)), ("text", text_block)],
        transformer_weights={"style": style_weight, "text": text_weight},
    )

