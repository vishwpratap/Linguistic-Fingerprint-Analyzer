"""Feature comparison and preliminary lexical similarity."""

from __future__ import annotations

from typing import Any

from .preprocessing import normalize_text, tokenize_words
from .stylometry import extract_features, function_word_frequencies

FEATURE_LABELS = {
    "character_count": "Characters",
    "word_count": "Words",
    "sentence_count": "Sentences",
    "average_sentence_length": "Average sentence length (words)",
    "average_word_length": "Average word length (characters)",
    "unique_word_count": "Unique words",
    "type_token_ratio": "Type-token ratio",
    "root_ttr": "Root type-token ratio (Guiraud)",
    "log_ttr": "Log type-token ratio",
    "paragraph_count": "Paragraphs",
    "hapax_count": "Hapax legomena (once)",
    "hapax_rate_per_1000": "Hapax rate per 1,000 words",
    "dislegomena_count": "Hapax dislegomena (twice)",
    "dislegomena_rate_per_1000": "Dislegomena rate per 1,000 words",
    "sentence_length_stddev": "Sentence length standard deviation",
}


def compare_features(features_a: dict[str, Any], features_b: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Compare numeric top-level features; relative difference uses A as baseline."""
    comparison: dict[str, dict[str, float | None]] = {}
    for key in FEATURE_LABELS:
        value_a = float(features_a.get(key, 0))
        value_b = float(features_b.get(key, 0))
        difference = value_b - value_a
        comparison[key] = {
            "document_a": value_a,
            "document_b": value_b,
            "absolute_difference": abs(difference),
            "signed_difference": difference,
            "relative_difference_percent": difference / abs(value_a) * 100 if value_a else None,
        }
    return comparison


def lexical_similarity(text_a: str, text_b: str, minimum_words: int = 20) -> dict[str, Any]:
    """Return TF-IDF cosine similarity and a reliability note.

    Similarity is lexical overlap only, not a measure of authorship or writing
    style. Scikit-learn is imported lazily to keep feature extraction reusable.
    """
    words_a, words_b = tokenize_words(text_a), tokenize_words(text_b)
    base: dict[str, Any] = {"score": None, "reliable": False, "warning": None}
    if min(len(words_a), len(words_b)) < minimum_words:
        base["warning"] = f"Each document needs at least {minimum_words} words for a more stable lexical comparison."
        return base
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = TfidfVectorizer().fit_transform([text_a, text_b])
        if matrix.shape[1] < 2:
            base["warning"] = "The documents contain too little distinct vocabulary to vectorize reliably."
            return base
        score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
        base.update(score=score, reliable=True, warning=None)
        return base
    except ImportError:
        base["warning"] = "Scikit-learn is not installed; install project dependencies to calculate lexical similarity."
    except ValueError:
        base["warning"] = "The text does not contain enough usable terms for TF-IDF vectorization."
    return base


def _ranked_features(matrix: Any, names: Any, row: int, limit: int) -> list[dict[str, float | str]]:
    values = matrix[row].toarray().ravel()
    indices = sorted(range(len(values)), key=lambda index: (-values[index], str(names[index])))
    return [{"feature": str(names[index]), "weight": float(values[index])} for index in indices if values[index] > 0][:limit]


def _vectorized_result(vectorizer: Any, texts: list[str], limit: int) -> dict[str, Any]:
    """Fit a two-text vectorizer and return similarities plus inspectable terms."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out()
        if matrix.shape[1] == 0:
            raise ValueError("The selected representation produced no features.")
        row_norms = matrix.getnnz(axis=1)
        if not row_norms[0] or not row_norms[1]:
            raise ValueError("At least one document has no usable features for this representation.")
        score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
        top_a = _ranked_features(matrix, names, 0, limit)
        top_b = _ranked_features(matrix, names, 1, limit)
        scores_a = matrix[0].toarray().ravel()
        scores_b = matrix[1].toarray().ravel()
        delta = scores_a - scores_b
        distinctive_a = sorted(range(len(delta)), key=lambda index: (-delta[index], str(names[index])))
        distinctive_b = sorted(range(len(delta)), key=lambda index: (delta[index], str(names[index])))
        return {
            "available": True,
            "similarity": score,
            "warning": None,
            "feature_count": int(matrix.shape[1]),
            "top_a": top_a,
            "top_b": top_b,
            "distinctive_a": [{"feature": str(names[index]), "difference": float(delta[index])} for index in distinctive_a if delta[index] > 0][:limit],
            "distinctive_b": [{"feature": str(names[index]), "difference": float(-delta[index])} for index in distinctive_b if delta[index] < 0][:limit],
        }
    except ImportError:
        return {
            "available": False, "similarity": None,
            "warning": "Install scikit-learn to compute TF-IDF representations.",
            "feature_count": 0, "top_a": [], "top_b": [], "distinctive_a": [], "distinctive_b": [],
        }
    except ValueError as error:
        return {
            "available": False, "similarity": None, "warning": str(error),
            "feature_count": 0, "top_a": [], "top_b": [], "distinctive_a": [], "distinctive_b": [],
        }


def build_document_representations(
    text_a: str,
    text_b: str,
    *,
    features_a: dict[str, Any] | None = None,
    features_b: dict[str, Any] | None = None,
    ngram_range: tuple[int, int] = (1, 3),
    char_ngram_range: tuple[int, int] = (3, 5),
    max_vocabulary: int = 1000,
    remove_stop_words: bool = False,
    text_weight: float = 1.0,
    style_weight: float = 1.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build pair-fitted TF-IDF and numerical stylometric representations.

    All vectorizers are fit only on this supplied pair. Their similarities and
    feature rankings are local comparisons, not reusable trained models.
    Combined score is an experimental cosine on the concatenation of separately
    L2-normalized word-unigram TF-IDF and numerical style vectors, scaled by the
    exposed text/style weights. It is not a probability or authorship measure.
    """
    if max_vocabulary < 1:
        raise ValueError("max_vocabulary must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if text_weight < 0 or style_weight < 0 or (text_weight == 0 and style_weight == 0):
        raise ValueError("Weights must be nonnegative and at least one must be positive")
    for value_range in (ngram_range, char_ngram_range):
        if value_range[0] < 1 or value_range[1] < value_range[0]:
            raise ValueError("N-gram ranges must satisfy 1 <= minimum <= maximum")

    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [text_a or "", text_b or ""]

    def normalized_tokenizer(text: str) -> list[str]:
        tokens = tokenize_words(text)
        if remove_stop_words:
            from .stylometry import STOP_WORDS

            return [token for token in tokens if token not in STOP_WORDS]
        return tokens

    unigram = TfidfVectorizer(tokenizer=normalized_tokenizer, preprocessor=None, token_pattern=None, lowercase=False, max_features=max_vocabulary, ngram_range=(1, 1))
    word_12 = TfidfVectorizer(tokenizer=normalized_tokenizer, preprocessor=None, token_pattern=None, lowercase=False, max_features=max_vocabulary, ngram_range=(1, 2))
    word_configured = TfidfVectorizer(tokenizer=normalized_tokenizer, preprocessor=None, token_pattern=None, lowercase=False, max_features=max_vocabulary, ngram_range=ngram_range)
    char = TfidfVectorizer(analyzer="char", preprocessor=lambda text: normalize_text(text).casefold(), lowercase=False, max_features=max_vocabulary, ngram_range=char_ngram_range)

    representations = {
        "word_unigram_tfidf": _vectorized_result(unigram, texts, top_k),
        "word_unigram_bigram_tfidf": _vectorized_result(word_12, texts, top_k),
        "word_ngram_tfidf": _vectorized_result(word_configured, texts, top_k),
        "character_ngram_tfidf": _vectorized_result(char, texts, top_k),
    }
    if min(len(tokenize_words(text_a)), len(tokenize_words(text_b))) < 20:
        for result in representations.values():
            if result["available"]:
                result["warning"] = "Both documents have usable features, but at least one is under 20 words; interpret this local similarity cautiously."

    if features_a is None:
        features_a = extract_features(text_a)
    if features_b is None:
        features_b = extract_features(text_b)
    numeric_keys = ("average_sentence_length", "average_word_length", "type_token_ratio", "sentence_length_stddev", "hapax_rate_per_1000", "dislegomena_rate_per_1000")
    numeric_keys += tuple(f"function:{group}:{word}" for group, words in function_word_frequencies(text_a).items() for word in words)
    function_a, function_b = function_word_frequencies(text_a), function_word_frequencies(text_b)

    def feature_value(features: dict[str, Any], function_words: dict[str, Any], key: str) -> float:
        if key.startswith("function:"):
            _, group, word = key.split(":", 2)
            return float(function_words[group][word]["per_1000_tokens"])
        return float(features.get(key, 0.0))

    numeric_a = [feature_value(features_a, function_a, key) for key in numeric_keys]
    numeric_b = [feature_value(features_b, function_b, key) for key in numeric_keys]
    scales = [max(abs(value_a), abs(value_b), 1e-12) for value_a, value_b in zip(numeric_a, numeric_b)]
    values_a = [value / scale for value, scale in zip(numeric_a, scales)]
    values_b = [value / scale for value, scale in zip(numeric_b, scales)]
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    style_matrix = normalize(np.asarray([values_a, values_b], dtype=float), norm="l2")
    style_score = float(cosine_similarity(style_matrix[0:1], style_matrix[1:2])[0, 0]) if style_matrix[0].any() and style_matrix[1].any() else None
    representations["stylometric_numeric"] = {
        "available": style_score is not None,
        "similarity": style_score,
        "warning": None if style_score is not None else "At least one document has no nonzero numerical style features.",
        "feature_count": len(numeric_keys),
        "top_a": [{"feature": key, "value": value} for key, value in zip(numeric_keys, values_a)],
        "top_b": [{"feature": key, "value": value} for key, value in zip(numeric_keys, values_b)],
        "distinctive_a": [],
        "distinctive_b": [],
    }

    text_result = representations["word_unigram_tfidf"]
    if text_result["available"]:
        from scipy.sparse import csr_matrix, hstack
        from sklearn.preprocessing import normalize as normalize_sparse

        text_vectorizer = TfidfVectorizer(tokenizer=normalized_tokenizer, preprocessor=None, token_pattern=None, lowercase=False, max_features=max_vocabulary, ngram_range=(1, 1))
        text_matrix = text_vectorizer.fit_transform(texts)
        text_matrix = normalize_sparse(text_matrix, norm="l2")
        style_sparse = csr_matrix(style_matrix)
        combined_matrix = hstack((text_matrix * text_weight, style_sparse * style_weight), format="csr")
        combined_score = float(cosine_similarity(combined_matrix[0], combined_matrix[1])[0, 0])
        representations["combined_experimental"] = {
            "available": True,
            "similarity": combined_score,
            "warning": "Pair-local experimental score. Formula: cosine([text_weight × L2-normalized word-unigram TF-IDF, style_weight × L2-normalized numeric style vector]); not an authorship probability.",
            "feature_count": int(combined_matrix.shape[1]),
            "weights": {"text": text_weight, "style": style_weight},
            "top_a": [], "top_b": [], "distinctive_a": [], "distinctive_b": [],
        }
    else:
        representations["combined_experimental"] = {
            "available": False, "similarity": None,
            "warning": "Combined text-plus-style representation unavailable because word unigram TF-IDF is unavailable.",
            "feature_count": 0, "weights": {"text": text_weight, "style": style_weight},
            "top_a": [], "top_b": [], "distinctive_a": [], "distinctive_b": [],
        }

    for result in representations.values():
        result.setdefault("scope", "vectorizer fit on this document pair only")
    return representations


def compare_texts_with_fitted_pipeline(pipeline: Any, text_a: str, text_b: str, *, top_k: int = 20) -> dict[str, Any]:
    """Compare two texts after the exact fitted feature transform used by a model.

    This deliberately transforms both texts with the model's saved vocabulary,
    scaler, POS configuration, and normalization. Cosine is descriptive in
    that feature space and is not an authorship probability.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    import numpy as np

    transform = pipeline.named_steps["features"]
    matrix = transform.transform([text_a or "", text_b or ""])
    from sklearn.metrics.pairwise import cosine_similarity

    row_norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel() if hasattr(matrix, "multiply") else np.linalg.norm(matrix, axis=1)
    available = bool(row_norms[0] and row_norms[1])
    score = float(cosine_similarity(matrix[0], matrix[1])[0, 0]) if available else None
    try:
        names = np.asarray(transform.get_feature_names_out(), dtype=object)
        row_a = matrix[0].toarray().ravel() if hasattr(matrix[0], "toarray") else np.asarray(matrix[0]).ravel()
        row_b = matrix[1].toarray().ravel() if hasattr(matrix[1], "toarray") else np.asarray(matrix[1]).ravel()
        delta = row_a - row_b
        order = sorted(range(len(delta)), key=lambda index: (-abs(delta[index]), str(names[index])))
        differences = [
            {"feature": str(names[index]), "document_a": float(row_a[index]), "document_b": float(row_b[index]), "absolute_difference": float(abs(delta[index]))}
            for index in order if delta[index] != 0
        ][:top_k]
    except (AttributeError, ValueError):
        differences = []
    classifier = pipeline.named_steps.get("classifier")
    return {
        "available": available,
        "cosine_similarity": score,
        "feature_count": int(matrix.shape[1]),
        "feature_differences": differences,
        "prediction_a": classifier.predict(transform.transform([text_a or ""]))[0] if classifier is not None and (text_a or "").strip() else None,
        "prediction_b": classifier.predict(transform.transform([text_b or ""]))[0] if classifier is not None and (text_b or "").strip() else None,
        "warning": "Fitted-model feature-space cosine is a descriptive comparison, not a probability of common authorship. Closed-set predictions are forced among known labels.",
    }
