from src.comparison import build_document_representations, compare_features, lexical_similarity
from src.stylometry import extract_features


def test_feature_comparison_absolute_and_relative_difference():
    result = compare_features(extract_features("one two."), extract_features("one two three four."))
    assert result["word_count"]["document_a"] == 2
    assert result["word_count"]["document_b"] == 4
    assert result["word_count"]["absolute_difference"] == 2
    assert result["word_count"]["signed_difference"] == 2
    assert result["word_count"]["relative_difference_percent"] == 100


def test_relative_difference_is_undefined_when_baseline_is_zero():
    result = compare_features(extract_features(""), extract_features("word"))
    assert result["word_count"]["relative_difference_percent"] is None


def test_similarity_short_text_warns_without_score():
    result = lexical_similarity("cat sat.", "cat slept.")
    assert result["score"] is None
    assert result["reliable"] is False
    assert "at least" in result["warning"]


def test_similarity_sufficient_text_returns_bounded_score():
    text_a = "the same words describe a quiet garden with flowers and trees. " * 5
    text_b = "the same words describe a quiet garden with flowers and trees. " * 4 + "birds sing nearby. " * 2
    result = lexical_similarity(text_a, text_b)
    assert result["reliable"] is True
    assert 0 <= result["score"] <= 1
    assert result["warning"] is None


def test_similarity_unvectorizable_single_term_warns():
    result = lexical_similarity("echo " * 25, "echo " * 30)
    assert result["score"] is None
    assert result["reliable"] is False
    assert "distinct vocabulary" in result["warning"]


def test_pair_representations_include_word_character_style_and_combined():
    text_a = "A careful writer places short phrases together, then varies every sentence with gentle rhythm. " * 3
    text_b = "A careful writer joins short phrases in another order, then varies each sentence with a gentle cadence. " * 3
    results = build_document_representations(text_a, text_b, ngram_range=(2, 3), char_ngram_range=(3, 4), max_vocabulary=200)
    assert results["word_unigram_tfidf"]["available"]
    assert results["word_unigram_bigram_tfidf"]["available"]
    assert results["word_ngram_tfidf"]["available"]
    assert results["character_ngram_tfidf"]["available"]
    assert results["stylometric_numeric"]["available"]
    assert results["combined_experimental"]["available"]
    assert 0 <= results["combined_experimental"]["similarity"] <= 1
    assert "not an authorship probability" in results["combined_experimental"]["warning"]
    assert results["word_ngram_tfidf"]["scope"] == "vectorizer fit on this document pair only"


def test_pair_representations_warn_for_empty_and_short_text():
    result = build_document_representations("", "tiny", max_vocabulary=20)
    assert result["word_unigram_tfidf"]["available"] is False
    assert result["combined_experimental"]["available"] is False
    assert result["word_unigram_tfidf"]["warning"]


def test_pair_representation_rejects_invalid_configuration():
    import pytest

    with pytest.raises(ValueError, match="max_vocabulary"):
        build_document_representations("a", "b", max_vocabulary=0)
