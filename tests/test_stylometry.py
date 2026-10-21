from src.stylometry import (
    character_ngram_counts,
    extract_features,
    function_word_frequencies,
    pos_tag_distribution,
    punctuation_statistics,
    word_frequencies,
    word_ngram_counts,
)


def test_statistics_and_average_sentence_length():
    features = extract_features("One short sentence. Here are four words!")
    assert features["word_count"] == 7
    assert features["sentence_count"] == 2
    assert features["average_sentence_length"] == 3.5
    assert features["unique_word_count"] == 7
    assert features["type_token_ratio"] == 1.0
    assert features["root_ttr"] == 7 / (7 ** 0.5)
    assert features["log_ttr"] == 1.0


def test_empty_and_punctuation_only_inputs_have_zero_vocabulary():
    for text in ("", " \n ", "...?!"):
        features = extract_features(text)
        assert features["word_count"] == 0
        assert features["sentence_count"] == 0
        assert features["average_sentence_length"] == 0
        assert features["type_token_ratio"] == 0


def test_punctuation_counts_and_rates():
    stats = punctuation_statistics("Hi, hi!", word_count=2)
    assert stats[","]["count"] == 1
    assert stats["!"]["per_100_words"] == 50
    assert stats["?"]["count"] == 0


def test_unicode_linebreaks_and_ttr():
    features = extract_features("Café repeats. Café continues.\n\n東京 waits.")
    assert features["word_count"] == 6
    assert features["unique_word_count"] == 5
    assert features["type_token_ratio"] == 5 / 6
    assert features["paragraph_count"] == 2


def test_word_frequency_filters_common_stop_words():
    assert word_frequencies("the cat and the dog cat") == [("cat", 2), ("dog", 1)]


def test_function_word_count_and_rate_are_case_insensitive():
    result = function_word_frequencies("THE the The, I i and AND wrote")
    assert result["articles"]["the"]["count"] == 3
    assert result["articles"]["the"]["per_1000_tokens"] == 375
    assert result["pronouns"]["i"]["count"] == 2
    assert result["conjunctions"]["and"]["per_1000_tokens"] == 250


def test_function_word_list_is_configurable():
    result = function_word_frequencies("Hello hello world", {"custom": ("hello",)})
    assert result == {"custom": {"hello": {"count": 2, "per_1000_tokens": 2000 / 3}}}


def test_word_unigram_bigram_and_trigram_counts():
    result = word_ngram_counts("A cat cat", (1, 3))
    assert result[1] == [(('cat',), 2), (('a',), 1)]
    assert set(result[2]) == {(('a', 'cat'), 1), (('cat', 'cat'), 1)}
    assert result[3] == [(('a', 'cat', 'cat'), 1)]


def test_character_ngram_whitespace_and_short_input():
    result = character_ngram_counts("A   b", (2, 3))
    assert result[2] == [("a ", 1), (" b", 1)]
    assert character_ngram_counts("x", (3, 5)) == {3: [], 4: [], 5: []}


def test_hapax_dislegomena_and_distributions():
    features = extract_features("a a b b c c c cat. c.")
    assert features["hapax_count"] == 1
    assert features["dislegomena_count"] == 2
    assert features["word_length_distribution"][3]["count"] == 1
    assert len(features["sentence_length_distribution"]) == 2
    assert features["sentence_length_stddev"] > 0
    assert "short sample" in features["sample_reliability"]


def test_pos_distribution_is_normalized_with_injected_tagger():
    observed = []

    def tagger(words):
        observed.extend(words)
        return [(words[0], "NNP"), (words[1], "VBP"), (words[2], "RB")]

    result = pos_tag_distribution("Cats run quickly", tagger)
    assert result["available"] is True
    assert result["tagged_token_count"] == 3
    assert observed == ["Cats", "run", "quickly"]
    assert result["distribution"]["NNP"]["frequency"] == 1 / 3
    assert result["distribution"]["NNP"]["description"] == "singular proper noun"


def test_pos_resource_failure_returns_clear_warning():
    def missing_model(_words):
        raise LookupError("resource missing")

    result = pos_tag_distribution("some text", missing_model)
    assert result["available"] is False
    assert result["distribution"] == {}
    assert "averaged-perceptron tagger data is missing" in result["warning"]


def test_missing_nltk_package_is_reported_without_fake_tags(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_nltk(name, *args, **kwargs):
        if name == "nltk":
            raise ImportError("simulated missing NLTK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_nltk)
    result = pos_tag_distribution("words here")
    assert result["available"] is False
    assert result["tagged_token_count"] == 0
    assert result["distribution"] == {}
    assert "Install nltk" in result["warning"]
