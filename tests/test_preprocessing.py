from src.preprocessing import count_paragraphs, normalize_text, segment_sentences, tokenize_words


def test_word_tokenization_handles_contractions_unicode_and_case():
    assert tokenize_words("Don't stop, naïve reader!") == ["don't", "stop", "naïve", "reader"]


def test_tokenization_normalizes_decomposed_unicode():
    assert tokenize_words("cafe\u0301 café") == ["café", "café"]


def test_empty_and_whitespace_inputs():
    assert tokenize_words(" \n\t ") == []
    assert segment_sentences("") == []
    assert count_paragraphs(" \n\n ") == 0


def test_sentence_count_and_line_break_handling():
    text = "First line.\nSecond line!\nStill same paragraph?"
    assert len(segment_sentences(text)) == 3
    assert count_paragraphs(text) == 1


def test_normalization_keeps_punctuation_and_collapses_whitespace():
    assert normalize_text("A,\n\t B!") == "A, B!"


def test_paragraphs_count_nonempty_blocks():
    assert count_paragraphs("One.\n\n Two.\n\n\nThree.") == 3
