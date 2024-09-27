# tests/preprocessing/test_text_preprocessing.py

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Tuple
import spacy
import numpy as np

# Import the functions to be tested
from centralized_nlp_package.preprocessing.text_preprocessing import (
    initialize_spacy,
    find_ngrams,
    tokenize_and_lemmatize_text,
    tokenize_matched_words
)

# Import the Config class if needed
from centralized_nlp_package.utils.config import Config


@pytest.fixture
def mock_config():
    """
    Fixture to mock the Config object.
    """
    with patch('centralized_nlp_package.preprocessing.text_preprocessing.Config') as MockConfig:
        instance = MockConfig()
        instance.preprocessing.preprocessing.spacy_model = "en_core_web_sm"
        instance.preprocessing.preprocessing.additional_stop_words = ["bottom", "top", "call"]
        instance.preprocessing.preprocessing.max_length = 1000000
        yield instance


@pytest.fixture
def mock_spacy_model():
    """
    Fixture to mock the SpaCy Language model.
    """
    with patch('centralized_nlp_package.preprocessing.text_preprocessing.spacy.load') as mock_load:
        mock_nlp = MagicMock(spec=spacy.Language)
        mock_nlp.Defaults.stop_words = set(["a", "the"])
        mock_load.return_value = mock_nlp
        yield mock_nlp


def test_initialize_spacy_success(mock_config, mock_spacy_model):
    """
    Test successful initialization of SpaCy model.
    """
    nlp = initialize_spacy()
    mock_spacy_model.__setattr__.assert_called_with('max_length', 1000000)
    assert nlp == mock_spacy_model
    mock_spacy_model.Defaults.stop_words == set(["a", "the"]) - set(["bottom", "top", "call"])


def test_initialize_spacy_model_loading_failure(mock_config):
    """
    Test initialization failure when SpaCy model cannot be loaded.
    """
    with patch('centralized_nlp_package.preprocessing.text_preprocessing.spacy.load', side_effect=OSError):
        with pytest.raises(OSError):
            initialize_spacy()


def test_find_ngrams_normal():
    """
    Test find_ngrams with a typical list and n.
    """
    input_list = ['this', 'is', 'a', 'test']
    n = 2
    expected_output = [('this', 'is'), ('is', 'a'), ('a', 'test')]
    assert find_ngrams(input_list, n) == expected_output


def test_find_ngrams_n_equals_one():
    """
    Test find_ngrams with n=1 (unigrams).
    """
    input_list = ['this', 'is', 'a', 'test']
    n = 1
    expected_output = [('this',), ('is',), ('a',), ('test',)]
    assert find_ngrams(input_list, n) == expected_output


def test_find_ngrams_n_greater_than_list():
    """
    Test find_ngrams with n greater than the length of the list.
    """
    input_list = ['this', 'is', 'a', 'test']
    n = 5
    expected_output = []
    assert find_ngrams(input_list, n) == expected_output


def test_find_ngrams_empty_list():
    """
    Test find_ngrams with an empty input list.
    """
    input_list = []
    n = 3
    expected_output = []
    assert find_ngrams(input_list, n) == expected_output


def test_tokenize_text_normal(mock_spacy_model):
    """
    Test tokenize_text with normal input.
    """
    doc = "This is a test sentence."
    # Mock the tokens returned by the spaCy model
    mock_token1 = MagicMock()
    mock_token1.lemma_ = "this"
    mock_token1.is_stop = False
    mock_token1.is_punct = False
    mock_token1.pos_ = 'PRON'

    mock_token2 = MagicMock()
    mock_token2.lemma_ = "be"
    mock_token2.is_stop = True
    mock_token2.is_punct = False
    mock_token2.pos_ = 'AUX'

    mock_token3 = MagicMock()
    mock_token3.lemma_ = "a"
    mock_token3.is_stop = True
    mock_token3.is_punct = False
    mock_token3.pos_ = 'DET'

    mock_token4 = MagicMock()
    mock_token4.lemma_ = "test"
    mock_token4.is_stop = False
    mock_token4.is_punct = False
    mock_token4.pos_ = 'NOUN'

    mock_token5 = MagicMock()
    mock_token5.lemma_ = "."
    mock_token5.is_stop = False
    mock_token5.is_punct = True
    mock_token5.pos_ = 'PUNCT'

    mock_spacy_model.__iter__.return_value = [mock_token1, mock_token2, mock_token3, mock_token4, mock_token5]

    expected_tokens = ['this', 'test']
    tokens = tokenize_text(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_text_with_stopwords_punctuations_numbers(mock_spacy_model):
    """
    Test tokenize_text with stopwords, punctuations, and numbers.
    """
    doc = "Hello, this is a 123 test!"
    
    # Mock tokens
    mock_token1 = MagicMock()
    mock_token1.lemma_ = "hello"
    mock_token1.is_stop = False
    mock_token1.is_punct = False
    mock_token1.pos_ = 'INTJ'

    mock_token2 = MagicMock()
    mock_token2.lemma_ = "this"
    mock_token2.is_stop = True
    mock_token2.is_punct = False
    mock_token2.pos_ = 'PRON'

    mock_token3 = MagicMock()
    mock_token3.lemma_ = "be"
    mock_token3.is_stop = True
    mock_token3.is_punct = False
    mock_token3.pos_ = 'AUX'

    mock_token4 = MagicMock()
    mock_token4.lemma_ = "a"
    mock_token4.is_stop = True
    mock_token4.is_punct = False
    mock_token4.pos_ = 'DET'

    mock_token5 = MagicMock()
    mock_token5.lemma_ = "123"
    mock_token5.is_stop = False
    mock_token5.is_punct = False
    mock_token5.pos_ = 'NUM'

    mock_token6 = MagicMock()
    mock_token6.lemma_ = "test"
    mock_token6.is_stop = False
    mock_token6.is_punct = False
    mock_token6.pos_ = 'NOUN'

    mock_token7 = MagicMock()
    mock_token7.lemma_ = "!"
    mock_token7.is_stop = False
    mock_token7.is_punct = True
    mock_token7.pos_ = 'PUNCT'

    mock_spacy_model.__iter__.return_value = [
        mock_token1, mock_token2, mock_token3, mock_token4,
        mock_token5, mock_token6, mock_token7
    ]

    expected_tokens = ['hello', 'test']
    tokens = tokenize_text(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_text_empty_string(mock_spacy_model):
    """
    Test tokenize_text with an empty string.
    """
    doc = ""
    mock_spacy_model.__iter__.return_value = []
    expected_tokens = []
    tokens = tokenize_text(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_text_all_stopwords(mock_spacy_model):
    """
    Test tokenize_text with a string containing only stopwords.
    """
    doc = "This is the a an and or but."
    
    # Mock tokens
    mock_token1 = MagicMock()
    mock_token1.lemma_ = "this"
    mock_token1.is_stop = True
    mock_token1.is_punct = False
    mock_token1.pos_ = 'PRON'

    mock_token2 = MagicMock()
    mock_token2.lemma_ = "be"
    mock_token2.is_stop = True
    mock_token2.is_punct = False
    mock_token2.pos_ = 'AUX'

    mock_token3 = MagicMock()
    mock_token3.lemma_ = "the"
    mock_token3.is_stop = True
    mock_token3.is_punct = False
    mock_token3.pos_ = 'DET'

    mock_token4 = MagicMock()
    mock_token4.lemma_ = "a"
    mock_token4.is_stop = True
    mock_token4.is_punct = False
    mock_token4.pos_ = 'DET'

    mock_token5 = MagicMock()
    mock_token5.lemma_ = "an"
    mock_token5.is_stop = True
    mock_token5.is_punct = False
    mock_token5.pos_ = 'DET'

    mock_token6 = MagicMock()
    mock_token6.lemma_ = "and"
    mock_token6.is_stop = True
    mock_token6.is_punct = False
    mock_token6.pos_ = 'CCONJ'

    mock_token7 = MagicMock()
    mock_token7.lemma_ = "or"
    mock_token7.is_stop = True
    mock_token7.is_punct = False
    mock_token7.pos_ = 'CCONJ'

    mock_token8 = MagicMock()
    mock_token8.lemma_ = "but"
    mock_token8.is_stop = True
    mock_token8.is_punct = False
    mock_token8.pos_ = 'CCONJ'

    mock_token9 = MagicMock()
    mock_token9.lemma_ = "."
    mock_token9.is_stop = False
    mock_token9.is_punct = True
    mock_token9.pos_ = 'PUNCT'

    mock_spacy_model.__iter__.return_value = [
        mock_token1, mock_token2, mock_token3, mock_token4,
        mock_token5, mock_token6, mock_token7, mock_token8, mock_token9
    ]

    expected_tokens = []
    tokens = tokenize_text(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_matched_words_normal(mock_spacy_model):
    """
    Test tokenize_matched_words with normal input.
    """
    doc = "John Doe went to New York."
    
    # Mock tokens
    mock_token1 = MagicMock()
    mock_token1.text = "John"
    mock_token1.pos_ = 'PROPN'
    mock_token1.is_stop = False
    mock_token1.is_punct = False

    mock_token2 = MagicMock()
    mock_token2.text = "Doe"
    mock_token2.pos_ = 'PROPN'
    mock_token2.is_stop = False
    mock_token2.is_punct = False

    mock_token3 = MagicMock()
    mock_token3.text = "went"
    mock_token3.pos_ = 'VERB'
    mock_token3.is_stop = False
    mock_token3.is_punct = False

    mock_token4 = MagicMock()
    mock_token4.text = "to"
    mock_token4.pos_ = 'ADP'
    mock_token4.is_stop = True
    mock_token4.is_punct = False

    mock_token5 = MagicMock()
    mock_token5.text = "New"
    mock_token5.pos_ = 'PROPN'
    mock_token5.is_stop = False
    mock_token5.is_punct = False

    mock_token6 = MagicMock()
    mock_token6.text = "York"
    mock_token6.pos_ = 'PROPN'
    mock_token6.is_stop = False
    mock_token6.is_punct = False

    mock_token7 = MagicMock()
    mock_token7.text = "."
    mock_token7.pos_ = 'PUNCT'
    mock_token7.is_stop = False
    mock_token7.is_punct = True

    mock_spacy_model.__iter__.return_value = [
        mock_token1, mock_token2, mock_token3, mock_token4,
        mock_token5, mock_token6, mock_token7
    ]

    expected_tokens = ['john', 'doe', 'went', 'new', 'york']
    tokens = tokenize_matched_words(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_matched_words_with_stopwords_and_punctuations(mock_spacy_model):
    """
    Test tokenize_matched_words with stopwords and punctuations.
    """
    doc = "Hello! This is an example."
    
    # Mock tokens
    mock_token1 = MagicMock()
    mock_token1.text = "Hello"
    mock_token1.pos_ = 'INTJ'
    mock_token1.is_stop = False
    mock_token1.is_punct = False

    mock_token2 = MagicMock()
    mock_token2.text = "!"
    mock_token2.pos_ = 'PUNCT'
    mock_token2.is_stop = False
    mock_token2.is_punct = True

    mock_token3 = MagicMock()
    mock_token3.text = "This"
    mock_token3.pos_ = 'PRON'
    mock_token3.is_stop = True
    mock_token3.is_punct = False

    mock_token4 = MagicMock()
    mock_token4.text = "is"
    mock_token4.pos_ = 'VERB'
    mock_token4.is_stop = True
    mock_token4.is_punct = False

    mock_token5 = MagicMock()
    mock_token5.text = "an"
    mock_token5.pos_ = 'DET'
    mock_token5.is_stop = True
    mock_token5.is_punct = False

    mock_token6 = MagicMock()
    mock_token6.text = "example"
    mock_token6.pos_ = 'NOUN'
    mock_token6.is_stop = False
    mock_token6.is_punct = False

    mock_token7 = MagicMock()
    mock_token7.text = "."
    mock_token7.pos_ = 'PUNCT'
    mock_token7.is_stop = False
    mock_token7.is_punct = True

    mock_spacy_model.__iter__.return_value = [
        mock_token1, mock_token2, mock_token3, mock_token4,
        mock_token5, mock_token6, mock_token7
    ]

    expected_tokens = ['hello', 'example']
    tokens = tokenize_matched_words(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_matched_words_empty_string(mock_spacy_model):
    """
    Test tokenize_matched_words with an empty string.
    """
    doc = ""
    mock_spacy_model.__iter__.return_value = []
    expected_tokens = []
    tokens = tokenize_matched_words(doc, mock_spacy_model)
    assert tokens == expected_tokens


def test_tokenize_matched_words_no_propn(mock_spacy_model):
    """
    Test tokenize_matched_words with no proper nouns.
    """
    doc = "the quick brown fox."
    
    # Mock tokens
    mock_token1 = MagicMock()
    mock_token1.text = "the"
    mock_token1.pos_ = 'DET'
    mock_token1.is_stop = True
    mock_token1.is_punct = False

    mock_token2 = MagicMock()
    mock_token2.text = "quick"
    mock_token2.pos_ = 'ADJ'
    mock_token2.is_stop = False
    mock_token2.is_punct = False

    mock_token3 = MagicMock()
    mock_token3.text = "brown"
    mock_token3.pos_ = 'ADJ'
    mock_token3.is_stop = False
    mock_token3.is_punct = False

    mock_token4 = MagicMock()
    mock_token4.text = "fox"
    mock_token4.pos_ = 'NOUN'
    mock_token4.is_stop = False
    mock_token4.is_punct = False

    mock_token5 = MagicMock()
    mock_token5.text = "."
    mock_token5.pos_ = 'PUNCT'
    mock_token5.is_stop = False
    mock_token5.is_punct = True

    mock_spacy_model.__iter__.return_value = [
        mock_token1, mock_token2, mock_token3, mock_token4, mock_token5
    ]

    expected_tokens = ['quick', 'brown', 'fox']
    tokens = tokenize_matched_words(doc, mock_spacy_model)
    assert tokens == expected_tokens
