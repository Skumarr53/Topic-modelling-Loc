# centralized_nlp_package/tests/test_preprocessing.py

import pytest
from centralized_nlp_package.preprocessing.text_processing import initialize_spacy_model, word_tokenize
from centralized_nlp_package.utils.config import Config, PreprocessingConfig
import spacy

@pytest.fixture
def mock_config():
    return Config(
        snowflake=None,
        dask=None,
        word2vec=None,
        preprocessing=PreprocessingConfig(
            spacy_model="en_core_web_sm",
            additional_stop_words=["bottom", "top", "call"],
            max_length=1000000000
        )
    )

def test_initialize_spacy_model(mock_config):
    nlp = initialize_spacy_model(mock_config)
    assert isinstance(nlp, spacy.Language)
    assert nlp.max_length == mock_config.preprocessing.max_length

def test_word_tokenize(mock_config):
    nlp = initialize_spacy_model(mock_config)
    doc = "The bottom line is that top performance requires a constant call."
    tokens = word_tokenize(doc, nlp)
    expected_tokens = ["bottom", "line", "performance", "require", "constant", "call"]
    # Depending on lemmatization and stop word removal, adjust expected tokens
    assert isinstance(tokens, list)
    assert len(tokens) == len(expected_tokens)
    for token in expected_tokens:
        assert token in tokens
