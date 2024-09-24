# centralized_nlp_package/tests/test_config.py
## execution : pytest centralized_nlp_package/tests/test_config.py


import pytest
from centralized_nlp_package.utils.config import get_config

def test_config_loading():
    config = get_config()
    assert config.snowflake.user != "", "Snowflake user should not be empty"
    assert config.dask.n_workers > 0, "Number of Dask workers should be positive"
    assert config.word2vec.vector_size > 0, "Word2Vec vector size should be positive"
    assert isinstance(config.preprocessing.additional_stop_words, list), "Stop words should be a list"
