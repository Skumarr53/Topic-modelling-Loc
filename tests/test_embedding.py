# centralized_nlp_package/tests/test_embedding.py

import pytest
from centralized_nlp_package.embedding.word2vec_model import train_word2vec, save_model
from gensim.models import Word2Vec
from pathlib import Path
import tempfile
import os

def test_train_word2vec():
    # Sample corpus
    feed = [
        ["this", "is", "a", "test"],
        ["this", "test", "is", "word2vec"],
        ["word2vec", "models", "are", "useful"]
    ]
    gen_bigram = True
    model_params = {
        'vector_size': 50,
        'window': 2,
        'min_count': 1,
        'workers': 1,
        'epochs': 5,
        'bigram_threshold': 1
    }
    model = train_word2vec(feed, gen_bigram, model_params)
    assert isinstance(model, Word2Vec)
    assert 'this_is' in model.wv.key_to_index
    assert 'word2vec_models' in model.wv.key_to_index

def test_save_model():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "word2vec.model")
        # Create a dummy model
        model = Word2Vec(vector_size=10, window=2, min_count=1, workers=1, epochs=1)
        model.build_vocab([["test", "model"]])
        model.train([["test", "model"]], total_examples=1, epochs=1)
        # Save the model
        save_model(model, model_path)
        # Check if the file exists
        assert Path(model_path).exists()
        # Load the model to verify
        loaded_model = Word2Vec.load(model_path)
        assert 'test' in loaded_model.wv.key_to_index
        assert 'model' in loaded_model.wv.key_to_index
