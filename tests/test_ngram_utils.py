# centralized_nlp_package/tests/test_ngram_utils.py

import pytest
from centralized_nlp_package.preprocessing.ngram_utils import find_ngrams, get_model_ngrams
from gensim.models import Word2Vec
import numpy as np

def test_find_ngrams():
    input_list = ['this', 'is', 'a', 'test']
    n = 2
    expected = [('this', 'is'), ('is', 'a'), ('a', 'test')]
    result = list(find_ngrams(input_list, n))
    assert result == expected

def test_get_model_ngrams():
    # Create a simple Word2Vec model
    feed = [
        ['this', 'is', 'a', 'test'],
        ['this', 'test', 'is', 'word2vec'],
        ['word2vec', 'models', 'are', 'useful']
    ]
    model = Word2Vec(sentences=feed, vector_size=10, window=2, min_count=1, workers=1, epochs=1)
    model.train(feed, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Add bigrams to the model's vocabulary
    bigrams = ['this_is', 'is_a', 'a_test', 'this_test', 'test_is', 'is_word2vec', 'word2vec_models', 'models_are', 'are_useful']
    for bigram in bigrams:
        model.wv.add_vector(bigram, np.random.rand(10))
    
    text = ['this', 'is', 'a', 'test']
    expected = ['this_is', 'is_a', 'a_test']
    result = get_model_ngrams(text, model)
    assert result == expected
