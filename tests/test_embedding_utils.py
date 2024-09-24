# centralized_nlp_package/tests/test_embedding_utils.py

import pytest
from centralized_nlp_package.embedding.embedding_utils import embed_text, nearest_neighbors
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

def test_embed_text():
    # Create a simple Word2Vec model
    feed = [
        ['test', 'embedding', 'example'],
        ['another', 'test', 'embedding']
    ]
    model = Word2Vec(sentences=feed, vector_size=10, window=2, min_count=1, workers=1, epochs=1)
    text = ['test', 'embedding']
    embedding = embed_text(text, model)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (10,)
    
    # Test with no valid tokens
    text_invalid = ['invalid', 'tokens']
    embedding_invalid = embed_text(text_invalid, model)
    assert embedding_invalid is None

def test_nearest_neighbors():
    # Create a simple Word2Vec model
    feed = [
        ['topic1', 'match1'],
        ['topic1', 'match2'],
        ['topic2', 'match3']
    ]
    model = Word2Vec(sentences=feed, vector_size=10, window=2, min_count=1, workers=1, epochs=1)
    
    words = pd.DataFrame({
        'label': ['topic1', 'topic2'],
        'match': ['match1', 'match3']
    })
    
    # Add some similar words
    model.wv.add_vector('match1_neighbor', np.random.rand(10))
    model.wv.add_vector('match3_neighbor', np.random.rand(10))
    
    # Find nearest neighbors
    neighbors_df = nearest_neighbors(words, model, num_neigh=1, regularize=False)
    
    assert isinstance(neighbors_df, pd.DataFrame)
    assert len(neighbors_df) == 2
    assert 'label' in neighbors_df.columns
    assert 'embed' in neighbors_df.columns
    assert 'match' in neighbors_df.columns
    assert 'sim' in neighbors_df.columns
