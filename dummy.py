
import umap
import plotly.express as px
import pandas as pd
from typing import Optional
import numpy as np
from loguru import logger
from gensim.models import Word2Vec
import numpy as np
from loguru import logger
from centralized_nlp_package.text_processing.text_preprocessing import tokenize_text
from centralized_nlp_package.text_processing.ngram_utils import find_ngrams
from centralized_nlp_package.embedding.embedding_utils import average_token_embeddings
from centralized_nlp_package.text_processing.ngram_utils import find_ngrams



def process_ngrams_tokens(x, model):
    """
    Tokenizes the input text and processes bigrams that exist in the model's vocabulary.

    Args:
        x (str): Input text.
        model: Word2Vec model with vocabulary 'wv'.

    Returns:
        list: Processed list of unigrams and bigrams.
    """
    unigrams = tokenize_text(x)
    bigrams = list(generate_ngrams(unigrams, 2))
    prev_removed = False

    if bigrams:
        bigram_joined = '_'.join(bigrams[0])
        if bigram_joined in model.wv:
            unigrams.remove(bigrams[0][0])
            unigrams.remove(bigrams[0][1])
            unigrams.append(bigram_joined)
            prev_removed = True

    for bigram in bigrams[1:]:
        bigram_joined = '_'.join(bigram)
        if bigram_joined in model.wv:
            unigrams.remove(bigram[1])
            unigrams.append(bigram_joined)
            if not prev_removed:
                unigrams.remove(bigram[0])
                prev_removed = True
        else:
            prev_removed = False

    return unigrams

# topic modelling
def compute_text_embedding(x, model):
    """
    Computes the embedding of the input text by averaging the embeddings of its unigrams and bigrams.

    Args:
        x (str): Input text.
        model: Word2Vec model with vocabulary 'wv'.

    Returns:
        numpy.ndarray or None: The embedding vector, or None if not found.
    """
    if '_' in x:
        try:
            return model.wv[x]
        except KeyError:
            pass  # Continue processing if the word is not in the vocabulary

    unigrams = process_ngrams_tokens(x, model)
    embeddings = [model.wv[phrase] for phrase in unigrams if phrase in model.wv]

    if embeddings:
        return np.mean(np.stack(embeddings), axis=0)
    else:
        try:
            return model.wv[x]
        except KeyError:
            return None
        

