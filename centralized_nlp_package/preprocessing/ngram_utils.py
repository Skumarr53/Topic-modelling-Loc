# centralized_nlp_package/preprocessing/ngram_utils.py

import numpy as np
from typing import List, Tuple, Iterator
from gensim.models import Word2Vec, Phrases
from centralized_nlp_package.preprocessing.text_preprocessing import tokenize_text

def find_ngrams(input_list: List[str], n: int) -> Iterator[Tuple[str, ...]]:
    """
    Generates n-grams from a list of tokens.

    Args:
        input_list (List[str]): List of tokens.
        n (int): The number of tokens in each n-gram.

    Yields:
        Iterator[Tuple[str, ...]]: An iterator over n-grams as tuples.
    """
    return zip(*[input_list[i:] for i in range(n)])


##  topic modelling
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
    bigrams = list(find_ngrams(unigrams, 2))
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
