# centralized_nlp_package/preprocessing/ngram_utils.py

from typing import List, Tuple, Iterator
from gensim.models import Word2Vec, Phrases

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

def get_model_ngrams(text: List[str], model: Word2Vec) -> List[str]:
    """
    Replaces tokens in the text with their corresponding n-grams if they exist in the Word2Vec model.

    Args:
        text (List[str]): List of tokens from the text.
        model (Word2Vec): Trained Word2Vec model.

    Returns:
        List[str]: List of tokens with possible n-gram replacements.
    """
    bigrams = find_ngrams(text, 2)
    replaced_tokens = []
    for bigram in bigrams:
        joined = '_'.join(bigram)
        if joined in model.wv:
            replaced_tokens.append(joined)
        else:
            replaced_tokens.append(bigram[0])
    # Add the last token if no bigram was formed
    if len(text) > 1:
        replaced_tokens.append(text[-1])
    return replaced_tokens
