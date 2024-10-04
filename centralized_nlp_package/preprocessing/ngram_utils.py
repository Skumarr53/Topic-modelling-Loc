# centralized_nlp_package/preprocessing/ngram_utils.py

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

def get_model_ngrams(text: List[str], model: Word2Vec) -> List[str]:
    """
    Replaces tokens in the text with their corresponding n-grams if they exist in the Word2Vec model.

    Args:
        text (List[str]): List of tokens from the text.
        model (Word2Vec): Trained Word2Vec model.

    Returns:
        List[str]: List of tokens with possible n-gram replacements.
    """
    unigrams = tokenize_text(x)
    vocab = {word: 0 for word in set(unigrams)}
    bigrams = [g for g in find_ngrams(unigrams, 2)]
    
    prev_removed = False
    if len(bigrams)>0:
        if '_'.join(bigrams[0]) in model.wv:
            unigrams.remove(bigrams[0][0])
            unigrams.remove(bigrams[0][1])
            unigrams.append('_'.join(bigrams[0]))
            prev_removed = True
    
    for bigram in bigrams[1:]:
        if '_'.join(bigram) in model.wv:
        
            unigrams.remove(bigram[1])
            unigrams.append('_'.join(bigram))
        
            if not prev_removed:
                unigrams.remove(bigram[0])
                prev_removed = True
        
    else:
        prev_removed = False
            
    return unigrams 
