# centralized_nlp_package/embedding/embedding_utils.py

from typing import List, Optional
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from centralized_nlp_package.utils.logging_setup import setup_logging
from centralized_nlp_package.preprocessing.text_preprocessing import tokenize_text
from  centralized_nlp_package.preprocessing.ngram_utils import find_ngrams

setup_logging()

def average_token_embeddings(tokens: List[str], model: Word2Vec) -> Optional[np.ndarray]:
    """
    Generates an embedding for the given list of tokens by averaging their vectors.

    Args:
        tokens (List[str]): List of tokens (unigrams or bigrams).
        model (Word2Vec): Trained Word2Vec model.

    Returns:
        Optional[np.ndarray]: Averaged embedding vector or None if no tokens are in the model.
    """
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not valid_vectors:
        return None
    return np.mean(np.stack((valid_vectors), axis=0))

def generate_ngram_embedding(x: str, model: Word2Vec) -> np.ndarray:
    """
    Computes the embedding for a given string `x` using a word2vec model. If bigrams exist in the model, 
    they are prioritized over unigrams.

    Args:
        x (str): The input string.
        model: The word2vec model that provides the word vectors.

    Returns:
        np.ndarray: The average embedding vector for the input string.
                    Returns None if no embedding can be found.
    """
    # Handle direct look-up for phrases with underscores
    if '_' in x:
        try:
            return model.wv[x]
        except KeyError:
            pass  # Continue to unigram/bigram handling if not found

    # Tokenize the input string into unigrams
    unigrams = tokenize_text(x)
    
    # Create a set of bigrams from the unigrams
    bigrams = [f"{b[0]}_{b[1]}" for b in find_ngrams(unigrams, 2)]
    
    # Process bigrams and adjust unigrams list if bigrams are found in the model
    final_tokens = []
    prev_bigram_used = False
    
    for i, bigram in enumerate(bigrams):
        if bigram in model.wv:
            # If bigram exists in the model, use it and remove the corresponding unigrams
            final_tokens.append(bigram)
            if i == 0:  # Remove the first two unigrams for the first bigram
                unigrams.pop(0)
                unigrams.pop(0)
            else:  # For subsequent bigrams, just remove the second word
                unigrams.pop(1)
            prev_bigram_used = True
        else:
            prev_bigram_used = False
    
    # Add remaining unigrams that were not removed by bigrams
    final_tokens.extend(unigrams)
    
    # Compute the mean of the embeddings for the final tokens
    try:
        return average_token_embeddings(final_tokens,model) # np.mean(np.stack([model.wv[token] for token in final_tokens if token in model.wv]), axis=0)
    except ValueError:  # Catch empty stack
        try:
            return model.wv[x]  # Fallback to direct lookup of the original string
        except KeyError:
            return None

def nearest_neighbors(words: pd.DataFrame, model: Word2Vec, num_neigh: int = 50, regularize: bool = False) -> pd.DataFrame:
    """
    Finds the nearest neighbor words for each topic based on the embeddings.

    Args:
        words (pd.DataFrame): DataFrame containing 'label' and 'match' columns.
        model (Word2Vec): Trained Word2Vec model.
        num_neigh (int, optional): Number of neighbors to retrieve. Defaults to 50.
        regularize (bool, optional): Whether to apply cosine similarity normalization. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing labels, embeddings, matched words, and similarity scores.
    """
    logger.info("Finding nearest neighbors for each topic.")
    alist = {'label': [], 'embed': [], 'match': [], 'sim': []}
    for topic in set(words['label']):
        logger.debug(f"Processing topic: {topic}")
        topic_matches = words[words['label'] == topic]['match'].tolist()
        positive = []
        for match in topic_matches:
            tokens = match.split()  # Assuming match is a string of tokens
            positive.extend([token for token in tokens if token in model.wv])
        if not positive:
            logger.warning(f"No valid tokens found for topic {topic}. Skipping.")
            continue
        # Get most similar words
        similar = model.wv.most_similar(positive=positive, topn=num_neigh)
        for word, similarity in similar:
            alist['label'].append(topic)
            alist['embed'].append(model.wv[word])
            alist['match'].append(word)
            alist['sim'].append(similarity)
    neighbors_df = pd.DataFrame(alist)
    logger.info("Nearest neighbors retrieval completed.")
    return neighbors_df
