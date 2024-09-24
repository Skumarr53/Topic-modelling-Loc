# centralized_nlp_package/embedding/embedding_utils.py

from typing import List, Optional
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger

def embed_text(tokens: List[str], model: Word2Vec) -> Optional[np.ndarray]:
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
    return np.mean(valid_vectors, axis=0)

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
