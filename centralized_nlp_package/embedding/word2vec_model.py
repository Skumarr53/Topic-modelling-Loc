# centralized_nlp_package/embedding/word2vec_model.py

from gensim.models import Word2Vec, Phrases
from typing import List, Dict, Any
from loguru import logger
from pathlib import Path

def train_word2vec(feed: List[List[str]], gen_bigram: bool, model_params: Dict[str, Any]) -> Word2Vec:
    """
    Trains a Word2Vec model on the provided corpus.

    Args:
        feed (List[List[str]]): Corpus of tokenized sentences.
        gen_bigram (bool): Whether to generate bigrams.
        model_params (Dict[str, Any]): Parameters for Word2Vec.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    if gen_bigram:
        logger.info("Generating bigrams using Phrases.")
        phrases = Phrases(feed, threshold=model_params.get('bigram_threshold', 2))
        sentences = phrases[feed]
        logger.debug("Bigrams generated.")
    else:
        sentences = feed
        logger.debug("Bigram generation skipped.")
    
    logger.info("Starting Word2Vec model training.")
    model = Word2Vec(sentences=sentences, **model_params)
    logger.info("Word2Vec model training completed.")
    return model

def save_model(model: Word2Vec, path: str) -> None:
    """
    Saves the trained Word2Vec model to the specified path.

    Args:
        model (Word2Vec): Trained Word2Vec model.
        path (str): File path to save the model.
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Word2Vec model to {model_path}")
    model.save(str(model_path))
    logger.info("Word2Vec model saved successfully.")
