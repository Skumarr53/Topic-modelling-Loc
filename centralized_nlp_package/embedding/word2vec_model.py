# centralized_nlp_package/embedding/word2vec_model.py

from gensim.models import Word2Vec, Phrases
from typing import List, Dict, Any
from loguru import logger
from centralized_nlp_package.utils.logging_setup import setup_logging
from pathlib import Path
from centralized_nlp_package.utils.config import config

setup_logging()

## TODO: Topic modelling 
def train_word2vec(sents: List[List[str]], bigram = False) -> Word2Vec:
    """
    Trains a Word2Vec model on the provided corpus.

    Args:
        feed (List[List[str]]): Corpus of tokenized sentences.
        model_params (Dict[str, Any]): Parameters for Word2Vec.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    model_params = (config.lib_config.word2vec_bigram 
                    if bigram else 
                    config.lib_config.word2vec_unigram)
    logger.info("Starting Word2Vec model training.")
    model = Word2Vec(sentences=sents, **model_params)
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
