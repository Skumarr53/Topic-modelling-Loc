# centralized_nlp_package/preprocessing/text_processing.py

import spacy
from typing import List
from ..utils.config import Config
from loguru import logger

def initialize_spacy_model(config: Config) -> spacy.Language:
    """
    Initializes and configures the SpaCy model with custom settings.

    Args:
        config (Config): Configuration object containing model settings.

    Returns:
        spacy.Language: Configured SpaCy model.
    """
    logger.info(f"Loading SpaCy model: {config.preprocessing.spacy_model}")
    nlp = spacy.load(config.preprocessing.spacy_model, disable=['parser'])
    logger.debug("Customizing stop words.")
    nlp.Defaults.stop_words -= set(config.preprocessing.additional_stop_words)
    nlp.max_length = config.preprocessing.max_length
    logger.info("SpaCy model initialized.")
    return nlp

def word_tokenize(doc: str, nlp: spacy.Language) -> List[str]:
    """
    Tokenizes and lemmatizes a document using SpaCy.

    Args:
        doc (str): The input text document.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        List[str]: List of lemmatized tokens.
    """
    tokens = [ent.lemma_.lower() for ent in nlp(doc) 
              if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']
    logger.debug(f"Tokenized document into {len(tokens)} tokens.")
    return tokens
