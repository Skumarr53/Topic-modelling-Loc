# centralized_nlp_package/preprocessing/text_processing.py

import re
from typing import List, Tuple, Optional, Union
import spacy
from loguru import logger
from centralized_nlp_package.utils.logging_setup import setup_logging
from centralized_nlp_package.utils.config import config
from centralized_nlp_package.text_processing.text_utils import check_datatype, expand_contractions, word_tokenizer


setup_logging()

def initialize_spacy() -> spacy.Language:
    """
    Initializes and configures the SpaCy model with custom settings.

    Args:
        config (Config): Configuration object containing model settings.

    Returns:
        spacy.Language: Configured SpaCy model.
    """
    logger.info(f"Loading SpaCy model: {config.lib_config.preprocessing.spacy_model}")
    nlp = spacy.load(config.lib_config.preprocessing.spacy_model, disable=['parser'])
    # Excluding financially relevant stopwords
    nlp.Defaults.stop_words -= set(config.lib_config.preprocessing.additional_stop_words)
    nlp.max_length = config.lib_config.preprocessing.max_length
    logger.info("SpaCy model initialized.")
    return nlp

def find_ngrams(input_list: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generates n-grams from a list of tokens.

    Args:
        input_list (List[str]): List of tokens.
        n (int): Number of tokens in each n-gram.

    Returns:
        List[Tuple[str, ...]]: List of n-grams as tuples.
    """
    return list(zip(*[input_list[i:] for i in range(n)]))

def remove_unwanted_phrases_and_validate(sentence: str) -> Optional[str]:
    """
    Cleans the input sentence by removing unwanted phrases.

    Args:
        sentence (str): The input sentence to process.

    Returns:
        Optional[str]: Cleaned sentence or None if it doesn't meet criteria.
    """
    logger.debug("Cleaning sentence.")
    # Remove specified phrases
    for phrase in Config.preprocessing.preprocessing.cleanup_phrases:
        sentence = sentence.replace(phrase, "")
    # Check word count
    if len(sentence.split()) < Config.preprocessing.preprocessing.min_word_length:
        logger.debug("Sentence below minimum word length. Skipping.")
        return None
    
    # Remove greetings
    if any(greet in sentence.lower() for greet in Config.preprocessing.preprocessing.greeting_phrases):
        logger.debug("Greeting phrase detected. Skipping.")
        return None

    logger.debug(f"Cleaned sentence: {sentence}")
    return sentence if sentence else None


def tokenize_and_lemmatize_text(doc: str, nlp: spacy.Language) -> List[str]:
    """
    Tokenizes and lemmatizes the document text.

    Args:
        doc (str): The input text to tokenize.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        List[str]: List of lemmatized and filtered tokens.
    """
    tokens = [ent.lemma_.lower() for ent in nlp(doc) 
              if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']
    logger.debug(f"Tokenized document into {len(tokens)} tokens.")
    return tokens

def tokenize_matched_words(doc: str, nlp: spacy.Language) -> List[str]:
    """
    Tokenizes matched words by removing stop words and numbers.

    Args:
        doc (str): The input text to tokenize.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        List[str]: List of lemmatized and filtered tokens.
    """
    ret = []
    for ent in nlp(doc):
        if ent.pos_ == 'PROPN' or ent.text[0].isupper():
            ret.append(ent.text.lower())
            continue
        if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM':
            ret.append(ent.lemma_.lower())
    logger.debug(f"Tokenized matched words into {len(ret)} tokens.")
    return ret

def preprocess_text(text_input: Optional[Union[str, List[str]]], spacy_tokenizer: spacy.Language) -> Tuple[Optional[str], List[str], int]:
    """
    Preprocesses the text by cleaning and tokenizing.

    Args:
        text_input (Optional[Union[str, List[str]]]): The input text or list of texts to preprocess.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[Optional[str], List[str], int]: The preprocessed text, list of input words, and word count.
    """
    text = check_datatype(text_input)
    if text:
        cleaned_text = clean_text(text)
        input_words = word_tokenizer(cleaned_text, spacy_tokenizer)
        word_count = len(input_words)
        logger.debug("Preprocessed single text input.")
        return cleaned_text, input_words, word_count
    else:
        logger.warning("Preprocessing failed due to invalid input.")
        return None, [], 0

def preprocess_text_list(text_list: List[str], spacy_tokenizer: spacy.Language) -> Tuple[List[str], List[List[str]], List[int]]:
    """
    Preprocesses a list of texts by cleaning and tokenizing each.

    Args:
        text_list (List[str]): The list of texts to preprocess.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        Tuple[List[str], List[List[str]], List[int]]: Cleaned texts, list of token lists, and word counts.
    """
    logger.debug("Preprocessing list of texts.")
    final_text_list = []
    input_word_list = []
    word_count_list = []

    for text in text_list:
        cleaned = clean_text(text)  # Pass the actual config
        if cleaned:
            tokens = word_tokenizer(cleaned, spacy_tokenizer)
            final_text_list.append(cleaned)
            input_word_list.append(tokens)
            word_count_list.append(len(tokens))
        else:
            final_text_list.append("")
            input_word_list.append([])
            word_count_list.append(0)

    logger.debug(f"Preprocessed {len(text_list)} texts.")
    return final_text_list, input_word_list, word_count_list

def clean_text(text: str) -> str:
    """
    Cleans the input text by expanding contractions, removing unwanted characters, and normalizing spaces.
    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    text = expand_contractions(re.sub('’', "'", text))
    text = text.strip().lower()
    text = text.replace('"', '')
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("\s+", ' ', text)
    logger.debug("Cleaned the text.")
    return text.strip()