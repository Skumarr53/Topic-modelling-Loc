
import re
from typing import List, Tuple, Optional, Dict, Iterator, Union
from pathlib import Path

import spacy
import numpy as np
from loguru import logger
from centralized_nlp_package.utils.logging_setup import setup_logging

from centralized_nlp_package.utils.config import config
from centralized_nlp_package.utils.exception import FilesNotLoadedException
# from centralized_nlp_package.preprocessing.text_preprocessing import clean_text 

setup_logging()



def check_datatype(text_list):
    """
    IF THE TEXT LIST HAS TEXT, PROCESS IT, OTHERWISE OUTPUT NAN
    
    Parameters:
    argument1 (list): list
   
    Returns:
    list:list
    
    """
    if (not isinstance(text_list,str) and text_list and ' '.join(text_list).strip(' ')) or (isinstance(text_list,str) and text_list.strip(' ')):
      #Text input is not and empty string or list
      if not isinstance(text_list,str):
        #Text input is a list
        text = ' '.join(text_list)
      else:
        #Text input is a string
        text = text_list
    else:
      text = False
        
    return text

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


def load_content_from_txt(file_path: str) -> str:
    """
    Reads the entire content of a text file from the given file path.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file.

    Raises:
        FilesNotLoadedException: If the file is not found at the given path.
    """
    try:
        with open(file_path, "r") as f_obj:
            content = f_obj.read()
        logger.debug(f"Loaded content from {file_path}.")
        return content
    except FileNotFoundError as ex:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(f"File not found: {file_path}") from ex


def load_list_from_txt(file_path: str, is_lower: bool = True) -> set:
    """
    Reads the content of a text file and returns it as a set of lines.

    Args:
        file_path (str): The path to the text file.
        is_lower (bool, optional): If True, converts the content to lowercase. Defaults to True.

    Returns:
        set: A set of lines from the text file.

    Raises:
        FilesNotLoadedException: If there is an error reading the file.
    """
    try:
        content = load_content_from_txt(file_path)
        if is_lower:
            content = content.lower()
        words_list = set(content.split('\n'))
        logger.debug(f"Loaded list from {file_path} with {len(words_list)} entries.")
        return words_list
    except Exception as e:
        logger.error(f"Error loading list from {file_path}: {e}")
        raise FilesNotLoadedException(f"Error loading list from {file_path}: {e}") from e


def expand_contractions(text):
    """Expand contractions

    Parameters:
    argument1 (str): text
   
    Returns:
    str:returns text with expanded contractions
    
    """
    # TODO: add contraction words to config or txt file then load
    contraction_dict = None
    for word in text.split():
      if word.lower() in contraction_dict:
        text = text.replace(word, contraction_dict[word.lower()])
    return text

def check_datatype(text_input: Optional[Union[str, List[str]]]) -> Optional[str]:
    """
    Validates and formats the input text.

    Args:
        text_input (Optional[Union[str, List[str]]]): The input text or list of texts to validate.

    Returns:
        Optional[str]: Joined text if valid, else None.
    """
    if isinstance(text_input, list):
        joined_text = ' '.join(text_input).strip()
    elif isinstance(text_input, str):
        joined_text = text_input.strip()
    else:
        joined_text = None

    if joined_text:
        logger.debug("Input text is valid and formatted.")
        return joined_text
    else:
        logger.warning("Input text is invalid or empty.")
        return None


def word_tokenizer(text: str, spacy_tokenizer: spacy.Language) -> List[str]:
    """
    Tokenizes the text and performs lemmatization.

    Args:
        text (str): The input text to tokenize.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        List[str]: A list of lemmatized words.
    """
    stop_words_path = Path(config.lib_config.paths.model_artifacts.path) / config.lib_config.filenames.stop_words_flnm
    stop_words_list = load_list_from_txt(str(stop_words_path), is_lower=True)

    doc = spacy_tokenizer(text.lower())
    token_lemmatized = [token.lemma_ for token in doc]
    filtered_words = [word for word in token_lemmatized if word not in stop_words_list]
    logger.debug(f"Tokenized and filtered words. {len(filtered_words)} words remaining.")
    return filtered_words

def combine_sent(x: int, y: int) -> float:
    """
    Combines two sentiment scores.

    Args:
        x (int): Positive sentiment count.
        y (int): Negative sentiment count.

    Returns:
        float: Combined sentiment score.
    """
    if (x + y) == 0:
        return 0.0
    else:
        combined_score = (x - y) / (x + y)
        logger.debug(f"Combined sentiment score: {combined_score}")
        return combined_score


def _is_complex(word: str) -> bool:
    """
    Determines if a word is complex based on syllable count.

    Args:
        word (str): The word to evaluate.
        config (Config): Configuration object containing file paths.

    Returns:
        bool: True if the word is complex, False otherwise.
    """
    syllables_path = Path(config.lib_config.paths.model_artifacts.path) / config.lib_config.filenames.syllable_flnm
    syllables = load_syllable_count(str(syllables_path))
    syllable_count = syllables.get(word.lower(), 0)
    is_complex_word = syllable_count > 2
    logger.debug(f"Word '{word}' has {syllable_count} syllables. Complex: {is_complex_word}")
    return is_complex_word

def load_syllable_count(file_path: str) -> Dict[str, int]:
    """
    Reads a file containing words and their syllable counts, and returns a dictionary.

    Args:
        file_path (str): The path to the text file.

    Returns:
        Dict[str, int]: A dictionary where keys are words and values are their syllable counts.

    Raises:
        FilesNotLoadedException: If the file is not found at the given path.
    """
    syllables = {}
    try:
        with open(file_path, 'r') as fs_pos_words:
            for line in fs_pos_words:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, count = parts
                    syllables[word.lower()] = int(count)
        logger.debug(f"Loaded syllable counts from {file_path}.")
        return syllables
    except FileNotFoundError as ex:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(f"File not found: {file_path}") from ex
    except ValueError as ve:
        logger.error(f"Value error in syllable count file: {ve}")
        raise FilesNotLoadedException(f"Invalid format in syllable count file: {ve}") from ve

