
import re
from typing import List, Tuple, Optional, Dict, Iterator
from pathlib import Path

import spacy
import numpy as np
from loguru import logger

from ..utils.config import Config
from ..utils.exceptions import FilesNotLoadedException


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


def expand_contractions(text: str, config: Config) -> str:
    """
    Expands contractions in the text based on a provided contractions list.

    Args:
        text (str): The input text.
        config (Config): Configuration object containing file paths.

    Returns:
        str: Text with expanded contractions.
    """
    contractions_path = Path(config.model_artifacts.path) / config.blob_filenames.contraction_flnm
    contractions = load_list_from_txt(str(contractions_path), is_lower=True)
    contractions_dict = {word.lower(): word for word in contractions}  # Assuming contractions list is "can't" etc.

    def replace(match):
        word = match.group(0)
        return contractions_dict.get(word.lower(), word)

    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions_dict.keys()) + r')\b')
    expanded_text = pattern.sub(replace, text)
    logger.debug("Expanded contractions in text.")
    return expanded_text


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


def _clean_text(text: str, config: Config) -> str:
    """
    Cleans the input text by expanding contractions, removing unwanted characters, and normalizing spaces.

    Args:
        text (str): The input text to clean.
        config (Config): Configuration object containing file paths.

    Returns:
        str: The cleaned text.
    """
    text = expand_contractions(text.replace('’', "'"), config)
    text = text.strip().lower()
    text = text.replace('"', '')
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("\s+", ' ', text)
    cleaned_text = text.strip()
    logger.debug("Cleaned the text.")
    return cleaned_text


def word_tokenizer(text: str, config: Config, spacy_tokenizer: spacy.Language) -> List[str]:
    """
    Tokenizes the text and performs lemmatization.

    Args:
        text (str): The input text to tokenize.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        List[str]: A list of lemmatized words.
    """
    stop_words_path = Path(config.model_artifacts.path) / config.blob_filenames.stop_words_flnm
    stop_words_list = load_list_from_txt(str(stop_words_path), is_lower=True)

    doc = spacy_tokenizer(text)
    token_lemmatized = [token.lemma_ for token in doc]
    filtered_words = [word for word in token_lemmatized if word not in stop_words_list]
    logger.debug(f"Tokenized and filtered words. {len(filtered_words)} words remaining.")
    return filtered_words


def preprocess_text(text_input: Optional[Union[str, List[str]]], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[Optional[str], List[str], int]:
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
        cleaned_text = _clean_text(text, config)
        input_words = word_tokenizer(cleaned_text, config, spacy_tokenizer)
        word_count = len(input_words)
        logger.debug("Preprocessed single text input.")
        return cleaned_text, input_words, word_count
    else:
        logger.warning("Preprocessing failed due to invalid input.")
        return None, [], 0


def preprocess_text_list(text_list: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[List[str], List[List[str]], List[int]]:
    """
    Preprocesses a list of texts by cleaning and tokenizing each.

    Args:
        text_list (List[str]): The list of texts to preprocess.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[List[str], List[List[str]], List[int]]: 
            - List of preprocessed texts.
            - List of input words for each text.
            - List of word counts for each text.
    """
    final_text_list = []
    input_word_list = []
    word_count_list = []

    for text in text_list:
        cleaned_text = _clean_text(text, config)
        token_word_list = word_tokenizer(cleaned_text, config, spacy_tokenizer)
        final_text_list.append(cleaned_text)
        input_word_list.append(token_word_list)
        word_count_list.append(len(token_word_list))
        logger.debug(f"Preprocessed text: {text[:50]}...")

    logger.info("Preprocessed list of texts.")
    return final_text_list, input_word_list, word_count_list

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


def _is_complex(word: str, config: Config) -> bool:
    """
    Determines if a word is complex based on syllable count.

    Args:
        word (str): The word to evaluate.
        config (Config): Configuration object containing file paths.

    Returns:
        bool: True if the word is complex, False otherwise.
    """
    syllables_path = Path(config.model_artifacts.path) / config.blob_filenames.syllable_flnm
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

