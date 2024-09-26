# centralized_nlp_package/text_processing/text_analysis.py

import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from collections import Counter
from loguru import logger
from ..data_access.snowflake_utils import read_from_snowflake
from ..preprocessing.text_processing import preprocess_text, preprocess_text_list
from ..utils.config import Config
from ..utils.exceptions import FilesNotLoadedException


def get_blob_storage_path(config: Config, filename: str) -> str:
    """
    Constructs the full path to a file in blob storage.

    Args:
        config (Config): Configuration object.
        filename (str): Name of the file.

    Returns:
        str: Full path to the file.
    """
    path = f"{config.psycholinguistics.model_artifacts_path}{filename}"
    logger.debug(f"Constructed blob storage path: {path}")
    return path


def load_word_set(blob_util: Any, config: Config, filename: str) -> set:
    """
    Loads a set of words from a specified file in blob storage.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        filename (str): Name of the file.

    Returns:
        set: Set of words.
    """
    file_path = get_blob_storage_path(config, filename)
    try:
        word_set = blob_util.load_list_from_txt(file_path)
        logger.debug(f"Loaded word set from {file_path} with {len(word_set)} words.")
        return word_set
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(filename=filename)
    except Exception as e:
        logger.error(f"Error loading word set from {file_path}: {e}")
        raise


def load_syllable_counts(blob_util: Any, config: Config, filename: str) -> Dict[str, int]:
    """
    Loads syllable counts from a specified file in blob storage.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        filename (str): Name of the file.

    Returns:
        Dict[str, int]: Dictionary mapping words to their syllable counts.
    """
    file_path = get_blob_storage_path(config, filename)
    try:
        syllable_counts = blob_util.read_syllable_count(file_path)
        logger.debug(f"Loaded syllable counts from {file_path}.")
        return syllable_counts
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(filename=filename)
    except Exception as e:
        logger.error(f"Error loading syllable counts from {file_path}: {e}")
        raise


def check_negation(input_words: List[str], index: int, preprocess_obj: Any) -> bool:
    """
    Checks if a word at a given index is preceded by a negation within three words.

    Args:
        input_words (List[str]): List of tokenized words.
        index (int): Current word index.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.

    Returns:
        bool: True if negated, False otherwise.
    """
    negation_window = 3
    start = max(0, index - negation_window)
    for i in range(start, index):
        if preprocess_obj.word_negated(input_words[i]):
            logger.debug(f"Negation found before word '{input_words[index]}' at position {i}.")
            return True
    return False


def calculate_polarity_score(
    input_words: List[str],
    positive_words: set,
    negative_words: set,
    preprocess_obj: Any,
    statistics_obj: Any
) -> Tuple[float, int, int, int, float]:
    """
    Calculates the polarity score based on positive and negative word counts.

    Args:
        input_words (List[str]): List of tokenized words.
        positive_words (set): Set of positive words.
        negative_words (set): Set of negative words.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.
        statistics_obj (Any): Instance of Statistics.

    Returns:
        Tuple[float, int, int, int, float]: Polarity score, word count, sum of negatives, count of positives, legacy score.
    """
    positive_count = 0
    negative_count = 0
    word_count = len(input_words)

    for i, word in enumerate(input_words):
        if word in negative_words:
            negative_count -= 1
            logger.debug(f"Negative word found: {word} at position {i}.")

        if word in positive_words:
            if check_negation(input_words, i, preprocess_obj):
                negative_count -= 1
                logger.debug(f"Positive word '{word}' at position {i} negated.")
            else:
                positive_count += 1
                logger.debug(f"Positive word found: {word} at position {i}.")

    sum_negative = -negative_count
    polarity_score = (positive_count - sum_negative) / word_count if word_count > 0 else np.nan
    legacy_score = statistics_obj.combine_sent(positive_count, sum_negative)
    logger.debug(f"Polarity Score: {polarity_score}, Word Count: {word_count}, Sum Negative: {sum_negative}, Positive Count: {positive_count}, Legacy Score: {legacy_score}")
    return polarity_score, word_count, sum_negative, positive_count, legacy_score


def polarity_score_per_section(
    blob_util: Any,
    config: Config,
    text_list: List[str],
    preprocess_obj: Any,
    statistics_obj: Any
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Analyzes text to calculate litigious, complex, and uncertain word scores.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        text_list (List[str]): List of texts to analyze.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.
        statistics_obj (Any): Instance of Statistics.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[int], Optional[int], Optional[int]]:
            Litigious score, Complex score, Uncertain score, Word count,
            Litigious count, Complex count, Uncertain count.
    """
    litigious_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.litigious_flnm)
    complex_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.complex_flnm)
    uncertain_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.uncertianity_flnm)

    cleaned_text, input_words, word_count = preprocess_text(text_list, preprocess_obj)

    if cleaned_text and word_count > 1:
        litigious_count = sum(1 for word in input_words if word in litigious_words)
        complex_count = sum(1 for word in input_words if word in complex_words)
        uncertain_count = sum(1 for word in input_words if word in uncertain_words)

        litigious_score = litigious_count / word_count
        complex_score = complex_count / word_count
        uncertain_score = uncertain_count / word_count

        logger.info(f"Section Analysis - Litigious: {litigious_score}, Complex: {complex_score}, Uncertain: {uncertain_score}")
        return (litigious_score, complex_score, uncertain_score, word_count, litigious_count, complex_count, uncertain_count)
    else:
        logger.warning("Insufficient data for section analysis.")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def polarity_score_per_sentence(
    blob_util: Any,
    config: Config,
    text_list: List[str],
    preprocess_obj: Any,
    statistics_obj: Any
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """
    Analyzes sentences to calculate polarity scores.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        text_list (List[str]): List of sentences to analyze.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.
        statistics_obj (Any): Instance of Statistics.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
            Word counts, positive word counts, negative word counts per sentence.
    """
    litigious_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.litigious_flnm)
    complex_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.complex_flnm)
    uncertain_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.uncertianity_flnm)

    _, input_words_list, word_count_list = preprocess_text_list(text_list, preprocess_obj)

    if text_list and word_count_list:
        word_counts = []
        positive_counts = []
        negative_counts = []

        for input_words in input_words_list:
            polarity, wc, sum_neg, pos_count, _ = calculate_polarity_score(
                input_words, litigious_words, uncertain_words, preprocess_obj, statistics_obj
            )
            word_counts.append(wc)
            positive_counts.append(pos_count)
            negative_counts.append(sum_neg)

        logger.info("Sentence-level polarity analysis completed.")
        return (word_counts, positive_counts, negative_counts)
    else:
        logger.warning("Insufficient data for sentence-level analysis.")
        return (None, None, None)


def is_complex(word: str, syllables: Dict[str, int]) -> bool:
    """
    Determines if a word is complex based on syllable count and suffix rules.

    Args:
        word (str): The word to evaluate.
        syllables (Dict[str, int]): Dictionary of syllable counts.

    Returns:
        bool: True if the word is complex, False otherwise.
    """
    if word not in syllables:
        return False

    suffix_rules = {
        'es': 2,
        'ing': 3,
        'ed': 2
    }

    for suffix, strip_length in suffix_rules.items():
        if word.endswith(suffix):
            root = word[:-strip_length]
            if root in syllables and syllables[root] > 2:
                return True

    if syllables.get(word, 0) > 2:
        return True

    return False


def fog_analysis_per_section(
    blob_util: Any,
    config: Config,
    text_list: List[str],
    preprocess_obj: Any
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
    """
    Calculates the Fog Index for the input text to evaluate readability.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        text_list (List[str]): List of texts to analyze.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.

    Returns:
        Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
            Fog index, complex word count, average words per sentence, total word count.
    """
    syllables = load_syllable_counts(blob_util, config, config.psycholinguistics.filecfg.syllable_flnm)

    raw_text = ' '.join(text_list) if isinstance(text_list, list) else text_list
    total_word_count = len(raw_text.split())
    sentences = raw_text.split('. ')
    average_words_per_sentence = np.mean([len(sentence.strip().split()) for sentence in sentences]) if sentences else 0

    cleaned_text, input_words, word_count = preprocess_text(text_list, preprocess_obj)

    if cleaned_text and word_count > 1:
        complex_word_count = sum(is_complex(word, syllables) for word in input_words)
        fog_index = 0.4 * (average_words_per_sentence + 100 * (complex_word_count / total_word_count))
        logger.info(f"Fog Analysis - Fog Index: {fog_index}, Complex Words: {complex_word_count}, Average Words/Sentence: {average_words_per_sentence}, Total Words: {total_word_count}")
        return (fog_index, complex_word_count, average_words_per_sentence, total_word_count)
    else:
        logger.warning("Insufficient data for Fog Analysis.")
        return (np.nan, np.nan, np.nan, np.nan)


def fog_analysis_per_sentence(
    blob_util: Any,
    config: Config,
    text_list: List[str],
    preprocess_obj: Any
) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
    """
    Calculates the Fog Index for each sentence in the input list.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        text_list (List[str]): List of sentences to analyze.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.

    Returns:
        Tuple[Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
            Fog index list, complex word count list, total word count list.
    """
    syllables = load_syllable_counts(blob_util, config, config.psycholinguistics.filecfg.syllable_flnm)

    word_count_list = [len(sentence.split()) for sentence in text_list]
    average_words_per_sentence = np.mean(word_count_list) if text_list else 0

    _, input_words_list, _ = preprocess_text_list(text_list, preprocess_obj)

    if text_list and word_count_list:
        fog_index_list = []
        complex_word_count_list = []
        total_word_count_list = word_count_list

        for input_words in input_words_list:
            complex_count = sum(is_complex(word, syllables) for word in input_words)
            word_count = len(input_words)
            fog_index = 0.4 * (average_words_per_sentence + 100 * (complex_count / word_count)) if word_count > 0 else np.nan
            fog_index_list.append(fog_index)
            complex_word_count_list.append(complex_count)

        logger.info("Sentence-level Fog Analysis completed.")
        return (fog_index_list, complex_word_count_list, total_word_count_list)
    else:
        logger.warning("Insufficient data for sentence-level Fog Analysis.")
        return (None, None, None)


def tone_count_with_negation_check(
    blob_util: Any,
    config: Config,
    text_list: List[str],
    preprocess_obj: Any,
    statistics_obj: Any
) -> Tuple[List[float], List[int], List[int], List[int], List[float]]:
    """
    Counts positive and negative words with negation checks and calculates legacy scores.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        text_list (List[str]): List of texts to analyze.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.
        statistics_obj (Any): Instance of Statistics.

    Returns:
        Tuple[List[float], List[int], List[int], List[int], List[float]]:
            Polarity scores, word counts, negative word counts, positive word counts, legacy scores.
    """
    polarity_scores = []
    legacy_scores = []
    word_counts = []
    negative_word_counts = []
    positive_word_counts = []

    litigious_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.litigious_flnm)
    uncertain_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.uncertianity_flnm)

    cleaned_text, input_words, word_count = preprocess_text(text_list, preprocess_obj)

    if cleaned_text and word_count > 1:
        polarity, wc, sum_neg, pos_count, legacy = calculate_polarity_score(
            input_words, litigious_words, uncertain_words, preprocess_obj, statistics_obj
        )
        polarity_scores.append(polarity)
        word_counts.append(wc)
        negative_word_counts.append(sum_neg)
        positive_word_counts.append(pos_count)
        legacy_scores.append(legacy)
    else:
        polarity_scores.append(np.nan)
        word_counts.append(np.nan)
        negative_word_counts.append(np.nan)
        positive_word_counts.append(np.nan)
        legacy_scores.append(np.nan)

    logger.info("Tone count with negation check completed.")
    return (polarity_scores, word_counts, negative_word_counts, positive_word_counts, legacy_scores)


def tone_count_with_negation_check_per_sentence(
    blob_util: Any,
    config: Config,
    text_list: List[str],
    preprocess_obj: Any,
    statistics_obj: Any
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """
    Counts positive and negative words with negation checks for each sentence.

    Args:
        blob_util (Any): Instance of BlobStorageUtility.
        config (Config): Configuration object.
        text_list (List[str]): List of sentences to analyze.
        preprocess_obj (Any): Instance of DictionaryModelPreprocessor.
        statistics_obj (Any): Instance of Statistics.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
            Word counts, positive word counts per sentence, negative word counts per sentence.
    """
    word_counts = []
    positive_counts = []
    negative_counts = []

    litigious_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.litigious_flnm)
    uncertain_words = load_word_set(blob_util, config, config.psycholinguistics.filecfg.uncertianity_flnm)

    _, input_words_list, _ = preprocess_text_list(text_list, preprocess_obj)

    if text_list and input_words_list:
        for input_words in input_words_list:
            polarity, wc, sum_neg, pos_count, _ = calculate_polarity_score(
                input_words, litigious_words, uncertain_words, preprocess_obj, statistics_obj
            )
            word_counts.append(wc)
            positive_counts.append(pos_count)
            negative_counts.append(sum_neg)

        logger.info("Sentence-level tone count with negation check completed.")
        return (word_counts, positive_counts, negative_counts)
    else:
        logger.warning("Insufficient data for sentence-level tone count with negation check.")
        return (None, None, None)


# Perform analyses
# section_polarity = polarity_score_per_section(blob_util, config, text_list, preprocess_obj, statistics_obj)
# sentence_polarity = polarity_score_per_sentence(blob_util, config, text_list, preprocess_obj, statistics_obj)
# section_fog = fog_analysis_per_section(blob_util, config, text_list, preprocess_obj)
# sentence_fog = fog_analysis_per_sentence(blob_util, config, text_list, preprocess_obj)
# tone_check = tone_count_with_negation_check(blob_util, config, text_list, preprocess_obj, statistics_obj)
# tone_check_sentence = tone_count_with_negation_check_per_sentence(blob_util, config, text_list, preprocess_obj, statistics_obj)
