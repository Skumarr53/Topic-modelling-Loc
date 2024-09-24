from centralized_nlp_package.text_processing.text_utils import (preprocess_text, 
                                                                preprocess_text_list, 
                                                                load_list_from_txt)
from pathlib import Path
from typing import List
from loguru import logger
import spacy
from ..utils.config import Config
from ..utils.exceptions import FilesNotLoadedException


def LM_analysis_per_section(text: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[float, float, float, int, int, int, int]:
    """
    Analyzes text for litigious, complex, and uncertain words.

    Args:
        text (List[str]): List of text strings.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[float, float, float, int, int, int, int]: 
            - Litigious words score
            - Complex words score
            - Uncertain words score
            - Word count
            - Litigious words count
            - Complex words count
            - Uncertain words count
    """
    logger.info("Starting LM_analysis_per_section.")
    cleaned_text, input_words, word_count = preprocess_text(text, config, spacy_tokenizer)

    if cleaned_text and word_count > 1:
        litigious_path = Path(config.model_artifacts.path) / config.blob_filenames.litigious_flnm
        complex_path = Path(config.model_artifacts.path) / config.blob_filenames.complex_flnm
        uncertain_path = Path(config.model_artifacts.path) / config.blob_filenames.uncertianity_flnm

        litigious_words = load_list_from_txt(str(litigious_path))
        complex_words = load_list_from_txt(str(complex_path))
        uncertain_words = load_list_from_txt(str(uncertain_path))

        litigious_words_count = sum(word in litigious_words for word in input_words)
        complex_words_count = sum(word in complex_words for word in input_words)
        uncertain_words_count = sum(word in uncertain_words for word in input_words)

        litigious_words_score = litigious_words_count / word_count
        complex_words_score = complex_words_count / word_count
        uncertain_words_score = uncertain_words_count / word_count

        logger.info("LM_analysis_per_section completed successfully.")
        return (
            litigious_words_score,
            complex_words_score,
            uncertain_words_score,
            word_count,
            litigious_words_count,
            complex_words_count,
            uncertain_words_count
        )
    else:
        logger.warning("Insufficient data for LM_analysis_per_section.")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def LM_analysis_per_sentence(texts: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[List[int], List[int], List[int]]:
    """
    Analyzes each sentence for litigious, complex, and uncertain words.

    Args:
        texts (List[str]): List of sentences.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[List[int], List[int], List[int]]: 
            - Litigious words count list per sentence
            - Complex words count list per sentence
            - Uncertain words count list per sentence
    """
    logger.info("Starting LM_analysis_per_sentence.")
    _, input_words_list, _ = preprocess_text_list(texts, config, spacy_tokenizer)

    litigious_path = Path(config.model_artifacts.path) / config.blob_filenames.litigious_flnm
    complex_path = Path(config.model_artifacts.path) / config.blob_filenames.complex_flnm
    uncertain_path = Path(config.model_artifacts.path) / config.blob_filenames.uncertianity_flnm

    litigious_words = load_list_from_txt(str(litigious_path))
    complex_words = load_list_from_txt(str(complex_path))
    uncertain_words = load_list_from_txt(str(uncertain_path))

    litigious_words_count_list = [
        sum(word in litigious_words for word in words)
        for words in input_words_list
    ]
    complex_words_count_list = [
        sum(word in complex_words for word in words)
        for words in input_words_list
    ]
    uncertain_words_count_list = [
        sum(word in uncertain_words for word in words)
        for words in input_words_list
    ]

    logger.info("LM_analysis_per_sentence completed successfully.")
    return (
        litigious_words_count_list,
        complex_words_count_list,
        uncertain_words_count_list
    )


def fog_analysis_per_section(texts: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[float, int, float, int]:
    """
    Calculates the Fog Index for the provided text.

    Args:
        texts (List[str]): List of text strings.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[float, int, float, int]: 
            - Fog index
            - Complex word count
            - Average words per sentence
            - Total word count
    """
    logger.info("Starting fog_analysis_per_section.")
    raw_text = ' '.join(texts)
    total_word_count = len(raw_text.split())
    average_word_per_sentence = np.mean([len(sentence.strip().split()) for sentence in raw_text.split('. ')])
    cleaned_text, input_words, word_count = preprocess_text(texts, config, spacy_tokenizer)

    if cleaned_text and word_count > 1:
        complex_word_count = sum(
            _is_complex(word, config) for word in input_words
        )
        fog_index = 0.4 * (average_word_per_sentence + 100 * (complex_word_count / total_word_count))
        logger.info("fog_analysis_per_section completed successfully.")
        return (fog_index, complex_word_count, average_word_per_sentence, total_word_count)
    else:
        logger.warning("Insufficient data for fog_analysis_per_section.")
        return (np.nan, np.nan, np.nan, np.nan)


def fog_analysis_per_sentence(texts: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[List[float], List[int], List[int]]:
    """
    Calculates the Fog Index for each sentence.

    Args:
        texts (List[str]): List of sentences.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[List[float], List[int], List[int]]: 
            - List of Fog indices per sentence
            - List of complex word counts per sentence
            - List of total word counts per sentence
    """
    logger.info("Starting fog_analysis_per_sentence.")
    _, input_words_list, word_count_list = preprocess_text_list(texts, config, spacy_tokenizer)

    fog_index_list = []
    complex_word_count_list = []
    total_word_count_list = word_count_list.copy()

    for words in input_words_list:
        complex_count = sum(_is_complex(word, config) for word in words)
        word_count = len(words)
        fog_index = 0.4 * (word_count + 100 * (complex_count / word_count)) if word_count > 0 else np.nan
        fog_index_list.append(fog_index)
        complex_word_count_list.append(complex_count)
        logger.debug(f"Sentence Fog Index: {fog_index}, Complex Words: {complex_count}")

    logger.info("fog_analysis_per_sentence completed successfully.")
    return (fog_index_list, complex_word_count_list, total_word_count_list)


def polarity_score_per_section(text: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[float, int, int, int, float]:
    """
    Calculates the polarity score for the provided text.

    Args:
        text (List[str]): List of text strings.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[float, int, int, int, float]: 
            - Polarity score
            - Word count
            - Sum of negative words
            - Positive words count
            - Legacy score
    """
    logger.info("Starting polarity_score_per_section.")
    cleaned_text, input_words, word_count = preprocess_text(text, config, spacy_tokenizer)
    negate_words_path = Path(config.model_artifacts.path) / config.blob_filenames.negate_words_flnm
    negate_words = load_list_from_txt(str(negate_words_path), is_lower=True)

    litigious_path = Path(config.model_artifacts.path) / config.blob_filenames.litigious_flnm
    vocab_pos_path = Path(config.model_artifacts.path) / config.blob_filenames.vocab_pos_flnm
    vocab_neg_path = Path(config.model_artifacts.path) / config.blob_filenames.vocab_neg_flnm

    litigious_words = load_list_from_txt(str(litigious_path))
    positive_words = load_list_from_txt(str(vocab_pos_path))
    negative_words = load_list_from_txt(str(vocab_neg_path))

    if cleaned_text and word_count > 1:
        positive_words_count = 0
        negative_words_count = 0
        positive_words_list = []
        negative_words_list = []

        for i, word in enumerate(input_words):
            if word in negative_words:
                negative_words_count -= 1
                negative_words_list.append(f"{word} (with negation)")
            if word in positive_words:
                negated = any(
                    prev_word in negate_words for prev_word in input_words[max(i - 3, 0):i]
                )
                if negated:
                    negative_words_count -= 1
                    negative_words_list.append(f"{word} (with negation)")
                else:
                    positive_words_count += 1
                    positive_words_list.append(word)

        sum_negative = abs(negative_words_count)
        polarity = (positive_words_count - sum_negative) / word_count
        legacy_score = combine_sent(positive_words_count, sum_negative)

        logger.info("polarity_score_per_section completed successfully.")
        return (polarity, word_count, sum_negative, positive_words_count, legacy_score)
    else:
        logger.warning("Insufficient data for polarity_score_per_section.")
        return (np.nan, np.nan, np.nan, np.nan, np.nan)


def polarity_score_per_sentence(texts: List[str], config: Config, spacy_tokenizer: spacy.Language) -> Tuple[List[float], List[int], List[int], List[int], List[float]]:
    """
    Calculates the polarity score for each sentence.

    Args:
        texts (List[str]): List of sentences.
        config (Config): Configuration object containing file paths.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[List[float], List[int], List[int], List[int], List[float]]: 
            - List of polarity scores per sentence
            - List of word counts per sentence
            - List of negative word counts per sentence
            - List of positive word counts per sentence
            - List of legacy scores per sentence
    """
    logger.info("Starting polarity_score_per_sentence.")
    _, input_words_list, word_count_list = preprocess_text_list(texts, config, spacy_tokenizer)
    negate_words_path = Path(config.model_artifacts.path) / config.blob_filenames.negate_words_flnm
    negate_words = load_list_from_txt(str(negate_words_path), is_lower=True)

    vocab_neg_path = Path(config.model_artifacts.path) / config.blob_filenames.vocab_neg_flnm
    vocab_pos_path = Path(config.model_artifacts.path) / config.blob_filenames.vocab_pos_flnm

    negative_words = load_list_from_txt(str(vocab_neg_path))
    positive_words = load_list_from_txt(str(vocab_pos_path))

    polarity_scores = []
    word_counts = []
    negative_word_counts = []
    positive_word_counts = []
    legacy_scores = []

    for words in input_words_list:
        positive_count = 0
        negative_count = 0

        for i, word in enumerate(words):
            if word in negative_words:
                negative_count -= 1
            if word in positive_words:
                negated = any(
                    prev_word in negate_words for prev_word in words[max(i - 3, 0):i]
                )
                if negated:
                    negative_count -= 1
                else:
                    positive_count += 1

        sum_negative = abs(negative_count)
        polarity = (positive_count - sum_negative) / len(words) if len(words) > 0 else np.nan
        legacy = combine_sent(positive_count, sum_negative)

        polarity_scores.append(polarity)
        word_counts.append(len(words))
        negative_word_counts.append(sum_negative)
        positive_word_counts.append(positive_count)
        legacy_scores.append(legacy)

        logger.debug(f"Sentence polarity: {polarity}, Positive: {positive_count}, Negative: {sum_negative}")

    logger.info("polarity_score_per_sentence completed successfully.")
    return (