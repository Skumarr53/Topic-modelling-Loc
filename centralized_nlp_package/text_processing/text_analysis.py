# centralized_nlp_package/text_processing/text_analysis.py

import os, re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from collections import Counter
from loguru import logger
from centralized_nlp_package.utils.logging_setup import setup_logging
from text_utils import load_list_from_txt, combine_sent, word_tokenizer, find_ngrams
from centralized_nlp_package.data_access.snowflake_utils import read_from_snowflake
from centralized_nlp_package.preprocessing.text_preprocessing import preprocess_text, preprocess_text_list, tokenize_matched_words
from centralized_nlp_package.utils.config import config
from centralized_nlp_package.utils.exception import FilesNotLoadedException

setup_logging()

def load_word_set(filename: str) -> set:
    """
    Loads a set of words from a specified file in blob storage.

    Args:
        config (Config): Configuration object.
        filename (str): Name of the file.

    Returns:
        set: Set of words.
    """
    file_path = os.path.join(config.lib_config.model_artifacts_path, filename)
    try:
        word_set = load_list_from_txt(file_path)
        logger.debug(f"Loaded word set from {file_path} with {len(word_set)} words.")
        return word_set
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(filename=filename)
    except Exception as e:
        logger.error(f"Error loading word set from {file_path}: {e}")
        raise


def load_syllable_counts(  filename: str) -> Dict[str, int]:
    """
    Loads syllable counts from a specified file in blob storage.

    Args:
        config (Config): Configuration object.
        filename (str): Name of the file.

    Returns:
        Dict[str, int]: Dictionary mapping words to their syllable counts.
    """
    file_path =  os.path.join(config.psycholinguistics.model_artifacts_path, filename)
    try:
        syllable_counts = load_syllable_counts(file_path)
        logger.debug(f"Loaded syllable counts from {file_path}.")
        return syllable_counts
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(filename=filename)
    except Exception as e:
        logger.error(f"Error loading syllable counts from {file_path}: {e}")
        raise


def check_negation(input_words: List[str], index: int) -> bool:
    """
    Checks if a word at a given index is preceded by a negation within three words.

    Args:
        input_words (List[str]): List of tokenized words.
        index (int): Current word index.

    Returns:
        bool: True if negated, False otherwise.
    """
    # TODO: add negation words to config or txt file then load
    negated_words = None
    negation_window = 3
    start = max(0, index - negation_window)
    for i in range(start, index):
        if input_words[i] in negated_words:
            logger.debug(f"Negation found before word '{input_words[index]}' at position {i}.")
            return True
    return False


def calculate_polarity_score(
    input_words: List[str],
    positive_words: set,
    negative_words: set,
) -> Tuple[float, int, int, int, float]:
    """
    Calculates the polarity score based on positive and negative word counts.

    Args:
        input_words (List[str]): List of tokenized words.
        positive_words (set): Set of positive words.
        negative_words (set): Set of negative words.

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
            if check_negation(input_words, i):
                negative_count -= 1
                logger.debug(f"Positive word '{word}' at position {i} negated.")
            else:
                positive_count += 1
                logger.debug(f"Positive word found: {word} at position {i}.")

    sum_negative = -negative_count
    polarity_score = (positive_count - sum_negative) / word_count if word_count > 0 else np.nan
    legacy_score = combine_sent(positive_count, sum_negative)
    logger.debug(f"Polarity Score: {polarity_score}, Word Count: {word_count}, Sum Negative: {sum_negative}, Positive Count: {positive_count}, Legacy Score: {legacy_score}")
    return polarity_score, word_count, sum_negative, positive_count, legacy_score


def polarity_score_per_section(
    
    text_list: List[str],
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[float]]:
    """
    Generates the polarity score for the input text to identify the sentiment.

    Args:
        config (Config): Configuration object.
        text_list (List[str]): List of texts to analyze.

    Returns:
        Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[float]]:
            Polarity score, word count, sum of negatives, count of positives, legacy score.
    """
    positive_words = load_word_set(config, config.psycholinguistics.filecfg.positive_flnm)
    negative_words = load_word_set(config, config.psycholinguistics.filecfg.negative_flnm)

    cleaned_text, input_words, word_count = preprocess_text(text_list)

    if cleaned_text and word_count > 1:
        polarity_score, word_count, sum_negative, positive_count, legacy_score = calculate_polarity_score(
            input_words, positive_words, negative_words
        )
        return polarity_score, word_count, sum_negative, positive_count, legacy_score
    else:
        logger.warning("Insufficient data for polarity score calculation.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

def polarity_score_per_sentence(
    
    text_list: List[str],
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """
    Analyzes sentences to calculate polarity scores.

    Args:
        config (Config): Configuration object.
        text_list (List[str]): List of sentences to analyze.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
            Word counts, positive word counts, negative word counts per sentence.
    """
    # TODO: decide how to hancle nlp variable here
    nlp = None

    # TODO: Load postyive and negative words list
    positive_words = load_word_set( config, config.psycholinguistics.filecfg.vocab_pos_flnm)
    negative_words = load_word_set( config, config.psycholinguistics.filecfg.vocab_neg_flnm)

    _, input_words_list, word_count_list = preprocess_text_list(text_list, nlp)

    if text_list and word_count_list:
        word_counts = []
        positive_counts = []
        negative_counts = []

        for input_words in input_words_list:
            polarity, wc, sum_neg, pos_count, _ = calculate_polarity_score(
                input_words, positive_words, negative_words
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
        'es': [2, 1],
        'ing': [3],
        'ed': [2, 1]
    }

    for suffix, strip_lengths in suffix_rules.items():
        if word.endswith(suffix):
            for strip_length in strip_lengths:
                root = word[:-strip_length]
                if root in syllables and syllables[root] > 2:
                    return True

    if syllables.get(word, 0) > 2:
        return True

    return False


def fog_analysis_per_section(
    
    text_list: List[str],
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
    """
    Calculates the Fog Index for the input text to evaluate readability.

    Args:
        config (Config): Configuration object.
        text_list (List[str]): List of texts to analyze.

    Returns:
        Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
            Fog index, complex word count, average words per sentence, total word count.
    """

    # TODO: decide how to hancle nlp variable here
    nlp = None
    syllables = load_syllable_counts( config, config.psycholinguistics.filecfg.syllable_flnm)

    raw_text = ' '.join(text_list) if isinstance(text_list, list) else text_list
    total_word_count = len(raw_text.split())
    sentences = raw_text.split('. ')
    average_words_per_sentence = np.mean([len(sentence.strip().split()) for sentence in sentences]) if sentences else 0

    cleaned_text, input_words, word_count = preprocess_text(text_list, nlp)

    if cleaned_text and word_count > 1:
        complex_word_count = sum(is_complex(word, syllables) for word in input_words)
        fog_index = 0.4 * (average_words_per_sentence + 100 * (complex_word_count / total_word_count))
        logger.info(f"Fog Analysis - Fog Index: {fog_index}, Complex Words: {complex_word_count}, Average Words/Sentence: {average_words_per_sentence}, Total Words: {total_word_count}")
        return (fog_index, complex_word_count, average_words_per_sentence, total_word_count)
    else:
        logger.warning("Insufficient data for Fog Analysis.")
        return (np.nan, np.nan, np.nan, np.nan)


def fog_analysis_per_sentence(
    
    text_list: List[str],
) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
    """
    Calculates the Fog Index for each sentence in the input list.

    Args:
        config (Config): Configuration object.
        text_list (List[str]): List of sentences to analyze.

    Returns:
        Tuple[Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
            Fog index list, complex word count list, total word count list.
    """
    # TODO: decide how to hancle nlp variable here
    nlp = None
    syllables = load_syllable_counts( config, config.psycholinguistics.filecfg.syllable_flnm)

    word_count_list = [len(sentence.split()) for sentence in text_list]
    average_words_per_sentence = np.mean(word_count_list) if text_list else 0

    _, input_words_list, _ = preprocess_text_list(text_list, nlp)

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


def tone_count_with_negation_check(self, text_list):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    Simple negation is taken to be observations of one of negate words occurring within three words
    preceding a positive words.
    Parameters:
    argument1 (list): text list

    Returns:
    array, int:returns polarity_scores, word_count, negative_word_count_list, positive_word_count_list, legacy_scores

    """ 

    polarity_scores = []
    legacy_scores = []
    word_count = []
    negative_word_count_list = []
    positive_word_count_list = []

    sentiment_metrics = polarity_score_per_section(text_list)

    polarity_scores.append(sentiment_metrics[0])
    word_count.append(sentiment_metrics[1])
    negative_word_count_list.append(sentiment_metrics[2])
    positive_word_count_list.append(sentiment_metrics[3])
    legacy_scores.append(sentiment_metrics[4])

    return (polarity_scores, word_count, negative_word_count_list, positive_word_count_list, legacy_scores)
    
def tone_count_with_negation_check_per_sentence(self, text_list):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    Simple negation is taken to be observations of one of negate words occurring within three words
    preceding a positive words.
    Parameters:
    argument1 (list): text list

    Returns:
    tuple:(polarity scores(float), word count(int), negative word count list(list), positive word count list(list), legacy scores(double)))

    """ 

    word_count = []
    positive_word_count_list_per_sentence=[]
    negative_word_count_list_per_sentence=[]
    sentiment_metrics = polarity_score_per_sentence(text_list)
    word_count.append(sentiment_metrics[0])
    positive_word_count_list_per_sentence.append(sentiment_metrics[1])
    negative_word_count_list_per_sentence.append(sentiment_metrics[2])

    return (word_count, positive_word_count_list_per_sentence,negative_word_count_list_per_sentence)



def get_match_set(matches: List[str]) -> Dict[str, set]:
    """
    Generates the match set including unigrams, bigrams, and phrases.

    Args:
        matches (List[str]): List of matched words or phrases.

    Returns:
        Dict[str, set]: Dictionary containing original matches, unigrams, bigrams, and phrases.
    """
    
    bigrams = set(
        [
            word.lower()
            for word in matches
            if len(word.split("_")) == 2
        ]
        + [
            word.lower().replace(" ", "_")
            for word in matches
            if len(word.split(" ")) == 2
        ]
        + [
            "_".join(tokenize_matched_words(word))
            for word in matches
            if len(word.split(" ")) == 2
        ]
    )

    unigrams = set(
        [
            tokenize_matched_words(match)[0]
            for match in matches
            if "_" not in match and len(match.split(" ")) == 1
        ]
        + [
            match.lower()
            for match in matches
            if "_" not in match and len(match.split(" ")) == 1
        ]
    )

    phrases = [phrase.lower() for phrase in matches if len(phrase.split(" ")) > 2]

    logger.debug(f"Generated match sets: {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(phrases)} phrases.")
    return {
        "original": set(matches),
        "unigrams": unigrams,
        "bigrams": bigrams,
        "phrases": set(phrases),
    }


def match_count(
    text: str,
    match_sets: Dict[str, Dict[str, set]],
    phrases: bool = True
) -> Dict[str, Any]:
    """
    Generates the count dictionary with matched count of unigrams, bigrams, and phrases.

    Args:
        text (str): The text to analyze.
        match_sets (Dict[str, Dict[str, set]]): Dictionary containing labeled sets of unigrams, bigrams, and phrases.
        phrases (bool): Whether to count phrases.

    Returns:
        Dict[str, Any]: Dictionary containing counts and statistics.
    """
    unigrams = word_tokenizer(text)
    bigrams = ["_".join(g) for g in find_ngrams(unigrams, 2)]

    # Initialize count dictionaries per label
    count_dict = {
        label: {match: 0 for match in match_set['unigrams'].union(match_set['bigrams'])}
        for label, match_set in match_sets.items()
    }

    unigram_count = {label: 0 for label in match_sets.keys()}
    bigram_count = {label: 0 for label in match_sets.keys()}

    if phrases:
        phrase_count = {label: 0 for label in match_sets.keys()}
    total_count = {label: 0 for label in match_sets.keys()}

    # Iterate over each label and its corresponding match sets
    for label, match_set in match_sets.items():
        # Count unigrams
        for word in unigrams:
            if word in match_set['unigrams']:
                count_dict[label][word] += 1
                unigram_count[label] += 1

        # Count bigrams
        for bigram in bigrams:
            if bigram in match_set['bigrams']:
                count_dict[label][bigram] += 1
                bigram_count[label] += 1

        # Count phrases as presence (1 if present, else 0)
        if phrases:
            for phrase in match_set.get('phrases', set()):
                if phrase.lower() in text.lower():
                    phrase_count[label] += 1
            # Total count per label
            total_count[label] = unigram_count[label] + bigram_count[label] + phrase_count[label]
        else:
            # Total count without phrases
            total_count[label] = unigram_count[label] + bigram_count[label]

    # Construct the return dictionary
    if phrases:
        ret = {
            label: {
                'uni': unigram_count[label],
                'bi': bigram_count[label],
                'phrase': phrase_count[label],
                'total': total_count[label],
                'stats': count_dict[label]
            }
            for label in match_sets.keys()
        }
    else:
        ret = {
            label: {
                'uni': unigram_count[label],
                'bi': bigram_count[label],
                'total': total_count[label],
                'stats': count_dict[label]
            }
            for label in match_sets.keys()
        }

    # Add additional metadata
    ret['len'] = len(unigrams)
    ret['raw_len'] = len(text.split(' '))
    ret['filt'] = text  # Optional: Remove if not necessary

    logger.debug(f"Match counts: {ret}")

    return ret

def merge_counts(counts: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Merges multiple count dictionaries into a single dictionary.

    Args:
        counts (List[Dict[str, int]]): List of count dictionaries.

    Returns:
        Dict[str, int]: Merged count dictionary.
    """
    try:
        merged = Counter()
        for count in counts:
            merged += Counter(count)
        if not merged:
            return {"NO_MATCH": 1}
        return merged #dict(merged)
    except Exception as e:
        logger.error(f"Error merging counts: {e}")
        return {"ERROR": 1}
    

def calculate_sentence_score(
    a: List[int],
    b: List[int],
    weight: bool = True
) -> Optional[float]:
    """
    Calculates the sentence score based on provided lists.

    Args:
        a (List[int]): List of relevant sentence indicators.
        b (List[int]): List of weights or counts.
        weight (bool): Whether to apply weighting.

    Returns:
        Optional[float]: Calculated sentence score or None.
    """
    length = len(a)
    if length == 0 or length != len(b):
        return None

    num_relevant = len([x for x in a if x > 0])
    if num_relevant == 0:
        return None

    if weight:
        score = np.dot(a, b) / num_relevant
    else:
        binary_a = [1 if x > 0 else 0 for x in a]
        score = np.dot(binary_a, b) / num_relevant

    logger.debug(f"Calculated sentence score: {score}")
    return score

def netscore(a: List[int], b: List[int]) -> Optional[float]:
    """
    Calculates the net score based on provided lists.

    Args:
        a (List[int]): List of counts.
        b (List[int]): List of indicators.

    Returns:
        Optional[float]: Calculated net score or None.
    """
    length = len(a)
    if length == 0 or length != len(b):
        return None

    num_relevant = len([x for x in a if x > 0])
    if num_relevant == 0:
        return None

    score = np.dot([1 if x > 0 else 0 for x in a], b)
    logger.debug(f"Calculated net score: {score}")
    return score


def generate_match_count(
    df: Any,
    word_set_dict: Dict[str, Dict[str, set]]    
) -> Any:
    """
    Generates the match count using the topics data.

    Args:
        df (Any): DataFrame to process.
        word_set_dict (Dict[str, Dict[str, set]]): Dictionary of word sets for each label.
        config (Config): Configuration object.

    Returns:
        Any: Updated DataFrame with match counts.
    """
    for label in config.FILT_sections:
        match_set = word_set_dict.get(label, {})
        df[f"matches_{label}"] = df[label].apply(
            lambda x: [match_count(sent, match_set, phrases=False) for sent in x],
            meta=(f"matches_{label}", "object")
        )
    logger.info("Generated match counts for all sections.")
    return df

def generate_topic_statistics(
    df: Any,
    word_set_dict: Dict[str, Dict[str, set]],
    
) -> Any:
    """
    Generates new columns with topic total count statistics.

    Args:
        df (Any): DataFrame to process.
        word_set_dict (Dict[str, Dict[str, set]]): Dictionary of word sets for each label.
        config (Config): Configuration object.

    Returns:
        Any: Updated DataFrame with topic statistics.
    """
    for label in config.FILT_sections:
        df[f"LEN_{label}"] = df[f"matches_{label}"].apply(
            lambda x: [calc['len'] for calc in x]
        )

        df[f"RAW_LEN_{label}"] = df[f"matches_{label}"].apply(
            lambda x: [calc['raw_len'] for calc in x]
        )

        for topic in word_set_dict.keys():
            df[f"{topic}_TOTAL_{label}"] = df[f"matches_{label}"].apply(
                lambda x: [calc[topic]['total'] for calc in x]
            )
            df[f"{topic}_STATS_{label}"] = df[f"matches_{label}"].apply(
                lambda x: merge_counts([calc[topic]['stats'] for calc in x])
            )
            df[f"{topic}_STATS_LIST_{label}"] = df[f"matches_{label}"].apply(
                lambda x: [dict(calc[topic]['stats']) for calc in x]
            )
        df[f"NUM_SENTS_{label}"] = df[f"LEN_{label}"].apply(lambda x: len(x))

        df.drop(columns=[f"matches_{label}"], inplace=True)
    logger.info("Generated topic statistics for all sections.")
    return df


def generate_sentence_relevance_score(
    df: Any,
    word_set_dict: Dict[str, Dict[str, set]],
    
) -> Any:
    """
    Generates relevance scores for each sentence.

    Args:
        df (Any): DataFrame to process.
        word_set_dict (Dict[str, Dict[str, set]]): Dictionary of word sets for each label.
        config (Config): Configuration object.

    Returns:
        Any: Updated DataFrame with sentence relevance scores.
    """
    for label in config.FILT_sections:
        df[f"SENT_{label}"] = df[f"SENT_LABELS_{label}"].apply(
            lambda x: float(np.sum(x) / len(x)) if len(x) > 0 else None
        )
        df[f"NET_SENT_{label}"] = df[f"SENT_LABELS_{label}"].apply(
            lambda x: np.sum(x) if len(x) > 0 else None
        )

        for topic in word_set_dict.keys():
            df[f"{topic}_RELEVANCE_{label}"] = df[f"{topic}_TOTAL_{label}"].apply(
                lambda x: len([a for a in x if a > 0]) / len(x) if len(x) > 0 else None
            )
            df[f"{topic}_SENT_{label}"] = df.apply(
                lambda row: calculate_sentence_score(
                    row[f"{topic}_TOTAL_{label}"],
                    row[f"SENT_LABELS_{label}"],
                    weight=False
                ),
                axis=1
            )

            df[f"{topic}_SENT_REL_{label}"] = df.apply(
                lambda row: float(row[f"{topic}_RELEVANCE_{label}"] * row[f"{topic}_SENT_{label}"]) if row[f"{topic}_SENT_{label}"] else None,
                axis=1
            )

            df[f"{topic}_SENT_WEIGHT_{label}"] = df.apply(
                lambda row: calculate_sentence_score(
                    [sum(1 for val in stat.values() if val > 0) for stat in row[f"{topic}_STATS_LIST_{label}"]],
                    row[f"SENT_LABELS_{label}"],
                    weight=True
                ),
                axis=1
            )
            df[f"{topic}_SENT_WEIGHT_REL_{label}"] = df.apply(
                lambda row: float(row[f"{topic}_RELEVANCE_{label}"] * row[f"{topic}_SENT_WEIGHT_{label}"]) if row[f"{topic}_SENT_WEIGHT_{label}"] else None,
                axis=1
            )

            df[f"{topic}_NET_SENT_{label}"] = df.apply(
                lambda row: netscore(row[f"{topic}_TOTAL_{label}"], row[f"SENT_LABELS_{label}"]),
                axis=1
            )
    logger.info("Generated sentence relevance scores for all sections.")
    return df


def find_nearest_words_with_embeddings(words, model, num_neigh=50, filename=False, regularize=False):
    alist = {'label': [], 'embed': [], 'match': [], 'sim': []}
    for topic in set(words['label']):
        topic_embed = [[word[0], model.wv[word[0]], word[1]] for word in model.wv.most_similar_cosmul(
            positive=words[words['label'] == topic]['match'].apply(lambda x: [y for y in get_model_ngrams(x, model) if y in model.wv] if x not in model.wv else [x]).sum(),
            topn=num_neigh
        )]
        topic_embed_norm = [[word[0], model.wv[word[0]], word[1]] for word in model.wv.most_similar(
            positive=words[words['label'] == topic]['match'].apply(lambda x: [y for y in get_model_ngrams(x, model) if y in model.wv] if x not in model.wv else [x]).sum(),
            topn=num_neigh
        )]

        alist['label'] += [topic] * num_neigh
        if regularize:
            alist['embed'] += [embed[1] for embed in topic_embed]
            alist['match'] += [word[0] for word in topic_embed]
            alist['sim'] += [word[2] for word in topic_embed]
        else:
            alist['embed'] += [embed[1] for embed in topic_embed_norm]
            alist['match'] += [word[0] for word in topic_embed_norm]
            alist['sim'] += [word[2] for word in topic_embed_norm]

    tdf = pd.DataFrame(alist)
    if filename:
        # Save to JSON
        tdf.to_json(f"{filename}_neighbors_n{num_neigh}.json", orient="records")
    return tdf