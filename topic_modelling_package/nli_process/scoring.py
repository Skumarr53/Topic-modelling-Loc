# topic_modelling_package/nli_process/scoring.py
import re
from typing import List, Dict, Any, Tuple
import json

from pyspark.sql import DataFrame, functions as F, types as T
from loguru import logger

from centralized_nlp_package.data_processing import create_spark_udf


def get_matched_sentences(sentences: List[str], matches: List[int]) -> List[str]:
    """
    Extracts sentences that have a corresponding match flag set to 1.

    Args:
        sentences (List[str]): A list of sentences.
        matches (List[int]): A list of match flags (1 or 0) corresponding to each sentence.

    Returns:
        List[str]: A list of sentences where the corresponding match flag is 1.

    Example:
        >>> sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
        >>> matches = [1, 0, 1]
        >>> get_matched_sentences(sentences, matches)
        ['Sentence one.', 'Sentence three.']
    """
    if not sentences or not matches:
        logger.warning("Empty sentences or matches list provided.")
        return []

    if len(sentences) != len(matches):
        logger.error("Length mismatch between sentences and matches.")
        raise ValueError("The length of sentences and matches must be equal.")

    matched_sentences = [sentence for sentence, match in zip(sentences, matches) if match == 1]
    logger.debug(f"Extracted {len(matched_sentences)} matched sentences out of {len(sentences)}.")
    return matched_sentences


def compute_section_metrics(
    row: Dict[str, List[float]],
    section: str,
    section_len: int,
    threshold: float
) -> Dict[str, Any]:
    """
    Computes metrics for a given section based on scores and a threshold.

    Args:
        row (Dict[str, List[float]]): A dictionary where keys are labels and values are lists of scores.
        section (str): The section identifier (e.g., 'FILT_MD', 'FILT_QA').
        section_len (int): The total number of items in the section.
        threshold (float): The score threshold to determine binary flags.

    Returns:
        Dict[str, Any]: A dictionary containing count, relation, total, and score metrics for each label.

    Example:
        >>> row = {'positive.': [0.9, 0.85], 'negative.': [0.4, 0.3]}
        >>> section = 'FILT_MD'
        >>> section_len = 2
        >>> threshold = 0.8
        >>> compute_section_metrics(row, section, section_len, threshold)
        {
            'positive._COUNT_FILT_MD': 2.0,
            'positive._REL_FILT_MD': 1.0,
            'positive._SCORE_FILT_MD': [0.9, 0.85],
            'positive._TOTAL_FILT_MD': [1.0, 1.0],
            'negative._COUNT_FILT_MD': 0.0,
            'negative._REL_FILT_MD': 0.0,
            'negative._SCORE_FILT_MD': [0.4, 0.3],
            'negative._TOTAL_FILT_MD': [0.0, 0.0]
        }
    """
    count_col: Dict[str, float] = {}
    rel_col: Dict[str, float] = {}
    score_col: Dict[str, List[float]] = {}
    total_col: Dict[str, List[float]] = {}

    for label, scores in row.items():
        if section_len != 0:
            score_binary = [1.0 if score > threshold else 0.0 for score in scores]
            total_col[f'{label}_TOTAL_{section}'] = score_binary
            count_col[f'{label}_COUNT_{section}'] = float(sum(score_binary))
            rel_col[f'{label}_REL_{section}'] = sum(score_binary) / section_len
            score_col[f'{label}_SCORE_{section}'] = [round(score, 4) for score in scores]
            logger.debug(
                f"Label: {label}, Scores: {scores}, Binary Flags: {score_binary}, "
                f"Count: {count_col[f'{label}_COUNT_{section}']}, "
                f"Relation: {rel_col[f'{label}_REL_{section}']}"
            )
        else:
            count_col[f'{label}_COUNT_{section}'] = 0.0
            rel_col[f'{label}_REL_{section}'] = 0.0
            total_col[f'{label}_TOTAL_{section}'] = []
            score_col[f'{label}_SCORE_{section}'] = []
            logger.debug(
                f"Label: {label}, Section length is zero. "
                f"Set count and relation to 0, and scores to empty lists."
            )

    metrics = {**count_col, **rel_col, **score_col, **total_col}
    logger.debug(f"Computed metrics for section '{section}': {metrics}")
    return metrics


def add_extracted_scores_columns(
    df: DataFrame,
    patterns: List[str],
    original_suffix: str = 'TOTAL',
    new_suffix: str = 'EXTRACT'
) -> DataFrame:
    """
    Extracts matched sentences from specified columns based on patterns and adds new columns with a new suffix.

    Args:
        df (DataFrame): The input Spark DataFrame.
        patterns (List[str]): A list of regex patterns to match column names.
        new_suffix (str): The suffix to append to new columns.
        original_suffix (str): The original suffix to replace in matched column names.

    Returns:
        DataFrame: The DataFrame with new columns added.

    Raises:
        Exception: If any error occurs during the extraction process.

    Example:
        >>> patterns = [r'.*_TOTAL_FILT_MD$', r'.*_TOTAL_FILT_QA$']
        >>> new_suffix = 'EXTRACT'
        >>> original_suffix = 'TOTAL'
        >>> df = add_extracted_scores_columns(df, patterns, new_suffix, original_suffix)
    """
    try:
        # Compile regex patterns
        compiled_patterns = [re.compile(pattern) for pattern in patterns]
        logger.debug(f"Compiled patterns: {compiled_patterns}")

        # Define the UDF
        extract_udf = create_spark_udf(get_matched_sentences, 'arr[str]')
        
        # Identify matched columns based on patterns
        matched_columns = [col for col in df.columns if any(pattern.match(col) for pattern in compiled_patterns)]
        logger.info(f"Matched columns for extraction: {matched_columns}")

        for col_name in matched_columns:
            # Determine the new column name by replacing the original suffix with the new suffix
            if original_suffix in col_name:
                new_col_name = col_name.replace(original_suffix, new_suffix)
                logger.debug(f"Creating new column '{new_col_name}' from '{col_name}'.")

                # Apply the UDF to create the new column
                if 'MD' == col_name: section = 'FILT_MD'
                elif 'QA' == col_name: section = 'FILT_QA'
                else:
                    raise ValueError("column name matching predefined sections")

                df = df.withColumn(new_col_name, extract_udf(F.col(section), F.col(col_name)))
                logger.debug(f"Added column '{new_col_name}' to DataFrame.")
            else:
                logger.warning(f"Original suffix '{original_suffix}' not found in column '{col_name}'. Skipping replacement.")

        logger.info("Scores extracted and new columns added successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to extract scores: {e}")
        raise e
