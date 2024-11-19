# topic_modelling_package/nli_process/data_prep.py

import json
from typing import List, Tuple, Union, Callable, Any
from pyspark.sql import DataFrame
from loguru import logger


def parse_json_list(json_str: str) -> List[Any]:
    """
    Parses a JSON-formatted string into a Python list.

    Args:
        json_str (str): A JSON-formatted string representing a list.

    Returns:
        List[Any]: The parsed list from the JSON string. Returns an empty list if parsing fails.

    Example:
        >>> parse_json_list('["apple", "banana", "cherry"]')
        ['apple', 'banana', 'cherry']
        >>> parse_json_list('invalid json')
        []
    """
    try:
        parsed_list = json.loads(json_str)
        if isinstance(parsed_list, list):
            logger.debug(f"Successfully parsed JSON string into list: {parsed_list}")
            return parsed_list
        else:
            logger.error(f"JSON parsed but is not a list: {parsed_list}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e} for input: {json_str}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e} for input: {json_str}")
        return []


def create_text_pairs(filters: List[str], labels: List[str]) -> List[str]:
    """
    Creates text pairs by combining each filter with each label using a specific separator.

    Args:
        filters (List[str]): A list of filter strings.
        labels (List[str]): A list of label strings.

    Returns:
        List[str]: A list of combined text pairs in the format "filter</s></s>label.".

    Example:
        >>> create_text_pairs(['filter1', 'filter2'], ['labelA', 'labelB'])
        ['filter1</s></s>labelA.', 'filter1</s></s>labelB.', 'filter2</s></s>labelA.', 'filter2</s></s>labelB.']
    """
    text_pairs = []
    for filter_item in filters:
        for label in labels:
            combined_text = f"{filter_item}</s></s>{label}."
            text_pairs.append(combined_text)
            logger.debug(f"Created text pair: {combined_text}")
    logger.info(f"Generated {len(text_pairs)} text pairs from filters and labels.")
    return text_pairs


def generate_label_columns(
    labels: List[str],
    metrics: List[str] = ['COUNT', 'REL', 'SCORE', 'TOTAL'],
    sec_filters: List[str] = ['FILT_MD', 'FILT_QA'],
) -> List[str]:
    """
    Generates a list of column names based on the provided labels, metrics, and secondary filters.

    Args:
        labels (List[str]): Labels such as 'consumer_strength', 'consumer_weakness', etc.
        metrics (List[str]): Metrics like 'COUNT', 'REL', 'SCORE', etc.
        sec_filters (List[str], optional): Secondary filters like 'FILT_MD', 'FILT_QA'. Defaults to ['FILT_MD', 'FILT_QA'].
        include_extract (bool, optional): Whether to include columns with '_EXTRACT_'. Defaults to False.

    Returns:
        List[str]: An ordered list of generated column names.

    Example:
        >>> labels = ['consumer_strength', 'consumer_weakness']
        >>> metrics = ['COUNT', 'REL']
        >>> generate_label_columns(labels, metrics, include_extract=True)
        ['consumer_strength_COUNT_FILT_MD', 'consumer_strength_REL_FILT_MD', 'consumer_strength_EXTRACT_FILT_MD',
         'consumer_strength_COUNT_FILT_QA', 'consumer_strength_REL_FILT_QA', 'consumer_strength_EXTRACT_FILT_QA',
         'consumer_weakness_COUNT_FILT_MD', 'consumer_weakness_REL_FILT_MD', 'consumer_weakness_EXTRACT_FILT_MD',
         'consumer_weakness_COUNT_FILT_QA', 'consumer_weakness_REL_FILT_QA', 'consumer_weakness_EXTRACT_FILT_QA']
    """
    dynamic_columns = []

    for label in labels:
        for sec_filter in sec_filters:
            for metric in metrics:
                # Base column
                column_name = f"{label}_{metric}_{sec_filter}"
                dynamic_columns.append(column_name)
                logger.debug(f"Generated column name: {column_name}")

    logger.info(f"Generated {len(dynamic_columns)} label columns.")
    return dynamic_columns
