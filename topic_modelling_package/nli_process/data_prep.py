# topic_modelling_package/nli_process/data_prep.py

import json
from pyexpat.errors import messages
from typing import List, Tuple, Union, Callable, Any
from pyspark.sql import DataFrame
#from loguru import logger


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
            print("Successfully parsed JSON string into list: {parsed_list}")
            return parsed_list
        else:
            print("JSON parsed but is not a list: {parsed_list}")
            return []
    except json.JSONDecodeError as e:
        print("JSON parsing error: {e} for input: {json_str}")
        return []
    except Exception as e:
        print("Unexpected error during JSON parsing: {e} for input: {json_str}")
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
            pair_dict = {"text": filter_item, "topic": label}
            text_pairs.append(pair_dict)
            print("Created text pair: {pair_dict}")
    print("Generated {len(text_pairs)} text pairs from filters and labels.")
    return text_pairs


def generate_label_columns(
    labels: List[str],
    metrics: List[str] = ['COUNT', 'REL', 'SCORE', 'TOTAL'],
    sec_filters: List[str] = ['FILT_MD', 'FILT_QA'],
) -> List[str]:
    """
    Generates a list of column names based on the provided labels, metrics, and secondary filters.

    The order of metrics should always follow: 'COUNT', 'REL', 'SCORE', 'TOTAL', and 'EXTRACT'.
    Some metrics may be omitted based on the use case, but the order must remain the same.

    Args:
        labels (List[str]): Labels such as 'consumer_strength', 'consumer_weakness', etc.
        metrics (List[str]): Metrics like 'COUNT', 'REL', 'SCORE', etc. Defaults to ['COUNT', 'REL', 'SCORE', 'TOTAL'].
        sec_filters (List[str], optional): Secondary filters like 'FILT_MD', 'FILT_QA'. Defaults to ['FILT_MD', 'FILT_QA'].

    Returns:
        List[str]: An ordered list of generated column names.

    Example:
        >>> labels = ['consumer_strength', 'consumer_weakness']
        >>> metrics = ['COUNT', 'REL']
        >>> generate_label_columns(labels, metrics)
        ['consumer_strength_COUNT_FILT_MD', 'consumer_strength_REL_FILT_MD',
         'consumer_strength_COUNT_FILT_QA', 'consumer_strength_REL_FILT_QA',
         'consumer_weakness_COUNT_FILT_MD', 'consumer_weakness_REL_FILT_MD',
         'consumer_weakness_COUNT_FILT_QA', 'consumer_weakness_REL_FILT_QA']
    """
    dynamic_columns = []
    for sec_filter in sec_filters:
        for label in labels:
            for metric in metrics:
                # Base column
                column_name = f"{label}_{metric}_{sec_filter}"
                dynamic_columns.append(column_name)
                print("Generated column name: {column_name}")
    print("Generated {len(dynamic_columns)} label columns.")
    return dynamic_columns

def add_non_entailment_rows(df, shuffle = True):
    """
    Adds 'non_entailment' rows for missing topics in each premise sentence.

    This function ensures that for every premise sentence (`sentence1`), there is a corresponding 'non_entailment' label for each possible topic (`sentence2`) that hasn't been labeled as 'entailment'. This is crucial for balanced datasets, especially in classification tasks where the absence of a label needs to be explicitly represented.

    Args:
        df (pd.DataFrame):
            The input DataFrame containing at least the following columns:
                - 'sentence1': The premise or source sentence.
                - 'sentence2': The hypothesis or target sentence.
                - 'label': The true label indicating entailment ('entailment') or non-entailment.
        shuffle (bool, optional):
            Whether to shuffle the DataFrame after adding the new rows. Defaults to True.

    Returns:
        pd.DataFrame:
            The updated DataFrame with additional 'non_entailment' rows for missing topics. If `shuffle` is True, the DataFrame is shuffled before being returned.

    Raises:
        None. The function assumes that `df` contains the required columns.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'sentence1': ["I love programming", "Python is great"],
        ...     'sentence2': ["programming is fun", "Python is easy", "I enjoy coding"],
        ...     'label': ["entailment", "entailment", "entailment"]
        ... }
        >>> df = pd.DataFrame(data)
        >>> updated_df = add_non_entailment_rows(df, shuffle=False)
        >>> print(updated_df)
              sentence1            sentence2          label
        0  I love programming  programming is fun    entailment
        1         Python is great      Python is easy    entailment
        2  I love programming      I enjoy coding    non_entailment
        3         Python is great  I enjoy coding    non_entailment
    """
    # Get all unique topics
    all_topics = df['sentence2'].unique()
    
    # Create a list to store new rows
    new_rows = []

    # Group by sentence1 to process each sentence individually
    grouped = df.groupby('sentence1')

    for sentence, group in grouped:
        # Get topics already labeled as 'entailment' for this sentence
        entailment_topics = group[group['label'] == 'entailment']['sentence2'].unique()
        
        # Determine missing topics
        missing_topics = set(all_topics) - set(entailment_topics)
        
        # Add a new row for each missing topic with 'non_entailment' label
        for topic in missing_topics:
            new_rows.append({'sentence1': sentence, 'sentence2': topic, 'label': 'non_entailment'})

    # Convert new_rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    
    # Concatenate the new rows with the original DataFrame
    df = pd.concat([df, new_rows_df], ignore_index=True)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df