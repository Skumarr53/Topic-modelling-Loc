import json
from typing import List
from pyspark import DataFrame
from loguru import logger

def parse_json_list(s):
    
    try:
        return json.loads(s)
    except Exception as e:
        logger.error(f"JSON parsing error: {e} for input: {s}")
        return []

def create_text_pairs(filt, labels):
    
    text_pairs = []
    for t in filt:
        for l in labels:
            text_pairs.append(f"{t}</s></s>{l}.")
    return text_pairs


def generate_label_columns(
    labels: List[str],
    metrics: List[str],
    sec_filters: List[str] = ['FILT_MD', 'FILT_QA'],
    include_extract: bool = False
) -> List[str]:
    """
    Generates a list of column names based on the provided categories, metrics, and filters.
    
    Parameters:
    - fixed_columns (List[str]): Columns that are always included.
    - categories (List[str]): Categories such as 'consumer strength', 'consumer weakness', etc.
    - metrics (List[str]): Metrics like 'COUNT', 'REL', 'SCORE', etc.
    - filters (List[str]): Filters like 'FILT_MD', 'FILT_QA'.
    - include_extract (bool): Whether to include columns with '_EXTRACT_'.
    
    Returns:
    - List[str]: Ordered list of column names.
    """
    dynamic_columns = []
    
    for category in labels:
        for filter_ in sec_filters:
            for metric in metrics:
                # Base column
                column = f"{category}_{metric}_{filter_}"
                dynamic_columns.append(column)
                
                # If include_extract is True, add the _EXTRACT_ version
            if include_extract:
                extract_column = f"{category}_EXTRACT_{filter_}"
                dynamic_columns.append(extract_column)
    
    return  dynamic_columns