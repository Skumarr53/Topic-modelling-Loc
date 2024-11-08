import pandas as pd
from typing import Dict, Any, List, Tuple
from topic_modelling_package.processing.match_operations import get_match_set, match_count_lowStat
from centralized_nlp_package.utils.helper import df_apply_transformations
from topic_modelling_package.utils.transformation import STATISTICS_MAP



def create_topic_dict(
    match_df: pd.DataFrame,
    label_col: str = 'label',
    negate_col: str = 'negate',
    match_col: str = 'match',
) -> Tuple[Dict[str, set], Dict[str, List[str]]]:
    """
    Creates two dictionaries: word_set_dict and negate_dict.

    Parameters:
    match_df : pd.DataFrame
        DataFrame containing the data with columns for labels, negation, and matches.
    label_col : str, optional
        Name of the column representing labels/topics. Default is 'label'.
    negate_col : str, optional
        Name of the column indicating negation (True/False). Default is 'negate'.
    match_col : str, optional
        Name of the column containing match strings. Default is 'match'.

    Returns:
    Tuple[Dict[str, set], Dict[str, List[str]]]
        - word_set_dict: Dictionary containing positive match sets for each formatted topic.
        - negate_dict: Dictionary containing negative match lists for each formatted topic.

    """
    unique_labels = match_df[label_col].unique()
    word_set_dict = {}
    negate_dict = {}

    for topic in unique_labels:
        formatted_topic = topic.replace(' ', '_').upper()
        # Create word_set_dict
        word_set_dict[formatted_topic] = get_match_set(
            match_df[(match_df[label_col] == topic) & (match_df[negate_col] == False)][match_col].values
        )
        # Create negate_dict
        negate_dict[formatted_topic] = [
            word.lower() for word in match_df[(match_df[label_col] == topic) & (match_df[negate_col] == True)][match_col].values.tolist()
        ]

    return word_set_dict, negate_dict


from typing import Dict, List

def replace_separator_in_dict_words(
    input_dict: Dict[str, List[str]],
    split_char: str = '_',
    separator: str = ' ',
    required_splits: int = 2
) -> Dict[str, List[str]]:
    """
    Transforms the input dictionary by replacing a specified separator in words that split into exactly a given number of parts.

    For each word in the lists associated with each key:
    - If the word contains exactly 'required_splits' parts when split by 'split_char',
      replace 'split_char' with 'separator'.
    - Otherwise, retain the word as is.

    Args:
        input_dict (Dict[str, List[str]]): 
            A dictionary with keys mapping to lists of words.
        split_char (str, optional):  
            The character used to split the words. Defaults to '_'.
        separator (str, optional):  
            The character used to join the split parts of the word. Defaults to ' '.
        required_splits (int, optional):  
            The exact number of parts a word must have after splitting to undergo transformation.  
            Defaults to 2.

    Returns:
        Dict[str, List[str]]: 
            A new dictionary with transformed words based on the specified conditions.

    Examples:
        >>> input_dict = {
        ...     'Category1': ['word_one', 'wordtwo', 'another_word'],
        ...     'Category2': ['simple', 'complex_here']
        ... }
        >>> transformed_dict = replace_separator_in_dict_words(input_dict)
        >>> print(transformed_dict)
        {
            'Category1': ['word one', 'wordtwo', 'another word'],
            'Category2': ['simple', 'complex here']
        }
    """
    transformed_dict = {k: [] for k in input_dict}

    for key, words in input_dict.items():
        for word in words:
            parts = word.split(split_char)
            if len(parts) == required_splits:
                transformed_word = separator.join(parts)
                transformed_dict[key].append(transformed_word)
            else:
                transformed_dict[key].append(word)

    return transformed_dict


def generate_topic_report(
    df: pd.DataFrame,
    word_set_dict: Dict[str, Any],
    negate_dict: Dict[str, Any],
    stats_list: List[str],
    label_column: str = "matches",
) -> pd.DataFrame:
    """
    Generates topic-specific columns for selected statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.
        word_set_dict (Dict[str, Any]): Dictionary of word sets for different topics.
        negate_dict (Dict[str, Any]): Dictionary for negation handling.
        stats_list (List[str]): List of statistic identifiers to compute. Supported:
            ['total', 'stats', 'relevance', 'count', 'extract', 'sentiment']
        label_column (str): Prefix for match labels in the DataFrame. Defaults to "matches".

    Returns:
        pd.DataFrame: Updated DataFrame with additional report columns for each topic and selected statistics.
    
    Raises:
        ValueError: If an unsupported statistic identifier is provided in stats_list.
    """
    # Validate stats_list
    unsupported_stats = set(stats_list) - set(STATISTICS_MAP.keys())
    if unsupported_stats:
        raise ValueError(f"Unsupported statistics requested: {unsupported_stats}")

    # Initial transformations: match counts
    labels = ["FILT_MD", "FILT_QA"]
    lab_sec_dict1 = [
        (f"{label_column}_{lab}", lab, lambda x: match_count_lowStat(x, word_set_dict, suppress=negate_dict))
        for lab in labels
    ]

    df = df_apply_transformations(df, lab_sec_dict1)

    # Iterate over labels and topics to apply selected statistics
    for label in labels:
        for topic in word_set_dict.keys():
            lab_sec_dict2 = []
            for stat in stats_list:
                transformation_func = STATISTICS_MAP[stat]
                lab_sec_dict2.append(transformation_func(topic, label, label_column))
            df = df_apply_transformations(df, lab_sec_dict2)

    # Drop intermediate match columns
    df.drop([f"{label_column}_{label}" for label in labels], axis=1, inplace=True)

    return df



def generate_top_matches_report(df: pd.DataFrame, topic: str, label: str, sort_by: str, top_n: int = 100, meta_cols: List[str] = ['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC', 'COMPANY_NAME']) -> pd.DataFrame:
    """
    Generates a report based on top matches for a given topic.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        topic (str): Topic name for which report is generated.
        label (str): Label indicating which section (e.g., FILT_MD, FILT_QA).
        sort_by (str): Column to sort by when selecting top matches.
        top_n (int): Number of rows to include in the report.

    Returns:
        pd.DataFrame: DataFrame containing top matches for the specified topic.
    """
    filtered_df = (df.sort_values(sort_by, ascending=False)[[*meta_cols, 
                                                             sort_by, topic + '_STATS_' + label, topic + '_TOTAL_' + label]]
                                                             .drop_duplicates('ENTITY_ID').head(top_n))
    return filtered_df.dropna(subset=[sort_by])
