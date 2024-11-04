import pandas as pd
from typing import Dict, Any, List
from topic_modelling_package.processing.match_operations import get_match_set, match_count_lowStat
from centralized_nlp_package.utils.helpers import df_apply_transformations
from topic_modelling_package.utils.transformation import STATISTICS_MAP



def create_topic_dict(match_df):
    """
    Creates two dictionaries: word_set_dict and negate_dict.

    Parameters:
    - match_df: pandas DataFrame containing columns 'label', 'negate', and 'match'

    Returns:
    - word_set_dict: Dictionary containing positive match sets for each topic.
    - negate_dict: Dictionary containing negative match lists for each topic.
    """
    unique_labels = match_df['label'].unique()
    word_set_dict = {}
    negate_dict = {}

    for topic in unique_labels:
        formatted_topic = topic.replace(' ', '_').upper()
        # Create word_set_dict
        word_set_dict[formatted_topic] = get_match_set(
            match_df[(match_df['label'] == topic) & (match_df['negate'] == False)]['match'].values
        )
        # Create negate_dict
        negate_dict[formatted_topic] = [
            word.lower() for word in match_df[(match_df['label'] == topic) & (match_df['negate'] == True)]['match'].values.tolist()
        ]

    return word_set_dict, negate_dict

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

def save_report_to_csv(df: pd.DataFrame, path: str):
    """
    Saves the report DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        path (str): File path to save the CSV report.
    """
    df.to_csv(path, index=False)
