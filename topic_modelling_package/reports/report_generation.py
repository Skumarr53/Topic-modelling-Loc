import pandas as pd
# from centralized_nlp_package.text_processing.text_analysis import calculate_sentence_score
from topic_modelling_package.processing.match_operations import get_match_set
import gc



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

def generate_topic_report(df: pd.DataFrame, word_set_dict: dict, label_column: str = "matches_") -> pd.DataFrame:
    """
    Generates topic-specific columns for match counts, relevance, and sentiment.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.
        word_set_dict (dict): Dictionary of word sets for different topics.
        label_column (str): Prefix for match labels in the DataFrame.

    Returns:
        pd.DataFrame: Updated DataFrame with additional report columns for each topic.
    """
    for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():
        for topic in word_set_dict.keys():
            df[topic + '_TOTAL_' + label] = df[label_column + label].apply(lambda x: x[topic]['total'])
            df[topic + '_STATS_' + label] = df[label_column + label].apply(lambda x: x[topic]['stats'])
            
            # Calculate derived statistics
            df[topic + '_REL_' + label] = df[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a > 0]) / len(x) if len(x) > 0 else None)
            df[topic + '_COUNT_' + label] = df[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a > 0]) if len(x) > 0 else None)
            df[topic + '_SENT_' + label] = df[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: calculate_sentence_score(x[0], x[1], weight=False), axis=1)
    
    df.drop([label_column + label for label in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}], axis=1, inplace=True)

    return df

def generate_top_matches_report(df: pd.DataFrame, topic: str, label: str, sort_by: str, top_n: int = 100) -> pd.DataFrame:
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
    filtered_df = df.sort_values(sort_by, ascending=False)[['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC', 'COMPANY_NAME', sort_by, topic + '_STATS_' + label, topic + '_TOTAL_' + label]].drop_duplicates('ENTITY_ID').head(top_n)
    return filtered_df.dropna(subset=[sort_by])

def save_report_to_csv(df: pd.DataFrame, path: str):
    """
    Saves the report DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        path (str): File path to save the CSV report.
    """
    df.to_csv(path, index=False)
