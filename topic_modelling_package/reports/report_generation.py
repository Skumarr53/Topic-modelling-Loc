# topic_modelling_package/reports/report_generation.py

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from loguru import logger

from topic_modelling_package.processing.match_operations import create_match_patterns, count_matches_in_single_sentence
from centralized_nlp_package.utils.helper import df_apply_transformations
from topic_modelling_package.reports import STATISTICS_MAP


def create_topic_dict(
    match_df: pd.DataFrame,
    label_col: str = 'label',
    negate_col: str = 'negate',
    match_col: str = 'match',
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """
    Creates dictionaries for positive and negative match patterns per topic.

    This function generates two dictionaries:
    1. `word_set_dict`: Contains positive match sets for each formatted topic.
    2. `negate_dict`: Contains lists of negative matches for each formatted topic.

    Args:
        match_df (pd.DataFrame): DataFrame containing the data with columns for labels, negation, and matches.
        label_col (str, optional): Name of the column representing labels/topics. Defaults to 'label'.
        negate_col (str, optional): Name of the column indicating negation (True/False). Defaults to 'negate'.
        match_col (str, optional): Name of the column containing match strings. Defaults to 'match'.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
            - word_set_dict: Dictionary containing positive match sets for each formatted topic.
            - negate_dict: Dictionary containing negative match lists for each formatted topic.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'label': ['Positive', 'Positive', 'Negative'],
        ...     'negate': [False, True, False],
        ...     'match': ['Good', 'Not Good', 'Bad']
        ... }
        >>> df = pd.DataFrame(data)
        >>> word_set, negate = create_topic_dict(df)
        >>> print(word_set)
        {
            'POSITIVE': {
                'original': ['Good'],
                'unigrams': {'good'},
                'bigrams': set(),
                'phrases': []
            }
        }
        >>> print(negate)
        {
            'POSITIVE': ['not good'],
            'NEGATIVE': []
        }
    """
    unique_labels = match_df[label_col].unique()
    word_set_dict: Dict[str, Dict[str, Any]] = {}
    negate_dict: Dict[str, List[str]] = {}

    for topic in unique_labels:
        formatted_topic = topic.replace(' ', '_').upper()
        # Create word_set_dict
        positive_matches = match_df[
            (match_df[label_col] == topic) & (match_df[negate_col] == False)
        ][match_col].values
        word_set_dict[formatted_topic] = create_match_patterns(list(positive_matches))
        
        # Create negate_dict
        negative_matches = match_df[
            (match_df[label_col] == topic) & (match_df[negate_col] == True)
        ][match_col].values.tolist()
        negate_dict[formatted_topic] = [word.lower() for word in negative_matches]

        logger.debug(
            f"Processed topic '{formatted_topic}': "
            f"{len(positive_matches)} positive matches, "
            f"{len(negative_matches)} negative matches."
        )

    logger.info("Created topic dictionaries successfully.")
    return word_set_dict, negate_dict


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

    Example:
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
    transformed_dict: Dict[str, List[str]] = {k: [] for k in input_dict}

    for key, words in input_dict.items():
        for word in words:
            parts = word.split(split_char)
            if len(parts) == required_splits:
                transformed_word = separator.join(parts)
                transformed_dict[key].append(transformed_word)
                logger.debug(f"Transformed '{word}' to '{transformed_word}' in category '{key}'.")
            else:
                transformed_dict[key].append(word)
                logger.debug(f"No transformation for '{word}' in category '{key}'.")
    
    logger.info("Replaced separators in dictionary words successfully.")
    return transformed_dict


def generate_topic_report(
    df: pd.DataFrame,
    word_set_dict: Dict[str, Any],
    negate_dict: Dict[str, List[str]],
    stats_list: List[str],
    label_column: str = "matches",
) -> pd.DataFrame:
    """
    Generates topic-specific columns for selected statistics.

    This function applies transformations to the DataFrame based on the provided statistics list.
    Supported statistics include:
        - 'total': Total counts of matches.
        - 'stats': Detailed statistics of matches.
        - 'relevance': Relevance scores.
        - 'count': Count of matches.
        - 'extract': Extracted matches.
        - 'sentiment': Sentiment analysis results.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.
        word_set_dict (Dict[str, Any]): Dictionary of word sets for different topics.
        negate_dict (Dict[str, List[str]]): Dictionary for negation handling.
        stats_list (List[str]): List of statistic identifiers to compute. Supported:
            ['total', 'stats', 'relevance', 'count', 'extract', 'sentiment']
        label_column (str, optional): Prefix for match labels in the DataFrame. Defaults to "matches".

    Returns:
        pd.DataFrame: Updated DataFrame with additional report columns for each topic and selected statistics.
    
    Raises:
        ValueError: If an unsupported statistic identifier is provided in stats_list.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'matches_FILT_MD': [['good', 'bad'], ['good']],
        ...     'matches_FILT_QA': [['excellent', 'poor'], ['average']],
        ...     'FILT_MD': ['some text', 'other text'],
        ...     'FILT_QA': ['additional text', 'more text']
        ... }
        >>> df = pd.DataFrame(data)
        >>> word_set_dict = {
        ...     'POSITIVE': {'original': ['good'], 'unigrams': {'good'}, 'bigrams': {'good_service'}, 'phrases': []},
        ...     'NEGATIVE': {'original': ['bad'], 'unigrams': {'bad'}, 'bigrams': {'bad_service'}, 'phrases': []}
        ... }
        >>> negate_dict = {
        ...     'POSITIVE': ['not good'],
        ...     'NEGATIVE': ['not bad']
        ... }
        >>> stats_list = ['total', 'count']
        >>> report_df = generate_topic_report(df, word_set_dict, negate_dict, stats_list)
        >>> print(report_df.columns)
        Index(['matches_FILT_MD', 'matches_FILT_QA', 'FILT_MD', 'FILT_QA', 
               'POSITIVE_TOTAL_FILT_MD', 'POSITIVE_COUNT_FILT_MD', 
               'NEGATIVE_TOTAL_FILT_MD', 'NEGATIVE_COUNT_FILT_MD', 
               'POSITIVE_TOTAL_FILT_QA', 'POSITIVE_COUNT_FILT_QA', 
               'NEGATIVE_TOTAL_FILT_QA', 'NEGATIVE_COUNT_FILT_QA'], 
              dtype='object')
    """
    # Validate stats_list
    unsupported_stats = set(stats_list) - set(STATISTICS_MAP.keys())
    if unsupported_stats:
        raise ValueError(f"Unsupported statistics requested: {unsupported_stats}")

    # Initial transformations: match counts
    labels = ["FILT_MD", "FILT_QA"]
    lab_sec_dict1 = [
        (f"{label_column}_{lab}", lab, lambda x: count_matches_in_single_sentence(x, word_set_dict, suppress=negate_dict))
        for lab in labels
    ]

    logger.info("Applying initial match count transformations.")
    df = df_apply_transformations(df, lab_sec_dict1)

    # Iterate over labels and topics to apply selected statistics
    for label in labels:
        for topic in word_set_dict.keys():
            lab_sec_dict2 = []
            for stat in stats_list:
                transformation_func = STATISTICS_MAP.get(stat)
                if transformation_func:
                    lab_sec_dict2.append(transformation_func(topic, label, label_column))
                else:
                    logger.warning(f"Statistic '{stat}' not found in STATISTICS_MAP.")
            if lab_sec_dict2:
                logger.info(f"Applying transformations for topic '{topic}' and label '{label}'.")
                df = df_apply_transformations(df, lab_sec_dict2)

    # Drop intermediate match columns
    intermediate_cols = [f"{label_column}_{label}" for label in labels]
    df.drop(columns=intermediate_cols, inplace=True, errors='ignore')
    logger.info(f"Dropped intermediate match columns: {intermediate_cols}")

    return df


def generate_top_matches_report(
    df: pd.DataFrame,
    topic: str,
    label: str,
    sort_by: str,
    top_n: int = 100,
    meta_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generates a report based on top matches for a given topic.

    This function sorts the DataFrame based on a specified column, selects the top N matches,
    and retains only the relevant metadata and match information.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        topic (str): Topic name for which the report is generated.
        label (str): Label indicating which section (e.g., FILT_MD, FILT_QA).
        sort_by (str): Column to sort by when selecting top matches.
        top_n (int, optional): Number of rows to include in the report. Defaults to 100.
        meta_cols (Optional[List[str]], optional): List of metadata columns to retain. 
            Defaults to ['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC', 'COMPANY_NAME'].

    Returns:
        pd.DataFrame: DataFrame containing top matches for the specified topic.

    Example:
        >>> data = {
        ...     'ENTITY_ID': [1, 2, 3],
        ...     'CALL_NAME': ['Call1', 'Call2', 'Call3'],
        ...     'EVENT_DATETIME_UTC': ['2021-01-01', '2021-01-02', '2021-01-03'],
        ...     'COMPANY_NAME': ['CompanyA', 'CompanyB', 'CompanyC'],
        ...     'score': [0.95, 0.85, 0.75],
        ...     'POSITIVE_STATS_FIL_MD': [5, 3, 1],
        ...     'POSITIVE_TOTAL_FIL_MD': [5, 3, 1]
        ... }
        >>> df = pd.DataFrame(data)
        >>> report = generate_top_matches_report(df, 'POSITIVE', 'FIL_MD', 'score', top_n=2)
        >>> print(report)
           ENTITY_ID CALL_NAME EVENT_DATETIME_UTC COMPANY_NAME  score  POSITIVE_STATS_FIL_MD  POSITIVE_TOTAL_FIL_MD
        0          1     Call1           2021-01-01     CompanyA   0.95                     5                     5
        1          2     Call2           2021-01-02     CompanyB   0.85                     3                     3
    """
    if meta_cols is None:
        meta_cols = ['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC', 'COMPANY_NAME']

    required_columns = meta_cols + [sort_by, f"{topic}_STATS_{label}", f"{topic}_TOTAL_{label}"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.warning(f"The following required columns are missing in the DataFrame: {missing_columns}")

    # Filter the DataFrame to include only the required columns that exist
    existing_columns = [col for col in required_columns if col in df.columns]
    if not existing_columns:
        logger.error("No required columns found in DataFrame. Cannot generate report.")
        return pd.DataFrame()  # Return empty DataFrame

    # Sort the DataFrame based on the 'sort_by' column in descending order
    sorted_df = df.sort_values(by=sort_by, ascending=False)

    # Drop duplicates based on 'ENTITY_ID' to retain the top match per entity
    sorted_df = sorted_df.drop_duplicates(subset='ENTITY_ID')

    # Select the top N matches
    top_matches_df = sorted_df.head(top_n)

    # Select only the existing columns
    report_df = top_matches_df[existing_columns].copy()

    # Drop rows with NaN in the 'sort_by' column
    report_df = report_df.dropna(subset=[sort_by])

    logger.info(f"Generated top {top_n} matches report for topic '{topic}' and label '{label}'.")
    return report_df
