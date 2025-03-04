# topic_modelling_package/processing/match_operations.py

import ast
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Optional
import spacy 

from centralized_nlp_package.text_processing import generate_ngrams
from centralized_nlp_package.text_processing import (
    tokenize_matched_words, 
    tokenize_and_lemmatize_text
)
from centralized_nlp_package.data_processing import df_apply_transformations
from loguru import logger

import re

def transform_match_keywords_df(match_keywords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a DataFrame containing 'Subtopic', 'Refined Keywords', and 'Negation' columns.
    This function splits out negation rows, processes them, and concatenates them back with
    the non-negation data to produce a standardized DataFrame with ['label', 'match', 'negate'] columns.
    
    The transformation involves:
        1. Validating the presence of required columns.
        2. Parsing 'Refined Keywords' and 'Negation' columns from string representations to Python objects.
        3. Separating negation entries and processing them.
        4. Exploding lists of matches into individual rows.
        5. Standardizing column names and concatenating negation and non-negation data.
    
    Args:
        match_keywords_df (pd.DataFrame): A Pandas DataFrame with the following columns:
            - 'Subtopic': The topic label.
            - 'Refined Keywords': A string representation of a list of keywords.
            - 'Negation': A string representation of a list of negated keywords (optional).
    
    Returns:
        pd.DataFrame: A transformed DataFrame with standardized columns ['label', 'match', 'negate'].
    
    Raises:
        ValueError: If any of the required columns are missing from the input DataFrame.
        Exception: If any error occurs during the transformation process.
    
    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'Subtopic': ['Positive', 'Positive', 'Negative'],
        ...     'Refined Keywords': ['["good", "excellent"]', '["great"]', '["bad", "poor"]'],
        ...     'Negation': ['["not good"]', None, '["not bad"]']
        ... }
        >>> df = pd.DataFrame(data)
        >>> transformed_df = transform_match_keywords_df(df)
        >>> print(transformed_df)
              label         match  negate
        0  POSITIVE           good   False
        1  POSITIVE      excellent   False
        2  POSITIVE          great   False
        3  POSITIVE      not good    True
        4  NEGATIVE            bad   False
        5  NEGATIVE           poor   False
        6  NEGATIVE       not bad    True
    """
    
    # 1. Validate required columns
    logger.info("Starting transformation of match_keywords_df.")
    required_columns = ["Subtopic", "Refined Keywords", "Negation"]
    for col in required_columns:
        if col not in match_keywords_df.columns:
            raise ValueError(f"Required column '{col}' is missing from input DataFrame.")

    # 2. Create a reference copy for negation processing
    match_keywords_df = df_apply_transformations(match_keywords_df, [('Refined Keywords', 'Refined Keywords', ast.literal_eval)])
    match_neg_df = match_keywords_df.copy()

    # 3. Extract rows containing negations
    df_negate = match_keywords_df[~match_keywords_df['Negation'].isna()][['Subtopic', 'Negation']]
    
    # 4. Apply transformations to Negation column (assuming df_apply_transformations is available)
    df_negate = df_apply_transformations(
        df_negate,
        transformations=[('Negation', 'Negation', ast.literal_eval)]
    )
    
    # 5. Explode negations into separate rows
    df_negate = df_negate.explode(column='Negation')
    df_negate['negate'] = True
    
    # 6. Rename columns for standardization
    df_negate = df_negate.rename(columns={'Subtopic': 'label', 'Negation': 'match'})
    
    # 7. Process main DataFrame portion (non-negation)

    match_neg_df = match_keywords_df[['Subtopic','Refined Keywords']].explode(column='Refined Keywords')

    match_neg_df['negate'] = False
    match_neg_df = match_neg_df.rename(columns={'Subtopic': 'label', 'Refined Keywords': 'match'})
    
    # 8. Concatenate negation rows with the main DataFrame
    df_output = pd.concat([match_neg_df, df_negate], ignore_index=True)
    
    return df_output

def transform_datm_keywords_df(datm_keywords_df: pd.DataFrame) -> pd.DataFrame:

    transformations2 =  [
        ("TOP60_KEYWORDS_COSINE", "TOP60_KEYWORDS_COSINE", ast.literal_eval),
        ("KEYWORDS", "KEYWORDS", ast.literal_eval),
        ("TOP60_KEYWORDS_COSINE", "TOP60_KEYWORDS_COSINE", lambda x: [(item['word'], item['score']) for item in x]),
        ("TOPIC_ID", "TOPIC_ID", int)
        ]

    datm_keywords_df = df_apply_transformations(datm_keywords_df, transformations2)

    datm_keywords_df = datm_keywords_df.sort_values(by=['TOPIC_ID'])
    datm_keywords_df.reset_index(drop=True, inplace=True)

    transformations3 = ["NeGATION", "NEGATION", lambda x: ast.literal_eval(x) if pd.notna(x) else x]

    labeled_m1_df = df_apply_transformations(labeled_m1_df, transformations3)

    datm_keywords_df = datm_keywords_df[datm_keywords_df['INCLUDED'] == 1]

    return datm_keywords_df


def create_match_patterns(matches: List[str], nlp: spacy.Language) -> Dict[str, Any]:
    """
    Creates a set of match patterns to handle variations in lemmatization and case.
    
    This function categorizes match phrases into unigrams, bigrams, and phrases.

    Args:
        matches (List[str]): List of match phrases.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'original': Original list of match phrases.
            - 'unigrams': Set of unigrams.
            - 'bigrams': Set of bigrams.
            - 'phrases': List of phrases longer than two words.

    Example:
        >>> from topic_modelling_package.processing import create_match_patterns
        >>> matches = ["Good", "Bad Service", "Average Experience"]
        >>> patterns = create_match_patterns(matches)
        >>> print(patterns)
        {
            'original': ["Good", "Bad Service", "Average Experience"],
            'unigrams': {"good"},
            'bigrams': {"bad_service", "bad_service", "average_experience"},
            'phrases': ["average experience"]
        }
    """
    bigrams = set(
        [
            word.lower() 
            for word in matches 
            if len(word.split('_')) == 2
        ] +
        [
            word.lower().replace(" ", '_') 
            for word in matches 
            if len(word.split(' ')) == 2
        ] +
        [
            '_'.join(tokenize_matched_words(word,nlp)) 
            for word in matches 
            if len(word.split(' ')) == 2
        ]
    )
    
    unigrams = set(
        [
            tokenize_matched_words(match, nlp)[0] 
            for match in matches 
            if ('_' not in match) and (len(match.split(' ')) == 1)
        ] +
        [
            match.lower() 
            for match in matches 
            if ('_' not in match) and (len(match.split(' ')) == 1)
        ]
    )

    phrases = [phrase.lower() for phrase in matches if len(phrase.split(" ")) > 2]

    logger.debug(f"Generated match patterns: { {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases} }")
    return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}


# match_count_lowStat
def count_matches_in_texts(
    texts: List[str],
    match_sets: Dict[str, Dict[str, Any]],
    nlp: spacy.Language,
    phrases: bool = True,
    suppress: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Counts occurrences of match patterns (unigrams, bigrams, phrases) within given texts.
    
    Args:
        texts (List[str]): List of sentences/documents.
        match_sets (Dict[str, Dict[str, Any]]): Dictionary containing match patterns categorized by labels.
        phrases (bool, optional): Whether to include phrase matching. Defaults to True.
        suppress (Optional[Dict[str, List[str]]], optional): Words to suppress from matching per label. 

            Defaults to None.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of counts for each match pattern categorized by labels.
    
    Example:
        >>> texts = ["I love good service.", "Bad service makes me unhappy."]
        >>> match_sets = {
        ...     'positive': {
        ...         'unigrams': {'good'},
        ...         'bigrams': {'good_service'},
        ...         'phrases': []
        ...     },
        ...     'negative': {
        ...         'unigrams': {'bad'},
        ...         'bigrams': {'bad_service'},
        ...         'phrases': []
        ...     }
        ... }
        >>> suppress = {
        ...     'positive': [],
        ...     'negative': []
        ... }
        >>> counts = count_matches_in_texts(texts, match_sets, phrases=True, suppress=suppress)
        >>> print(counts)
        {
            'positive': {
                'total': [1, 0],
                'stats': {'good': 1, 'good_service': 1}
            },
            'negative': {
                'total': [0, 1],
                'stats': {'bad': 1, 'bad_service': 1}
            }
        }
    """
    count_dict: Dict[str, Dict[str, int]] = {
        label: {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams'])} 
        for label, match_set in match_sets.items()
    }
    total_counts: Dict[str, List[int]] = {
        label: [] for label in match_sets.keys()
    }

    for text in texts:
        counted: Dict[str, int] = {label: 0 for label in match_sets.keys()}
        unigrams = tokenize_and_lemmatize_text(text, nlp)
        bigrams = ['_'.join(g) for g in generate_ngrams(unigrams, 2)]
        
        text_lower = text.lower()
        for label, match_set in match_sets.items(): 
            if suppress and label in suppress:
                if any(item in text_lower for item in suppress[label]):
                    logger.debug(f"Suppressing label '{label}' for text: {text}")
                    continue

            for word in unigrams:
                if word in match_set['unigrams']:
                    count_dict[label][word] += 1
                    counted[label] += 1

            for word in bigrams:
                if word in match_set['bigrams']:
                    count_dict[label][word] += 1
                    counted[label] += 1
            
            if phrases:
                for phrase in match_set.get('phrases', []):
                    if phrase in text_lower:
                        counted[label] += 1
                        logger.debug(f"Phrase '{phrase}' found in text: {text}")

        for label in match_sets.keys():
            total_counts[label].append(counted[label])

    result = {
        label: {
            'total': total_counts[label],
            'stats': count_dict[label]
        } 
        for label in match_sets.keys()
    }

    logger.info("Completed counting matches in texts.")
    return result

#  match_count_lowStat_singleSent
def count_matches_in_single_sentence(
    texts: str,
    match_sets: Dict[str, Dict[str, Any]],
    nlp: spacy.Language,
    phrases: bool = True,
    suppress: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Counts occurrences of match patterns within a single sentence.
    
    Args:
        text (str): Sentence to process.
        match_sets (Dict[str, Dict[str, Any]]): Dictionary containing match patterns categorized by labels.
        phrases (bool, optional): Whether to include phrase matching. Defaults to True.
        suppress (Optional[Dict[str, List[str]]], optional): Words to suppress from matching per label. 

            Defaults to None.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of counts for each match pattern categorized by labels.
    
    Example:
        >>> text = "I love good service and bad support."
        >>> match_sets = {
        ...     'positive': {
        ...         'unigrams': {'good'},
        ...         'bigrams': {'good_service'},
        ...         'phrases': []
        ...     },
        ...     'negative': {
        ...         'unigrams': {'bad'},
        ...         'bigrams': {'bad_support'},
        ...         'phrases': []
        ...     }
        ... }
        >>> suppress = {
        ...     'positive': [],
        ...     'negative': []
        ... }
        >>> counts = count_matches_in_single_sentence(text, match_sets, phrases=True, suppress=suppress)
        >>> print(counts)
        {
            'positive': {
                'total': 2,
                'stats': {'good': 1, 'good_service': 1}
            },
            'negative': {
                'total': 2,
                'stats': {'bad': 1, 'bad_support': 1}
            }
        }
    """
    count_dict = {label : {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}
    total_counts = {label: [] for label in match_sets.keys()}

    for text in texts:
        
        counted = {label: 0 for label in match_sets.keys()}
        unigrams = tokenize_and_lemmatize_text(text, nlp)
        bigrams = ['_'.join(g) for g in generate_ngrams(unigrams, 2)]
        
        text = text.lower()
        for label, match_set in match_sets.items(): 
        
            if any(item in text for item in suppress[label]):
                counted[label] += 0
                continue
                
            for word in unigrams:
                if word in match_set['unigrams']:
                    count_dict[label][word]+=1
                    counted[label] += 1

            for word in bigrams:
                if word in match_set['bigrams']:
                    count_dict[label][word]+=1
                    counted[label] += 1
            
            if phrases:
                if any(phrase in text for phrase in match_set['phrases']):
                    counted[label] += 1
                    continue

        for label in match_sets.keys():
        
            total_counts[label].append(counted[label])

        
    return {label : {'total': total_counts[label], 'stats' : count_dict[label]} for label in match_sets.keys()}

def calculate_binary_match_flags(
    texts: List[str],
    match_sets: Dict[str, Dict[str, Any]],
    suppress: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[int]]:
    """
    Calculates binary match counts for each label across multiple texts.
    
    For each text, this function tokenizes it into unigrams and bigrams, then checks for 
    the presence of any tokens or phrases from the provided match_sets. A match is recorded 
    as 1 if any unigram, bigram, or phrase is found *and* none of the suppression tokens 
    are present in the text; otherwise, a 0 is recorded.
    
    Args:
      texts (List[str]): List of texts to analyze.
      match_sets (Dict[str, Dict[str, Any]]): A dictionary where each key is a label and each 
          value is a dictionary with the following keys:
            - 'unigrams': A set or list of unigram tokens.
            - 'bigrams': A set or list of bigram tokens.
            - 'phrases': A list of phrase strings.
      suppress (Optional[Dict[str, List[str]]]): A dictionary where each key is a label and 
          each value is a list of tokens that, if present in the text, will suppress a match.
          If None, no suppression is applied.
          
    Returns:
      Dict[str, List[int]]: A dictionary mapping each label to a list of binary values (1 for 
          match, 0 otherwise) corresponding to each input text.
    
    Example:
      >>> texts = ["This is a sample text", "Another text with match"]
      >>> match_sets = {
      ...     "label1": {
      ...         "unigrams": {"sample", "match"}, 
      ...         "bigrams": {"this_is"}, 
      ...         "phrases": ["sample text"]
      ...     }
      ... }
      >>> suppress = {"label1": ["not"]}
      >>> match_count_lowStat_negation(texts, match_sets, suppress)
      {'label1': [1, 1]}
    """
    
    # If suppress is not provided, assign an empty list for each label.
    if suppress is None:
        suppress = {label: [] for label in match_sets.keys()}
    
    total_counts = {label: [] for label in match_sets.keys()}
    
    for text in texts:
        # Standardize text to lowercase for uniform matching.
        text_lower = text.lower()
        # Tokenize the text into unigrams.
        unigrams = tokenize_and_lemmatize_text(text_lower)
        # Generate bigrams by joining tokens with an underscore.
        bigrams = ['_'.join(gram) for gram in generate_ngrams(unigrams, 2)]
        
        for label, word_set in match_sets.items():
            # Ensure that tokens are in set form for faster lookups.
            unigrams_set = set(word_set.get('unigrams', []))
            bigrams_set = set(word_set.get('bigrams', []))
            phrases_list = word_set.get('phrases', [])
            suppress_list = suppress.get(label, [])
            
            # Check if any suppression token is present in the text.
            suppression_triggered = any(s_token in text_lower for s_token in suppress_list)
            
            # Check for matches in unigrams, bigrams, and phrases.
            match_unigrams = bool(set(unigrams) & unigrams_set)
            match_bigrams = bool(set(bigrams) & bigrams_set)
            match_phrases = any(phrase in text_lower for phrase in phrases_list)
            
            # If there's a match and no suppression, record a 1; otherwise, record a 0.
            if (match_unigrams or match_bigrams or match_phrases) and not suppression_triggered:
                total_counts[label].append(1)
            else:
                total_counts[label].append(0)
                
    return total_counts

def merge_count_dicts(count_list: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Merges a list of count dictionaries into a single count dictionary.
    
    Args:
        count_list (List[Dict[str, int]
        ]): List of dictionaries to merge.

    Returns:
        Dict[str, int]: Merged dictionary with aggregated counts.
    
    Example:
        >>> count_list = [{'good': 1, 'bad': 2}, {'good': 3, 'average': 1}]
        >>> merged = merge_count_dicts(count_list)
        >>> print(merged)
        {'good': 4, 'bad': 2, 'average': 1}
    """
    try:
        if not count_list:
            logger.warning("Empty count list provided. Returning empty dictionary.")
            return {}
        
        merged = Counter()
        for count_dict in count_list:
            merged.update(count_dict)
        
        if not merged:
            logger.warning("No matches found in count list. Returning default {'NO_MATCH': 1}.")
            return {'NO_MATCH': 1}
        
        logger.debug(f"Merged count dictionary: {dict(merged)}")
        return dict(merged)
    except Exception as e:
        logger.error(f"Failed to merge count dictionaries: {e}")
        return {'ERROR': 1}
