# topic_modelling_package/utils/transformation.py
from typing import Callable, Dict, List, Tuple, Union
from centralized_nlp_package.text_processing import calculate_sentence_score, calculate_net_score



Transformation = Tuple[str, Union[str, List[str]], Callable]

def total_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation for the total count of a label within a topic.

    This function generates a transformation tuple that specifies how to extract the total count of a particular label within a given topic. The transformation includes the new column name, the source column from which to extract data, and a lambda function that defines the extraction logic.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'COUNT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_TOTAL_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that extracts the total count from the source data.

    Example:
        >>> transformation = total_transformation('consumer_strength', 'COUNT', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_TOTAL_COUNT', 'FILT_MD_COUNT', <function total_transformation.<locals>.<lambda> at 0x7f9c8c3e1d30>)
    """
    return (
        f"{topic}_TOTAL_{label}",
        f"{label_column}_{label}",
        lambda x: x[topic]["total"],
    )

def stats_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation for statistics of a label within a topic.

    This function generates a transformation tuple that specifies how to extract statistical data of a particular label within a given topic. The transformation includes the new column name, the source column from which to extract data, and a lambda function that defines the extraction logic.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'COUNT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_STATS_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that extracts the statistical data from the source data.

    Example:
        >>> transformation = stats_transformation('consumer_strength', 'STATS', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_STATS_STATS', 'FILT_MD_STATS', <function stats_transformation.<locals>.<lambda> at 0x7f9c8c3e1e50>)
    """
    return (
        f"{topic}_STATS_{label}",
        f"{label_column}_{label}",
        lambda x: x[topic]["stats"],
    )

def relevance_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation for the relevance score of a label within a topic.

    This function generates a transformation tuple that calculates the relevance score of a particular label within a given topic. The relevance score is determined by the proportion of positive occurrences relative to the total number of instances in the section.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'COUNT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_REL_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that calculates the relevance score.

    Example:
        >>> transformation = relevance_transformation('consumer_strength', 'REL', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_REL_REL', 'consumer_strength_TOTAL_REL', <function relevance_transformation.<locals>.<lambda> at 0x7f9c8c3e1f70>)
    """
    return (
        f"{topic}_REL_{label}",
        f"{topic}_TOTAL_{label}",
        lambda x: len([a for a in x if a > 0]) / len(x) if len(x) > 0 else None,
    )

def count_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation for counting positive occurrences of a label within a topic.

    This function generates a transformation tuple that counts the number of positive occurrences (values greater than zero) of a particular label within a given topic.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'COUNT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_COUNT_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that counts the number of positive occurrences.

    Example:
        >>> transformation = count_transformation('consumer_strength', 'COUNT', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_COUNT_COUNT', 'consumer_strength_TOTAL_COUNT', <function count_transformation.<locals>.<lambda> at 0x7f9c8c3e2040>)
    """
    return (
        f"{topic}_COUNT_{label}",
        f"{topic}_TOTAL_{label}",
        lambda x: len([a for a in x if a > 0]) if len(x) > 0 else None,
    )

def extract_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation to extract relevant matches based on a label within a topic.

    This function generates a transformation tuple that extracts relevant matched text segments based on the label within a given topic. It concatenates the matched labels where the corresponding total count exceeds zero.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'EXTRACT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_EXTRACT_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that extracts and concatenates relevant matches.

    Example:
        >>> transformation = extract_transformation('consumer_strength', 'EXTRACT', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_EXTRACT_EXTRACT', ['FILT_MD', 'consumer_strength_TOTAL_EXTRACT'], <function extract_transformation.<locals>.<lambda> at 0x7f9c8c3e21d0>)
    """
    return (
        f"{topic}_EXTRACT_{label}",
        [label, f"{topic}_TOTAL_{label}"],
        lambda x: " ".join([y for y, z in zip(x['label'], x[f"{topic}_TOTAL_{label}"]) if z > 0]),
    )

def sentiment_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation to calculate the sentiment score of a label within a topic.

    This function generates a transformation tuple that calculates the sentiment score for a particular label within a given topic. It utilizes external functions `calculate_sentence_score` to compute the score based on total counts and sentiment labels.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'SENT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_SENT_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that calculates the sentiment score.

    Example:
        >>> transformation = sentiment_transformation('consumer_strength', 'SENT', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_SENT_SENT', ['consumer_strength_TOTAL_SENT', 'SENT_LABELS_SENT'], <function sentiment_transformation.<locals>.<lambda> at 0x7f9c8c3e2370>)
    """
    return (
        f"{topic}_SENT_{label}",
        [f"{topic}_TOTAL_{label}", f"SENT_LABELS_{label}"],
        lambda x: calculate_sentence_score(x[f"{topic}_TOTAL_{label}"], x[f"SENT_LABELS_{label}"], apply_weight=False),
    )

def net_sent_transformation(topic: str, label: str, label_column: str) -> Transformation:
    """
    Creates a transformation to calculate the net sentiment score of a label within a topic.

    This function generates a transformation tuple that calculates the net sentiment score for a particular label within a given topic. It utilizes external functions `calculate_net_score` to compute the score based on total counts and sentiment labels.

    Args:
        topic (str):
            The topic name (e.g., 'consumer_strength', 'consumer_weakness').
        label (str):
            The label name associated with the topic (e.g., 'NET_SENT', 'REL').
        label_column (str):
            The column name in the DataFrame that contains the label data (e.g., 'FILT_MD', 'FILT_QA').

    Returns:
        Transformation:
            A tuple containing:
                - str: The new column name in the format "{topic}_NET_SENT_{label}".
                - Union[str, List[str]]: The source column(s) from which to extract data.
                - Callable: A lambda function that calculates the net sentiment score.

    Example:
        >>> transformation = net_sent_transformation('consumer_strength', 'NET_SENT', 'FILT_MD')
        >>> print(transformation)
        ('consumer_strength_NET_SENT_NET_SENT', ['consumer_strength_TOTAL_NET_SENT', 'SENT_LABELS_NET_SENT'], <function net_sent_transformation.<locals>.<lambda> at 0x7f9c8c3e24f0>)
    """
    return (
        f"{topic}_NET_SENT_{label}",
        [f"{topic}_TOTAL_{label}", f"SENT_LABELS_{label}"],
        lambda x: calculate_net_score(x[f"{topic}_TOTAL_{label}"], x[f"SENT_LABELS_{label}"]),
    )

# Mapping of transformation names to their corresponding functions
STATISTICS_MAP: Dict[str, Callable[[str, str, str], Transformation]] = {
    "total": total_transformation,
    "stats": stats_transformation,
    "relevance": relevance_transformation,
    "count": count_transformation,
    "extract": extract_transformation,
    "sentiment": sentiment_transformation,
    "net_sentiment": net_sent_transformation,  # Added new transformation
}