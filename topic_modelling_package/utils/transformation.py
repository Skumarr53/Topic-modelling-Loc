
from typing import Callable, Dict, List, Tuple, Union
from centralized_nlp_package.text_processing.text_analysis import calculate_sentence_score



Transformation = Tuple[str, Union[str, List[str]], Callable]

def total_transformation(topic: str, label: str, label_column: str) -> Transformation:
    return (
        f"{topic}_TOTAL_{label}",
        f"{label_column}_{label}",
        lambda x: x[topic]["total"],
    )

def stats_transformation(topic: str, label: str, label_column: str) -> Transformation:
    return (
        f"{topic}_STATS_{label}",
        f"{label_column}_{label}",
        lambda x: x[topic]["stats"],
    )

def relevance_transformation(topic: str, label: str, label_column: str) -> Transformation:
    return (
        f"{topic}_REL_{label}",
        f"{topic}_TOTAL_{label}",
        lambda x: len([a for a in x if a > 0]) / len(x) if len(x) > 0 else None,
    )

def count_transformation(topic: str, label: str, label_column: str) -> Transformation:
    return (
        f"{topic}_COUNT_{label}",
        f"{label_column}_{label}",
        lambda x: len([a for a in x if a > 0]) if len(x) > 0 else None,
    )

def extract_transformation(topic: str, label: str, label_column: str) -> Transformation:
    return (
        f"{topic}_EXTRACT_{label}",
        [label, f"{topic}_TOTAL_{label}"],
        lambda x: " ".join([y for y, z in zip(x[0], x[1]) if z > 0]),
    )

def sentiment_transformation(topic: str, label: str, label_column: str) -> Transformation:
    return (
        f"{topic}_SENT_{label}",
        [f"{topic}_TOTAL_{label}", f"SENT_LABELS_{label}"],
        lambda x: calculate_sentence_score(x[0], x[1], weight=False),
    )


STATISTICS_MAP: Dict[str, Callable[[str, str], Transformation]] = {
    "total": total_transformation,
    "stats": stats_transformation,
    "relevance": relevance_transformation,
    "count": count_transformation,
    "extract": extract_transformation,
    "sentiment": sentiment_transformation,
}