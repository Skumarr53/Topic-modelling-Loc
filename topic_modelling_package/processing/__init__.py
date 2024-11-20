# centralized_nlp_package/processing/__init__.py

from .match_operations import (
    create_match_patterns,
    count_matches_in_texts,
    count_matches_in_single_sentence,
    merge_count_dicts
)

__all__ = [
    'create_match_patterns',
    'count_matches_in_texts',
    'count_matches_in_single_sentence',
    'merge_count_dicts',
]
