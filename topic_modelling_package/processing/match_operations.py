from collections import Counter
from centralized_nlp_package.text_processing.text_utils import find_ngrams
from centralized_nlp_package.preprocessing.text_preprocessing import (tokenize_matched_words, 
                                                                      tokenize_and_lemmatize_text)
def get_match_set(matches: list) -> dict:
    """
    Create a set of match patterns to handle variations in lemmatization and case.

    Args:
        matches (list): List of match phrases.

    Returns:
        dict: Dictionary containing unigrams, bigrams, and phrases.
    """
    bigrams = set(
        [word.lower() for word in matches if len(word.split('_')) == 2] +
        [word.lower().replace(" ", '_') for word in matches if len(word.split(' ')) == 2] +
        ['_'.join(tokenize_matched_words(word)) for word in matches if len(word.split(' ')) == 2]
    )
    
    unigrams = set(
        [tokenize_matched_words(match)[0] for match in matches if ('_' not in match) and (len(match.split(' ')) == 1)] +
        [match.lower() for match in matches if ('_' not in match) and (len(match.split(' ')) == 1)]
    )

    phrases = [phrase.lower() for phrase in matches if len(phrase.split(" ")) > 2]

    return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}

def match_count_lowStat(texts: list, match_sets: dict, phrases: bool = True, suppress: dict = None) -> dict:
    """
    Count occurrences of match patterns (unigrams, bigrams, phrases) within given texts.

    Args:
        texts (list): List of sentences/documents.
        match_sets (dict): Dictionary containing match patterns.
        phrases (bool): Whether to include phrase matching.
        suppress (dict): Words to suppress from matching.

    Returns:
        dict: Dictionary of counts for each match pattern.
    """
    count_dict = {label: {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams'])} for label, match_set in match_sets.items()}
    total_counts = {label: [] for label in match_sets.keys()}

    for text in texts:
        counted = {label: 0 for label in match_sets.keys()}
        unigrams = tokenize_and_lemmatize_text(text)
        bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
        
        text = text.lower()
        for label, match_set in match_sets.items(): 
            if suppress and any(item in text for item in suppress[label]):
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
                if any(phrase in text for phrase in match_set['phrases']):
                    counted[label] += 1

        for label in match_sets.keys():
            total_counts[label].append(counted[label])

    return {label: {'total': total_counts[label], 'stats': count_dict[label]} for label in match_sets.keys()}

def match_count_lowStat_singleSent(text: str, match_sets: dict, phrases: bool = True, suppress: dict = None) -> dict:
    """
    Count occurrences of match patterns within a single sentence.

    Args:
        text (str): Sentence to process.
        match_sets (dict): Dictionary containing match patterns.
        phrases (bool): Whether to include phrase matching.
        suppress (dict): Words to suppress from matching.

    Returns:
        dict: Dictionary of counts for each match pattern.
    """
    count_dict = {label: {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams'])} for label, match_set in match_sets.items()}
    counted = {label: 0 for label in match_sets.keys()}
    
    unigrams = word_tokenize(text)
    bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
    text = text.lower()

    for label, match_set in match_sets.items():
        if suppress and any(item in text for item in suppress[label]):
            continue

        for word in unigrams:
            if word in match_set['unigrams']:
                count_dict[label][word] += 1
                counted[label] += 1

        for word in bigrams:
            if word in match_set['bigrams']:
                count_dict[label][word] += 1
                counted[label] += 1

        if phrases and any(phrase in text for phrase in match_set['phrases']):
            counted[label] += 1

    return {label: {'total': counted[label], 'stats': count_dict[label]} for label in match_sets.keys()}

def merge_count(count_list: list) -> dict:
    """
    Merges a list of Counter dictionaries.

    Args:
        count_list (list): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary.
    """
    try:
        merge = Counter(count_list[0])
        for calc in count_list[1:]:
            merge += Counter(calc)
        return merge if len(merge.keys()) > 0 else {'NO_MATCH': 1}
    except:
        return {'ERROR': 1}
