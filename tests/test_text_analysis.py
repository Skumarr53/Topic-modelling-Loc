# tests/text_processing/test_text_analysis.py

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Import the functions to be tested
from centralized_nlp_package.text_processing.text_analysis import (
    get_blob_storage_path,
    load_word_set,
    load_syllable_counts,
    check_negation,
    calculate_polarity_score,
    polarity_score_per_section,
    polarity_score_per_sentence,
    is_complex,
    fog_analysis_per_section,
    fog_analysis_per_sentence,
    tone_count_with_negation_check,
    tone_count_with_negation_check_per_sentence
)

# Import custom exceptions
from centralized_nlp_package.utils.exception import FilesNotLoadedException

# Import Config class
from centralized_nlp_package.utils.config import Config


@pytest.fixture
def mock_config():
    """
    Fixture to mock the Config object.
    """
    with patch('centralized_nlp_package.text_processing.text_analysis.Config') as MockConfig:
        instance = MockConfig()
        # Mock the attributes used in the functions
        instance.psycholinguistics.model_artifacts_path = "path/to/model_artifacts/"
        instance.psycholinguistics.filecfg.litigious_flnm = "litigious_words.txt"
        instance.psycholinguistics.filecfg.complex_flnm = "complex_words.txt"
        instance.psycholinguistics.filecfg.uncertianity_flnm = "uncertainty_words.txt"
        instance.psycholinguistics.filecfg.syllable_flnm = "syllable_count.txt"
        instance.psycholinguistics.filecfg.vocab_pos_flnm = "vocab_pos.txt"
        instance.psycholinguistics.filecfg.vocab_neg_flnm = "vocab_neg.txt"
        yield instance


@pytest.fixture
def mock_blob_util():
    """
    Fixture to mock the BlobStorageUtility.
    """
    with patch('centralized_nlp_package.text_processing.text_analysis.BlobStorageUtility') as MockBlob:
        blob_instance = MockBlob.return_value
        yield blob_instance


@pytest.fixture
def mock_preprocess_obj():
    """
    Fixture to mock the DictionaryModelPreprocessor.
    """
    with patch('centralized_nlp_package.text_processing.text_analysis.DictionaryModelPreprocessor') as MockPreproc:
        preprocess_instance = MockPreproc.return_value
        # Mock the word_negated method
        preprocess_instance.word_negated = MagicMock(return_value=False)
        yield preprocess_instance


@pytest.fixture
def mock_statistics_obj():
    """
    Fixture to mock the Statistics object.
    """
    with patch('centralized_nlp_package.text_processing.text_analysis.Statistics') as MockStats:
        stats_instance = MockStats.return_value
        # Mock the combine_sent method
        stats_instance.combine_sent = MagicMock(return_value=0.0)
        yield stats_instance


def test_get_blob_storage_path(mock_config):
    """
    Test that get_blob_storage_path constructs the correct path.
    """
    filename = "test_file.txt"
    expected_path = "path/to/model_artifacts/test_file.txt"
    path = get_blob_storage_path(mock_config, filename)
    assert path == expected_path


def test_load_word_set_success(mock_blob_util, mock_config):
    """
    Test that load_word_set successfully loads a word set.
    """
    filename = "test_words.txt"
    expected_words = {"word1", "word2", "word3"}
    mock_blob_util.load_list_from_txt.return_value = expected_words

    words = load_word_set(mock_blob_util, mock_config, filename)
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/test_words.txt")
    assert words == expected_words


def test_load_word_set_file_not_found(mock_blob_util, mock_config):
    """
    Test that load_word_set raises FilesNotLoadedException when file not found.
    """
    filename = "missing_words.txt"
    mock_blob_util.load_list_from_txt.side_effect = FileNotFoundError

    with pytest.raises(FilesNotLoadedException) as exc_info:
        load_word_set(mock_blob_util, mock_config, filename)
    
    assert exc_info.value.filename == filename


def test_load_syllable_counts_success(mock_blob_util, mock_config):
    """
    Test that load_syllable_counts successfully loads syllable counts.
    """
    filename = "syllables.txt"
    expected_syllables = {"word1": 2, "word2": 3}
    mock_blob_util.read_syllable_count.return_value = expected_syllables

    syllables = load_syllable_counts(mock_blob_util, mock_config, filename)
    mock_blob_util.read_syllable_count.assert_called_with("path/to/model_artifacts/syllables.txt")
    assert syllables == expected_syllables


def test_load_syllable_counts_file_not_found(mock_blob_util, mock_config):
    """
    Test that load_syllable_counts raises FilesNotLoadedException when file not found.
    """
    filename = "missing_syllables.txt"
    mock_blob_util.read_syllable_count.side_effect = FileNotFoundError

    with pytest.raises(FilesNotLoadedException) as exc_info:
        load_syllable_counts(mock_blob_util, mock_config, filename)
    
    assert exc_info.value.filename == filename


def test_check_negation_with_negation_present(mock_preprocess_obj):
    """
    Test that check_negation returns True when a negation is present within the window.
    """
    input_words = ["this", "is", "not", "good"]
    index = 3
    mock_preprocess_obj.word_negated.side_effect = [False, False, True]

    result = check_negation(input_words, index, mock_preprocess_obj)
    assert result is True
    mock_preprocess_obj.word_negated.assert_called_with("is")


def test_check_negation_no_negation(mock_preprocess_obj):
    """
    Test that check_negation returns False when no negation is present within the window.
    """
    input_words = ["this", "is", "a", "good"]
    index = 3
    mock_preprocess_obj.word_negated.side_effect = [False, False, False]

    result = check_negation(input_words, index, mock_preprocess_obj)
    assert result is False
    mock_preprocess_obj.word_negated.assert_called_with("a")


def test_calculate_polarity_score_normal(mock_preprocess_obj, mock_statistics_obj):
    """
    Test calculate_polarity_score with normal input.
    """
    input_words = ["happy", "sad", "joyful", "terrible"]
    positive_words = {"happy", "joyful"}
    negative_words = {"sad", "terrible"}

    polarity, wc, sum_neg, pos_count, legacy = calculate_polarity_score(
        input_words,
        positive_words,
        negative_words,
        mock_preprocess_obj,
        mock_statistics_obj
    )

    assert polarity == (2 - 2) / 4  # (2 - 2)/4 = 0.0
    assert wc == 4
    assert sum_neg == 2
    assert pos_count == 2
    mock_statistics_obj.combine_sent.assert_called_with(2, 2)
    assert legacy == 0.0


def test_calculate_polarity_score_with_negation(mock_preprocess_obj, mock_statistics_obj):
    """
    Test calculate_polarity_score when a positive word is negated.
    """
    input_words = ["not", "happy", "sad", "joyful"]
    positive_words = {"happy", "joyful"}
    negative_words = {"sad", "terrible"}

    # Simulate negation for "happy"
    mock_preprocess_obj.word_negated.side_effect = [False, True, False, False]

    polarity, wc, sum_neg, pos_count, legacy = calculate_polarity_score(
        input_words,
        positive_words,
        negative_words,
        mock_preprocess_obj,
        mock_statistics_obj
    )

    # "happy" is negated, so it's treated as negative
    # positive_count = 1 ("joyful")
    # sum_neg = 1 ("happy" negated) + 1 ("sad") = 2
    assert polarity == (1 - 2) / 4  # -0.25
    assert wc == 4
    assert sum_neg == 2
    assert pos_count == 1
    mock_statistics_obj.combine_sent.assert_called_with(1, 2)
    assert legacy == 0.0


def test_calculate_polarity_score_empty_input(mock_preprocess_obj, mock_statistics_obj):
    """
    Test calculate_polarity_score with empty input.
    """
    input_words = []
    positive_words = {"happy", "joyful"}
    negative_words = {"sad", "terrible"}

    polarity, wc, sum_neg, pos_count, legacy = calculate_polarity_score(
        input_words,
        positive_words,
        negative_words,
        mock_preprocess_obj,
        mock_statistics_obj
    )

    assert polarity is np.nan
    assert wc == 0
    assert sum_neg == 0
    assert pos_count == 0
    mock_statistics_obj.combine_sent.assert_called_with(0, 0)
    assert legacy == 0.0


def test_polarity_score_per_section_normal(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test polarity_score_per_section with normal input.
    """
    text_list = ["I am happy with the results.", "This is a sad day."]
    litigious_words = {"results", "day"}
    complex_words = {"happy", "sad"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, complex_words, uncertain_words]
    mock_preprocess_obj.preprocess_text.return_value = ("cleaned text", ["i", "am", "happy", "with", "the", "results", "this", "is", "a", "sad", "day"], 11)

    expected_output = (
        len(litigious_words & set(["i", "am", "happy", "with", "the", "results", "this", "is", "a", "sad", "day"])) / 11,
        len(complex_words & set(["i", "am", "happy", "with", "the", "results", "this", "is", "a", "sad", "day"])) / 11,
        len(uncertain_words & set(["i", "am", "happy", "with", "the", "results", "this", "is", "a", "sad", "day"])) / 11,
        11,
        2,  # "results", "day"
        2,  # "happy", "sad"
        0   # No uncertain words
    )

    output = polarity_score_per_section(mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj)
    assert output == expected_output
    assert mock_blob_util.load_list_from_txt.call_count == 3
    mock_preprocess_obj.preprocess_text.assert_called_once_with(text_list, mock_preprocess_obj)


def test_polarity_score_per_section_insufficient_data(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test polarity_score_per_section with insufficient data.
    """
    text_list = ["I am happy."]
    litigious_words = {"results", "day"}
    complex_words = {"happy"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, complex_words, uncertain_words]
    mock_preprocess_obj.preprocess_text.return_value = ("cleaned text", ["i", "am", "happy"], 3)

    # Since word_count <=1 is not the case, but uncertain words count is 0
    expected_output = (
        len(litigious_words & set(["i", "am", "happy"])) / 3,
        len(complex_words & set(["i", "am", "happy"])) / 3,
        len(uncertain_words & set(["i", "am", "happy"])) / 3,
        3,
        0,  # No litigious words
        1,  # "happy"
        0   # No uncertain words
    )

    output = polarity_score_per_section(mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj)
    assert output == expected_output
    assert mock_blob_util.load_list_from_txt.call_count == 3
    mock_preprocess_obj.preprocess_text.assert_called_once_with(text_list, mock_preprocess_obj)


def test_polarity_score_per_sentence_normal(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test polarity_score_per_sentence with normal input.
    """
    text_list = ["I am happy.", "This is a sad day."]
    litigious_words = {"day"}
    complex_words = {"happy", "sad"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, complex_words, uncertain_words]
    mock_preprocess_obj.preprocess_text_list.return_value = (
        "cleaned text",
        [["i", "am", "happy"], ["this", "is", "a", "sad", "day"]],
        [3, 5]
    )

    # Setup calculate_polarity_score mock
    def side_effect_calculate_polarity_score(input_words, pos_words, neg_words, preproc, stats):
        if input_words == ["i", "am", "happy"]:
            return (1.0, 3, 0, 1, 0.0)
        elif input_words == ["this", "is", "a", "sad", "day"]:
            return (-0.4, 5, 2, 0, 0.0)
    
    with patch('centralized_nlp_package.text_processing.text_analysis.calculate_polarity_score', side_effect=side_effect_calculate_polarity_score):
        word_counts, positive_counts, negative_counts = polarity_score_per_sentence(
            mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj
        )
    
    assert word_counts == [3, 5]
    assert positive_counts == [1, 0]
    assert negative_counts == [0, 2]


def test_polarity_score_per_sentence_insufficient_data(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test polarity_score_per_sentence with insufficient data.
    """
    text_list = []
    litigious_words = {"day"}
    complex_words = {"happy", "sad"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, complex_words, uncertain_words]
    mock_preprocess_obj.preprocess_text_list.return_value = ("", [], 0)

    word_counts, positive_counts, negative_counts = polarity_score_per_sentence(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj
    )

    assert word_counts is None
    assert positive_counts is None
    assert negative_counts is None
    assert mock_blob_util.load_list_from_txt.call_count == 3
    mock_preprocess_obj.preprocess_text_list.assert_called_once_with(text_list, mock_preprocess_obj)


def test_is_complex_true(syllables):
    """
    Test is_complex returns True for complex words.
    """
    syllables = {"unbelievable": 5, "running": 2, "tested": 2}
    assert is_complex("unbelievable", syllables) is True
    assert is_complex("running", syllables) is False
    assert is_complex("tested", syllables) is False


def test_is_complex_false(syllables):
    """
    Test is_complex returns False for non-complex words.
    """
    syllables = {"happy": 2, "sad": 1}
    assert is_complex("happy", syllables) is False
    assert is_complex("sad", syllables) is False


def test_is_complex_suffix_rule(syllables):
    """
    Test is_complex with suffix rules applied.
    """
    syllables = {"fix": 1, "fixes": 1, "running": 2, "tested": 2, "happiness": 3}
    assert is_complex("fixes", syllables) is False  # "fix" syllables <=2
    assert is_complex("happiness", syllables) is True  # syllables >2
    assert is_complex("running", syllables) is False
    assert is_complex("tested", syllables) is False


def test_fog_analysis_per_section_normal(mock_blob_util, mock_config, mock_preprocess_obj):
    """
    Test fog_analysis_per_section with normal input.
    """
    text_list = ["This is a simple sentence.", "Another complex sentence with many syllables."]
    syllable_counts = {"This": 1, "is": 1, "a": 1, "simple": 2, "sentence": 2, "Another": 3, "complex": 2, "with": 1, "many": 2, "syllables": 3}

    # Setup mocks
    mock_blob_util.read_syllable_count.return_value = syllable_counts
    mock_preprocess_obj.preprocess_text.return_value = ("cleaned text", ["this", "is", "a", "simple", "sentence", "another", "complex", "sentence", "with", "many", "syllables"], 11)

    expected_fog_index = 0.4 * (2.5 + 100 * (3 / 11))  # average_words_per_sentence = 2.5, complex_word_count =3, total_word_count=11

    fog_index, complex_word_count, avg_words, total_wc = fog_analysis_per_section(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj
    )

    assert np.isclose(fog_index, 0.4 * (2.5 + 100 * (3 / 11)))
    assert complex_word_count == 3
    assert avg_words == 2.5
    assert total_wc == 11
    mock_blob_util.read_syllable_count.assert_called_with("path/to/model_artifacts/syllable_count.txt")
    mock_preprocess_obj.preprocess_text.assert_called_once_with(text_list, mock_preprocess_obj)


def test_fog_analysis_per_section_insufficient_data(mock_blob_util, mock_config, mock_preprocess_obj):
    """
    Test fog_analysis_per_section with insufficient data.
    """
    text_list = ["A short."]
    syllable_counts = {"A": 1, "short": 1}

    # Setup mocks
    mock_blob_util.read_syllable_count.return_value = syllable_counts
    mock_preprocess_obj.preprocess_text.return_value = ("", [], 0)

    fog_index, complex_word_count, avg_words, total_wc = fog_analysis_per_section(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj
    )

    assert np.isnan(fog_index)
    assert np.isnan(complex_word_count)
    assert np.isnan(avg_words)
    assert np.isnan(total_wc)
    mock_blob_util.read_syllable_count.assert_called_with("path/to/model_artifacts/syllable_count.txt")
    mock_preprocess_obj.preprocess_text.assert_called_once_with(text_list, mock_preprocess_obj)


def test_fog_analysis_per_sentence_normal(mock_blob_util, mock_config, mock_preprocess_obj):
    """
    Test fog_analysis_per_sentence with normal input.
    """
    text_list = ["This is a simple sentence.", "Another complex sentence with many syllables."]
    syllable_counts = {"this": 1, "is": 1, "a": 1, "simple": 2, "sentence": 2, "another": 3, "complex": 2, "with": 1, "many": 2, "syllables": 3}

    # Setup mocks
    mock_blob_util.read_syllable_count.return_value = syllable_counts
    mock_preprocess_obj.preprocess_text_list.return_value = (
        "cleaned text",
        [["this", "is", "a", "simple", "sentence"], ["another", "complex", "sentence", "with", "many", "syllables"]],
        [5, 6]
    )

    # Expected fog indices
    # For first sentence:
    # average_words_per_sentence = 5.5
    # complex_word_count = 2 (simple, sentence)
    # total_word_count = 11
    # fog_index = 0.4 * (5.5 + 100 * (2 / 11)) ≈ 0.4 * (5.5 + 18.18) ≈ 0.4 * 23.68 ≈ 9.47

    # For second sentence:
    # complex_word_count = 3 (complex, syllables, another)
    # fog_index = 0.4 * (5.5 + 100 * (3 / 11)) ≈ 0.4 * (5.5 + 27.27) ≈ 0.4 * 32.77 ≈ 13.11

    expected_fog_indices = [
        0.4 * (5.5 + 100 * (2 / 11)),
        0.4 * (5.5 + 100 * (3 / 11))
    ]
    expected_complex_counts = [2, 3]
    expected_total_wc = [5, 6]

    fog_index_list, complex_word_count_list, total_word_count_list = fog_analysis_per_sentence(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj
    )

    assert len(fog_index_list) == 2
    assert len(complex_word_count_list) == 2
    assert len(total_word_count_list) == 2
    assert np.isclose(fog_index_list[0], expected_fog_indices[0], atol=1e-2)
    assert np.isclose(fog_index_list[1], expected_fog_indices[1], atol=1e-2)
    assert complex_word_count_list == expected_complex_counts
    assert total_word_count_list == expected_total_wc
    mock_blob_util.read_syllable_count.assert_called_with("path/to/model_artifacts/syllable_count.txt")
    mock_preprocess_obj.preprocess_text_list.assert_called_once_with(text_list, mock_preprocess_obj)


def test_fog_analysis_per_sentence_insufficient_data(mock_blob_util, mock_config, mock_preprocess_obj):
    """
    Test fog_analysis_per_sentence with insufficient data.
    """
    text_list = []
    syllable_counts = {}

    # Setup mocks
    mock_blob_util.read_syllable_count.return_value = syllable_counts
    mock_preprocess_obj.preprocess_text_list.return_value = ("", [], 0)

    fog_index_list, complex_word_count_list, total_word_count_list = fog_analysis_per_sentence(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj
    )

    assert fog_index_list is None
    assert complex_word_count_list is None
    assert total_word_count_list is None
    mock_blob_util.read_syllable_count.assert_called_with("path/to/model_artifacts/syllable_count.txt")
    mock_preprocess_obj.preprocess_text_list.assert_called_once_with(text_list, mock_preprocess_obj)


def test_tone_count_with_negation_check_normal(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test tone_count_with_negation_check with normal input.
    """
    text_list = ["I am happy.", "This is a sad day."]
    litigious_words = {"day"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, set(), uncertain_words]
    mock_preprocess_obj.preprocess_text.return_value = ("cleaned text", ["i", "am", "happy", "this", "is", "a", "sad", "day"], 8)
    mock_statistics_obj.combine_sent.return_value = 0.0

    expected_output = (
        [0.0],  # polarity_scores
        [8],     # word_counts
        [0],     # negative_word_counts
        [1],     # positive_word_counts
        [0.0]    # legacy_scores
    )

    output = tone_count_with_negation_check(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj
    )

    assert output == expected_output
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/litigious_words.txt")
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/complex_words.txt")  # Assuming empty set
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/uncertainty_words.txt")
    mock_preprocess_obj.preprocess_text.assert_called_once_with(text_list, mock_preprocess_obj)
    mock_statistics_obj.combine_sent.assert_called_once_with(1, 0)


def test_tone_count_with_negation_check_insufficient_data(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test tone_count_with_negation_check with insufficient data.
    """
    text_list = []
    litigious_words = {"day"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, set(), uncertain_words]
    mock_preprocess_obj.preprocess_text.return_value = ("", [], 0)

    expected_output = (
        [np.nan],  # polarity_scores
        [np.nan],  # word_counts
        [np.nan],  # negative_word_counts
        [np.nan],  # positive_word_counts
        [np.nan]   # legacy_scores
    )

    output = tone_count_with_negation_check(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj
    )

    assert output == expected_output
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/litigious_words.txt")
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/complex_words.txt")  # Assuming empty set
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/uncertainty_words.txt")
    mock_preprocess_obj.preprocess_text.assert_called_once_with(text_list, mock_preprocess_obj)
    mock_statistics_obj.combine_sent.assert_not_called()


def test_tone_count_with_negation_check_per_sentence_normal(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test tone_count_with_negation_check_per_sentence with normal input.
    """
    text_list = ["I am happy.", "This is a sad day."]
    litigious_words = {"day"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, set(), uncertain_words]
    mock_preprocess_obj.preprocess_text_list.return_value = (
        "cleaned text",
        [["i", "am", "happy"], ["this", "is", "a", "sad", "day"]],
        [3, 5]
    )
    mock_statistics_obj.combine_sent.side_effect = [0.0, 0.0]

    expected_output = (
        [3, 5],  # word_counts
        [1, 0],  # positive_counts
        [0, 2]   # negative_counts
    )

    output = tone_count_with_negation_check_per_sentence(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj
    )

    assert output == expected_output
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/litigious_words.txt")
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/complex_words.txt")  # Assuming empty set
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/uncertainty_words.txt")
    mock_preprocess_obj.preprocess_text_list.assert_called_once_with(text_list, mock_preprocess_obj)
    assert mock_statistics_obj.combine_sent.call_count == 2
    mock_statistics_obj.combine_sent.assert_any_call(1, 0)
    mock_statistics_obj.combine_sent.assert_any_call(0, 2)


def test_tone_count_with_negation_check_per_sentence_insufficient_data(mock_blob_util, mock_config, mock_preprocess_obj, mock_statistics_obj):
    """
    Test tone_count_with_negation_check_per_sentence with insufficient data.
    """
    text_list = []
    litigious_words = {"day"}
    uncertain_words = {"maybe", "perhaps"}

    # Setup mocks
    mock_blob_util.load_list_from_txt.side_effect = [litigious_words, set(), uncertain_words]
    mock_preprocess_obj.preprocess_text_list.return_value = ("", [], 0)

    expected_output = (
        None,
        None,
        None
    )

    output = tone_count_with_negation_check_per_sentence(
        mock_blob_util, mock_config, text_list, mock_preprocess_obj, mock_statistics_obj
    )

    assert output == expected_output
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/litigious_words.txt")
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/complex_words.txt")  # Assuming empty set
    mock_blob_util.load_list_from_txt.assert_called_with("path/to/model_artifacts/uncertainty_words.txt")
    mock_preprocess_obj.preprocess_text_list.assert_called_once_with(text_list, mock_preprocess_obj)
    mock_statistics_obj.combine_sent.assert_not_called()
