import pytest
import pandas as pd
from reports.report_generation import generate_topic_report, generate_top_matches_report

@pytest.fixture
def sample_dataframe():
    data = {
        'ENTITY_ID': ['E1', 'E2'],
        'CALL_NAME': ['Call 1', 'Call 2'],
        'FILT_MD': ['Data 1', 'Data 2'],
        'matches_FILT_MD': [{'RECOVERY': {'total': [1, 0], 'stats': {'key': 1}}}, {'RECOVERY': {'total': [2, 1], 'stats': {'key': 2}}}]
    }
    return pd.DataFrame(data)

def test_generate_topic_report(sample_dataframe):
    word_set_dict = {
        "RECOVERY": {'original': [], 'unigrams': [], 'bigrams': [], 'phrases': []}
    }
    updated_df = generate_topic_report(sample_dataframe, word_set_dict)
    assert 'RECOVERY_TOTAL_FILT_MD' in updated_df.columns, "The column for total counts should be added."
    assert 'RECOVERY_STATS_FILT_MD' in updated_df.columns, "The column for stats should be added."

def test_generate_top_matches_report(sample_dataframe):
    report_df = generate_top_matches_report(sample_dataframe, topic="RECOVERY", label="FILT_MD", sort_by="RECOVERY_TOTAL_FILT_MD", top_n=1)
    assert len(report_df) == 1, "The report should only contain the top 1 row."
