import ast
import pandas as pd
from topic_modelling_package import config
from data.data_extraction import DataExtraction
from processing.text_processing import word_tokenize
from processing.match_operations import get_match_set, match_count_lowStat
from reports.report_generation import generate_topic_report, generate_top_matches_report, save_report_to_csv
from utils.dask_utils import initialize_dask_client, dask_compute_with_progress
from utils.db_access import DBFShelper
import hydra
from utils.logging import setup_logging
from centralized_nlp_package.preprocessing.text_preprocessing import (initialize_spacy, 
                                                                      tokenize_and_lemmatize_text, 
                                                                      tokenize_matched_words)
from centralized_nlp_package.text_processing.text_utils import find_ngrams
from centralized_nlp_package.data_access.dask_utils import initialize_dask_client
from centralized_nlp_package.data_access.snowflake_utils import read
from centralized_nlp_package.utils.helpers import (df_convert_str2py_objects, 
                                                   query_constructor, 
                                                   get_date_range, 
                                                   df_apply_transformations)
from topic_modelling_pakage import create_topic_dict


dask_client = initialize_dask_client()

min_dt, max_dt = get_date_range(months_back=1)

## base query: "select * from table where date between {min_date} and {max_date}"
## query_path: Load  base_query from a file | ex: config/inp_query  
query = query_constructor(query_path, min_date = min_dt, max_date = max_dt) 


curr_df = read_from_snowflake(query)


# row transformation 
## transformed column name as key if column does not exist a new column will be created and function as values
row_transformations1 = [
                            ('CALL_ID', str, False),
                            ('FILT_MD', ast.literal_eval, False),
                            ('FILT_QA', ast.literal_eval, False),
                            ('LEN_MD', lambda row: len(row['FILT_MD']), True),
                            ('LEN_QA', lambda row: len(row['FILT_QA']), True),
                            ('FILT_MD', lambda row: [y for y, z in zip(row['FILT_MD'], row['SENT_LABELS_FILT_MD']) if (z == 1 and not y.endswith('?'))], True),
                            ('FILT_QA', lambda row: [y for y, z in zip(row['FILT_QA'], row['SENT_LABELS_FILT_QA']) if (z == 1 and not y.endswith('?'))], True)
                        ]

curr_df = df_apply_transformations(curr_df, row_transformations1)

nlp = initialize_spacy()

match_df = pd.read_csv(dbutils.widgets.get("Match list path"))

## cretaion dict topic
word_set_dict, negate_dict = create_topic_dict(match_df)

dask_df = dask_compute_with_progress(curr_df)


# ### 
row_transformations2 = [('matches_' + label, 
                        lambda x: match_count_lowStat(x, word_set_dict, suppress = negate_dict), True)
                        for label in ['FILT_MD', 'FILT_QA']]

dask_df = df_apply_transformations(dask_df, row_transformations2)

#Generates a report based on top matches for a given topic.
generate_topic_report(dask_df, word_set_dict, label_column='matches_')




# df_apply_transformations(dask_df,  row_transformations2)

# @hydra.main(config_path="config", config_name="config")
# def pipeline_one(cfg):
#     setup_logging()
    
#     # Step 1: Initialize Dask Client
#     dask_client = initialize_dask_client(config.processing.n_tasks)

#     # Step 2: Data Extraction
#     db_helper = DBFShelper()
#     data_extractor = DataExtraction(db_helper)
#     transcripts_query = cfg.database.transcripts_query_one
#     currdf = data_extractor.get_snowflake_data(transcripts_query)

#     # Step 3: Convert Pandas to Dask DataFrame
#     import dask.dataframe as dd
#     dask_df = dd.from_pandas(currdf, npartitions=cfg.processing.n_tasks)

#     # Step 4: Compute with Dask
#     computed_df = dask_compute_with_progress(dask_df, use_progress=True)

#     # Step 5: Match Operations
#     match_df = pd.read_csv(cfg.paths.match_list_path)
#     word_set_dict = {topic.replace(' ', '_').upper(): get_match_set(match_df[(match_df['label'] == topic) & (match_df['negate'] == False)]['match'].values) for topic in match_df['label'].unique()}
    
#     # Apply match counts
#     computed_df['matches_filt_md'] = computed_df['FILT_MD'].apply(lambda x: match_count_lowStat(x, word_set_dict), meta=('matches_filt_md', object))

#     # Step 6: Report Generation
#     computed_df = generate_topic_report(computed_df, word_set_dict)

#     # Generate and Save Specific Reports
#     recovery_report = generate_top_matches_report(computed_df, topic="RECOVERY", label="FILT_MD", sort_by="RECOVERY_REL_FILT_MD", top_n=100)
#     save_report_to_csv(recovery_report, cfg.paths.recovery_report_path)

# if __name__ == "__main__":
#     pipeline_one()
