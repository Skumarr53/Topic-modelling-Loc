import ast
import pandas as pd
from reports.report_generation import generate_topic_report, generate_top_matches_report
from utils.dask_utils import initialize_dask_client, dask_compute_with_progress
from centralized_nlp_package.data_access import read_from_snowflake, write_dataframe_to_snowflake
from centralized_nlp_package.utils import determine_environment
from centralized_nlp_package.preprocessing import (initialize_spacy)
from centralized_nlp_package.text_processing import find_ngrams
from centralized_nlp_package.data_processing import concatenate_and_reset_index, pandas_to_spark, initialize_dask_client, convert_columns_to_timestamp
from centralized_nlp_package.utils.helpers import (df_convert_str2py_objects, 
                                                   query_constructor, 
                                                   get_date_range, 
                                                   df_apply_transformations)
from topic_modelling_pakage.reports.report_generation import create_topic_dict, generate_topic_report


ENV = determine_environment()

dask_client = initialize_dask_client()

min_dt, max_dt = get_date_range(months_back=1)

## base query: "select * from table where date between {min_date} and {max_date}"
## query_path: Load  base_query from a file | ex: config/inp_query  
query = query_constructor(query_path, min_date = min_dt, max_date = max_dt) 


curr_df = read_from_snowflake(query)


# row transformation 
## transformed column name as key if column does not exist a new column will be created and function as values
row_transformations1 = [
                            ('CALL_ID', 'CALL_ID', str),
                            ('FILT_MD', 'FILT_MD',ast.literal_eval),
                            ('FILT_QA', 'FILT_QA',ast.literal_eval),
                            ('SENT_LABELS_FILT_QA', 'SENT_LABELS_FILT_QA',ast.literal_eval),
                            ('SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_MD',ast.literal_eval),
                            ('LEN_MD', 'FILT_MD',lambda row: len(row['FILT_MD'])),
                            ('LEN_QA', 'FILT_QA', lambda row: len(row['FILT_QA'])),
                            ('FILT_MD', ['FILT_MD', 'SENT_LABELS_FILT_MD'], lambda x: [y for y,z in zip(x[0], x[1]) if ((z==1) & (not y.endswith('?')))]),
                            ('FILT_QA', ['FILT_QA', 'SENT_LABELS_FILT_QA'], lambda x: [y for y,z in zip(x[0], x[1]) if ((z==1) & (not y.endswith('?')))])
                        ]

curr_df = df_apply_transformations(curr_df, row_transformations1)

nlp = initialize_spacy()

match_df = pd.read_csv(dbutils.widgets.get("Match list path"))

## cretaion dict topic
word_set_dict, negate_dict = create_topic_dict(match_df)

dask_df = dask_compute_with_progress(curr_df)


# ### 
# row_transformations2 = [(topic + '_TOTAL_' + label, 'matches_' + label, lambda x: x[topic]['total'])
#                         for label in ['FILT_MD', 'FILT_QA']]

# dask_df = df_apply_transformations(dask_df, row_transformations2)

#Generates a report based on top matches for a given topic.
dask_df = generate_topic_report(dask_df, word_set_dict, stats=['total', 'stats', 'relevance', , 'extract'], label_column='matches')


rdf = generate_top_matches_report(dask_df, 
                            topic= 'RECOVERY', 
                            sortby = 'RECOVERY_REL_FILT_MD')
cdf = generate_top_matches_report(dask_df, 
                            topic= 'CYCLE', 
                            sortby = 'CYCLE_REL_FILT_MD')
sdf = generate_top_matches_report(dask_df, 
                            topic= 'S&D', 
                            sortby = 'S&D_REL_FILT_MD')


concatdf = concatenate_and_reset_index([rdf, cdf, sdf])

concatdf = concatdf[(concatdf['RECOVERY_REL_FILT_MD']>0) | (concatdf['CYCLE_REL_FILT_MD']>0) | (concatdf['S&D_REL_FILT_MD']>0)]

def simpDict(x):
 # print(type(x))
  return {key : val for key, val in x.items() if val!=0}

stats_col_transform = [(col, col, lambda x: simpDict(x) if type(x)==dict else None) for col in concatdf.columns if 'STATS' in col]

concatdf = df_apply_transformations(concatdf, stats_col_transform)

concatdf['REPORT_DATE'] = pd.to_datetime(max_dt[1:-1])
concatdf.to_csv('/dbfs/mnt/access_work/UC25/SF_Equities_Reports/sf_equities_report_' + str(month) + '_' + str(year) + '.csv')


spark_df = pandas_to_spark(concatdf)
spark_df = spark_df.replace(np.nan, None)
spark_df = (convert_columns_to_timestamp(spark_df, {
                                                # 'DATE': 'yyyy-MM-dd',
                                                'REPORT_DATE': 'yyyy-MM-dd HH mm ss',
                                                'EVENT_DATETIME_UTC': 'yyyy-MM-dd HH mm ss'}))

write_dataframe_to_snowflake(spark_df, database = 'EDS_PROD', schema = 'QUANT', table_name='PARTHA_SF_REPORT_CTS_STG_1')