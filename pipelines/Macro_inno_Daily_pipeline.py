import ast

from centralized_nlp_package.data_access import (
    initialize_dask_client,
    read_from_snowflake,
    write_dataframe_to_snowflake

)
from centralized_nlp_package.data_processing import check_pd_dataframe_for_records
from centralized_nlp_package.utils import determine_environment


from centralized_nlp_package.text_processing import initialize_spacy
from centralized_nlp_package.data_processing import (
    df_apply_transformations,
    dask_compute_with_progress,
    pandas_to_spark,
    convert_columns_to_timestamp
)
from topic_modelling_pakage.reports import create_topic_dict, generate_topic_report

ENV = determine_environment()

dask_client = initialize_dask_client()

tsQuery= ("select TOP 1200 CALL_ID,ENTITY_ID, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ from EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW t2 where not exists (select 1 from EDS_PROD.QUANT.PARTHA_MACRO_INNO_CTS_STG_1 t1 where  t1.CALL_ID = t2.CALL_ID and t1.ENTITY_ID = t2.ENTITY_ID and t1.VERSION_ID = t2.VERSION_ID) ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;")


resultspkdf = read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

check_pd_dataframe_for_records(currdf, datetime_col = 'PARSED_DATETIME_EASTERN_TZ')


col_inti_tranform = [
    ("FILT_MD", "FILT_MD", ast.literal_eval),
    ("FILT_QA", "FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", ast.literal_eval),
    ('LEN_MD', 'FILT_MD', len),
    ('LEN_QA', 'FILT_QA', len),
]


curr_df = df_apply_transformations(currdf, col_inti_tranform)


nlp = initialize_spacy()


word_set_dict, negate_dict = create_topic_dict(curr_df)

dask_df = dask_compute_with_progress(curr_df)


dask_df = generate_topic_report(dask_df, word_set_dict, negate_dict, stats=['relevance', 'count', 'sentiment', 'net_sentiment'], label_column='matches')


spark_df = pandas_to_spark(dask_df)
spark_df = spark_df.replace(np.nan, None)
spark_df = (convert_columns_to_timestamp(spark_df, {
                                                # 'DATE': 'yyyy-MM-dd',
                                                'PARSED_DATETIME_EASTERN_TZ': 'yyyy-MM-dd HH mm ss',
                                                'EVENT_DATETIME_UTC': 'yyyy-MM-dd HH mm ss'}))

write_dataframe_to_snowflake(spark_df, database = 'EDS_PROD', schema = 'QUANT', table_name='PARTHA_MACRO_INNO_CTS_STG_1')