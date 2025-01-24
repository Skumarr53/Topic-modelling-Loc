

import ast
from centralized_nlp_package.data_access import (
    initialize_dask_client,
    read_from_snowflake,
    write_dataframe_to_snowflake
)
from centralized_nlp_package.data_processing import (
    df_apply_transformations,
    dask_compute_with_progress,
    pandas_to_spark,
    convert_columns_to_timestamp
)
from topic_modelling_pakage.reports import create_topic_dict, generate_topic_report, replace_separator_in_dict_words
from centralized_nlp_package.data_processing import check_pd_dataframe_for_records



tsQuery= ("SELECT CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ "
          
   "FROM QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H  t2 where not exists (select 1 from EDS_PROD.QUANT.YUJING_MASS_LABOR_MACRO_DEV_2 t1 where  t1.CALL_ID = CAST(t2.CALL_ID AS VARCHAR(16777216)) and t1.ENTITY_ID = t2.ENTITY_ID and t1.VERSION_ID = t2.VERSION_ID) ORDER BY PARSED_DATETIME_EASTERN_TZ DESC; ")


resultspkdf = read_from_snowflake(tsQuery)


currdf = resultspkdf.toPandas()

check_pd_dataframe_for_records(currdf, datetime_col = 'PARSED_DATETIME_EASTERN_TZ')

currdf['SENT_LABELS_FILT_QA'] = currdf.apply(lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')], axis=1)

currdf['FILT_QA'] = currdf['FILT_QA'].apply(lambda x: [sent for sent in x if not sent.endswith('?')])

col_inti_tranform = [
    ("FILT_MD", "FILT_MD", ast.literal_eval),
    ("FILT_QA", "FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", ast.literal_eval),
    ('LEN_MD', 'FILT_MD', len),
    ('LEN_QA', 'FILT_QA', len),
    ('SENT_LABELS_FILT_QA', ['SENT_LABELS_FILT_QA','FILT_QA'], 
     lambda x,y: [[label for label, sent in zip(labels, sentences) if not sent[0].endswith('?')] for labels, sentences in zip(x, y)]),
     ('FILT_QA', 'FILT_QA', lambda x: [sent for sent in x if not sent.endswith('?')])

]


curr_df = df_apply_transformations(currdf, col_inti_tranform)

nlp = initialize_spacy()


match_df_v0 = df_apply_transformations(match_df_v0, [('Refined Keywords', 'Refined Keywords', ast.literal_eval)])

match_df = match_df_v0[['Subtopic','Refined Keywords']].explode(column='Refined Keywords')
match_df_negate = match_df_v0[~match_df_v0['Negation'].isna()][['Subtopic', 'Negation']]#.apply(lambda x: ast.literal_eval(x['Negation']), axis=1)#.explode(column = 'Negation')
match_df_negate = df_apply_transformations(match_df_negate, [('Negation', 'Negation', ast.literal_eval)])
match_df_negate = match_df_negate.explode(column = 'Negation')
match_df_negate['negate'] = True
match_df_negate = match_df_negate.rename(columns = {'Subtopic': 'label', 'Negation': 'match'})
match_df['negate'] = False
match_df = match_df.rename(columns={'Subtopic':'label', 'Refined Keywords':'match'})
match_df = pd.concat([match_df, match_df_negate])


word_set_dict, negate_dict = create_topic_dict(curr_df)


negate_dict1 = replace_separator_in_dict_words(negate_dict)

currdf = generate_topic_report(curr_df, word_set_dict, negate_dict1, stats_list = ['relevance', 'count', 'sentiment'])


currdf['DATE'] = pd.to_datetime(currdf['DATE'])


spark_df = pandas_to_spark(dask_df)
spark_df = spark_df.replace(np.nan, None)
spark_df = convert_columns_to_timestamp(
                                       spark_df,
                                       {
                                          "DATE": "yyyy-MM-dd",
                                          "PARSED_DATETIME_EASTERN_TZ": "yyyy-MM-dd HH mm ss",
                                          "EVENT_DATETIME_UTC": "yyyy-MM-dd HH mm ss",
                                       },
                                    )
tablename_curr = 'YUJING_MASS_LABOR_MACRO_DEV_2'
write_dataframe_to_snowflake(spark_df, database = 'EDS_PROD', schema = 'QUANT', table_name=tablename_curr)


