from pyspark.sql.functions import pandas_udf

from functools import partial
import pyspark.sql.functions as F
from centralized_nlp_package.data_access import read_from_snowflake, write_dataframe_to_snowflake
from centralized_nlp_package.data_processing import initialize_spark_session, create_spark_udf
from centralized_nlp_package.common_utils import get_date_range, query_constructor
from centralized_nlp_package.nli_utils import initialize_nli_pipeline
from topic_modelling_pakage.nli_process import parse_json_list, create_text_pairs, processing_nested_columns, convert_column_types
from centralized_nlp_package.data_processing import sparkdf_apply_transformations


spark = initialize_spark_session()

### Config
LABELS_MAPPING = {
    "This text is about consumer strength": "CONSUMER_STRENGTH",
    "This text is about consumer weakness": "CONSUMER_WEAKNESS",
    "This text is about reduced consumer's spending patterns": "CONSUMER_SPENDING_PATTERNS"
}
MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
labels_broadcast = spark.sparkContext.broadcast(LABELS_MAPPING.values())

# Configuration Flags
ENABLE_QUANTIZATION = True  # Set to False to disable quantization
BATCH_SIZE = 32  # Adjust based on GPU memory and performance


## create udf

create_text_pairs_partial = partial(create_text_pairs, labels=labels_broadcast.value)

parse_json_udf = create_spark_udf(parse_json_list, 'arr[str]')
create_text_pairs_udf = create_spark_udf(create_text_pairs_partial, 'arr[str]')

## Initialize NLI pipeline on Spark Session
nli_model = initialize_nli_pipeline(model_path)

## Fetch data
ts_query = f"""
    SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME,
           EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID,
           EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD,
           SENT_LABELS_FILT_QA
    FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW
    WHERE DATE >= '{min_date}' AND DATE < '{max_date}'
    ORDER BY PARSED_DATETIME_EASTERN_TZ DESC
    """
min_dt, max_dt = get_date_range(years_back=1)
query = query_constructor(query_path, min_date = min_dt, max_date = max_dt) 


spark_df = read_from_snowflake(ts_query)

spark_df = (spark_df.dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC'])
            .orderBy(F.F.col('UPLOAD_DT_UTC').asc()))

create_text_pairs_udf

spark_df = sparkdf_apply_transformations(
    spark_df,
    [
        ("FILT_MD", "FILT_MD", parse_json_udf),
        ("FILT_QA", "FILT_QA", parse_json_udf),
        ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", parse_json_udf),
        ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", parse_json_udf),
        ("FILT_MD", "LEN_FILT_MD", F.size),
        ("FILT_QA", "LEN_FILT_QA", F.size),
        ("TEXT_PAIRS_MD", "FILT_MD", create_text_pairs_udf),
        ("TEXT_PAIRS_QA", "FILT_QA", create_text_pairs_udf),
    ],
)


# Define the schema for the inference results
inference_schema = ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
]))


inference_udf_init = partial(inference_udf, max_length=512, batch_size=32, enable_quantization=False) 

infernece_udf_func = pandas_udf(inference_udf_init, inference_schema, functionType=PandasUDFType.SCALAR_ITER)


spark_df = sparkdf_apply_transformations(
    spark_df,
    [
        ("MD_RESULT", "TEXT_PAIRS_MD", infernece_udf_func),
        ("QA_RESULT", "TEXT_PAIRS_QA", infernece_udf_func)
    ])


schema_summary = StructType([
    StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False),
    StructField("score_dict", MapType(StringType(), ArrayType(FloatType())), False)
])

summary_udf = F.udf(lambda texts, inference_result: inference_summary(texts, inference_result), schema_summary)

currdf_spark = currdf_spark \
    .withColumn("MD_SUMMARY", summary_udf(F.col("TEXT_PAIRS_MD"),F.col("MD_RESULT"))) \
    .withColumn("QA_SUMMARY", summary_udf(F.col("TEXT_PAIRS_QA"),F.col("QA_RESULT")))

# #EDITED: Extract summary fields
currdf_spark = (currdf_spark
    .withColumn("MD_FINAL_TOTAL", F.col("MD_SUMMARY.total_dict")) 
    .withColumn("MD_FINAL_SCORE", F.col("MD_SUMMARY.score_dict")) 
    .withColumn("QA_FINAL_TOTAL", F.col("QA_SUMMARY.total_dict")) 
    .withColumn("QA_FINAL_SCORE", F.col("QA_SUMMARY.score_dict")))


currdf_spark = processing_nested_columns(currdf_spark,
                                         nested_columns = ['MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED'],
                                         fixed_columns = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA']
                                         )

float_type_cols = ["This text is about consumer weakness_COUNT_FILT_MD",
                    "This text is about consumer strength_COUNT_FILT_MD",
                    "This text is about reduced consumer's spending patterns_COUNT_FILT_MD",
                    "This text is about consumer weakness_REL_FILT_MD",
                    "This text is about consumer strength_REL_FILT_MD",
                    "This text is about reduced consumer's spending patterns_REL_FILT_MD",
                    "This text is about consumer weakness_COUNT_FILT_QA",
                    "This text is about consumer strength_COUNT_FILT_QA",
                    "This text is about reduced consumer's spending patterns_COUNT_FILT_QA",
                    "This text is about consumer weakness_REL_FILT_QA",
                    "This text is about consumer strength_REL_FILT_QA",
                    "This text is about reduced consumer's spending patterns_REL_FILT_QA"] 

array_type_cols = ["This text is about consumer weakness_SCORE_FILT_MD",
                  "This text is about consumer strength_SCORE_FILT_MD",
                  "This text is about consumer strength_TOTAL_FILT_MD",
                  "This text is about consumer weakness_TOTAL_FILT_MD",
                  "This text is about reduced consumer's spending patterns_TOTAL_FILT_MD",
                  "This text is about reduced consumer's spending patterns_SCORE_FILT_MD",
                  "This text is about consumer weakness_SCORE_FILT_QA",
                  "This text is about consumer strength_SCORE_FILT_QA",
                  "This text is about consumer strength_TOTAL_FILT_QA",
                  "This text is about consumer weakness_TOTAL_FILT_QA",
                  "This text is about reduced consumer's spending patterns_TOTAL_FILT_QA",
                  "This text is about reduced consumer's spending patterns_SCORE_FILT_QA"
                  ]       


float_type_cols = generate_label_columns(LABELS_MAPPING.keys(),[])

currdf_spark = convert_column_types(currdf_spark, 
                                    float_type_cols = float_type_cols, 
                                    float_type_cols = array_type_cols)


# Define the mapping of original labels to new labels



currdf_spark = rename_columns_by_label_matching(currdf_spark, LABELS_MAPPING)

extract_udf = create_spark_udf(extract_matched_sentences, 'arr[str]')


currdf_spark = apply_extract_udf_sections(currdf_spark, extract_udf)


base_columns = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME',
       'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA']

labels = ['CONSUMER_STRENGTH', 'CONSUMER_WEAKNESS', 'CONSUMER_SPENDING_PATTERNS']
metrics = ['COUNT', 'REL', 'SCORE', 'TOTAL']

columns_order = generate_label_columns(LABELS_MAPPING.values(),metrics,include_extract=True)