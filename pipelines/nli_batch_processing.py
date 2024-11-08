from centralized_nlp_package.data_access import read_from_snowflake, write_dataframe_to_snowflake
from centralized_nlp_package.data_processing import initialize_spark_session, create_spark_udf
from centralized_nlp_package.common_utils import get_date_range, query_constructor
from centralized_nlp_package.nli_utils import initialize_nli_pipeline
from topic_modelling_pakage.nli_process import parse_json_list, create_text_pairs

spark = initialize_spark_session()

### Config
MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]

labels_broadcast = spark.sparkContext.broadcast(LABELS)

# Configuration Flags
ENABLE_QUANTIZATION = True  # Set to False to disable quantization
BATCH_SIZE = 32  # Adjust based on GPU memory and performance


## create udf  
parse_json_udf = create_spark_udf(parse_json_list, 'arr[str]')
create_text_pairs_udf = create_spark_udf(create_text_pairs, 'arr[str]', 'arr[str]')

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



transform_data()
