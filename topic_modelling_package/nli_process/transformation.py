from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from loguru import logger
from centralized_nlp_package.data_processing import sparkdf_apply_transformations


def rename_columns_by_label_matching(df: DataFrame, labels_mapping: dict) -> DataFrame:
    """
    Renames columns in a Spark DataFrame based on a provided mapping.

    Args:
        df (DataFrame): The Spark DataFrame whose columns are to be renamed.
        labels_mapping (dict): A dictionary mapping original labels (substrings) 
                               to new labels. If a column name contains an 
                               original label, it will be replaced with the new label.

    Returns:
        DataFrame: A new Spark DataFrame with renamed columns.

    Raises:
        ValueError: If the input DataFrame is not valid or if the labels_mapping is not a dictionary.
        Exception: If the renaming process fails for any other reason.
    """
    if not isinstance(df, DataFrame):
        raise ValueError("The provided input is not a valid Spark DataFrame.")
    
    if not isinstance(labels_mapping, dict):
        raise ValueError("The labels_mapping must be a dictionary.")

    try:
        # Create a mapping for renaming
        new_column_names = {}
        for old_col in df.columns:
            for original_label, new_label in labels_mapping.items():
                if original_label in old_col:
                    new_column_names[old_col] = old_col.replace(original_label, new_label)
                    break  # Break after the first match to avoid multiple replacements
        
        # Rename columns if any new names are generated
        for old_col, new_col in new_column_names.items():
            df = df.withColumnRenamed(old_col, new_col)
        
        logger.info("Columns renamed successfully.")
        return df

    except Exception as e:
        logger.error(f"Failed to rename columns: {e}")
        raise e
    


def convert_column_types(df: DataFrame, float_type_cols: list, array_type_cols: list):
    ## TODO: Add more types and highly customized types
    from utils import literal_eval_udf
    try:
        for col_name in float_type_cols:
            df = df.withColumn(col_name, F.lit(literal_eval_udf(F.col(col_name))))
        
        for col_name in array_type_cols:
            df = df.withColumn(col_name, F.lit(literal_eval_udf(F.col(col_name))))
        
        logger.info("Column types converted successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to convert column types: {e}")
        raise e
    



def transform_data(df: DataFrame, labels_broadcast):
    try:
        df = sparkdf_apply_transformations(
            df,
            [
                ("FILT_MD", "FILT_MD", parse_json_udf),
                ("FILT_QA", "FILT_QA", parse_json_udf),
                ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", parse_json_udf),
                ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", parse_json_udf),
                ("FILT_MD", "LEN_FILT_MD", F.size),
                ("FILT_QA", "LEN_FILT_QA", F.size),
            ],
        )
        df = df.dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC']).orderBy(F.col('UPLOAD_DT_UTC').asc())

        # Create text pairs for MD and QA
        create_text_pairs_udf = udf(lambda filt: create_text_pairs(filt, labels_broadcast.value), ArrayType(StringType()))
        df = df.withColumn('TEXT_PAIRS_MD', create_text_pairs_udf(col('FILT_MD'))) \
               .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf(col('FILT_QA')))

        logger.info("Data transformation completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise e
