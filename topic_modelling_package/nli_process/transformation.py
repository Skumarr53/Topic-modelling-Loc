# topic_modelling_package/nli_process/transformation.py

import ast
import re
from typing import List, Dict, Any, Callable

from pyspark.sql import DataFrame, functions as F, types as T
#from loguru import logger

from pyspark.sql.functions import udf, col
from pyspark.sql.types import  FloatType, ArrayType, MapType, DoubleType


def apply_extract_udf_sections(
    df: DataFrame,
    extract_udf: Callable[[Any, Any], Any],
    patterns: List[str] = ['MD', 'QA']
) -> DataFrame:
    """
    Applies a specified UDF to DataFrame columns matching predefined patterns based on provided suffixes.

    This function performs the following operations for each pattern in the provided list:
    1. Identifies all columns in the DataFrame that end with "_TOTAL_FILT_{pattern}".
    2. For each matched column, replaces the suffix "_TOTAL_FILT_{pattern}" with "_EXTRACT_FILT_{pattern}" to create a new column name.
    3. Applies the provided UDF to create the new column based on a base column "FILT_{pattern}" and the matched column.

    Args:
        df (DataFrame): The input Spark DataFrame to process.
        extract_udf (Callable[[Any, Any], Any]): The User-Defined Function to apply. It should accept two columns as input.
        patterns (List[str], optional): A list of suffixes (e.g., ['MD', 'QA']) to define column naming patterns. Defaults to ['MD', 'QA'].

    Returns:
        DataFrame: A new Spark DataFrame with the newly created columns.

    Raises:
        Exception: If any error occurs during the processing steps.

    Example:
        >>> from pyspark.sql.types import StringType
        >>> from pyspark.sql.functions import udf
        
        ... # Define the extract UDF
        >>> def extract_function(base, total_filt):
        ... # Example implementation of the UDF
        ... return f"{base}_{total_filt}"

        >>> extract_udf = udf(extract_function, StringType())

        ... # Define patterns
        >>> patterns = ['MD', 'QA']

        ... # Apply the function
        >>> processed_df = apply_extract_udf_sections(currdf_spark, extract_udf, patterns)
    """
    try:
        for pattern in patterns:
            # Define the regex pattern to match columns ending with "_TOTAL_FILT_{pattern}"
            regex_pattern = re.compile(rf".*_TOTAL_FILT_{re.escape(pattern)}$")

            # Define the suffix to replace and the base column
            replace_suffix = f"_EXTRACT_FILT_{pattern}"
            base_column = f"FILT_{pattern}"

            print("Processing pattern: {pattern}")

            # Find all columns matching the current pattern
            matched_columns = [c for c in df.columns if regex_pattern.match(c)]
            print("Matched columns for pattern '{pattern}': {matched_columns}")

            for col_name in matched_columns:
                # Generate the new column name by replacing the suffix
                new_col_name = col_name.replace(f"_TOTAL_FILT_{pattern}", replace_suffix)
                print("Creating new column '{new_col_name}' from '{col_name}' using base column '{base_column}'")

                # Apply the UDF to create the new column
                df = df.withColumn(new_col_name, extract_udf(F.col(base_column), F.col(col_name)))
                print("Added new column '{new_col_name}' to DataFrame.")

        print("Scores extracted successfully.")
        return df

    except Exception as e:
        print("Error in apply_extract_udf_sections: {e}")
        raise e


def rename_columns_by_label_matching(
    df: DataFrame,
    labels_mapping: Dict[str, str]
) -> DataFrame:
    """
    Renames columns in a Spark DataFrame based on a provided mapping.

    Args:
        df (DataFrame): The Spark DataFrame whose columns are to be renamed.
        labels_mapping (Dict[str, str]): A dictionary mapping original labels (substrings) to new labels.
                                         If a column name contains an original label, it will be replaced with the new label.

    Returns:
        DataFrame: A new Spark DataFrame with renamed columns.

    Raises:
        ValueError: If the input DataFrame is not valid or if the labels_mapping is not a dictionary.
        Exception: If the renaming process fails for any other reason.

    Example:
        >>> labels_mapping = {
        ...    "old_label1": "new_label1",
        ...    "old_label2": "new_label2"
        ... }
        >>> renamed_df = rename_columns_by_label_matching(df_spark, labels_mapping)
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
            print("Renamed column '{old_col}' to '{new_col}'")

        print("Columns renamed successfully.")
        return df

    except Exception as e:
        print("Failed to rename columns: {e}")
        raise e


def literal_eval_safe(data_str):
    try:
        return ast.literal_eval(data_str)
    except (ValueError, SyntaxError):
        return None

def convert_column_types(
    df: DataFrame,
    float_type_cols: List[str],
    array_type_cols: List[str]
) -> DataFrame:
    """
    Converts specified columns in a Spark DataFrame to float and array types.

    Currently supports converting columns to 'double' and 'array<double>' types.
    TODO: Add support for more types and highly customized type conversions.

    Args:
        df (DataFrame): The input Spark DataFrame.
        float_type_cols (List[str]): List of column names to convert to float (double) type.
        array_type_cols (List[str]): List of column names to convert to array of doubles.

    Returns:
        DataFrame: A new Spark DataFrame with converted column types.

    Raises:
        Exception: If any error occurs during the type conversion process.

    Example:
        >>> float_cols = ['col1', 'col2']
        >>> array_cols = ['col3', 'col4']
        >>> converted_df = convert_column_types(df_spark, float_cols, array_cols)
    """
    # TODO: Add more types and highly customized types
    try:
        # Define UDFs for type conversion
        array_convert_udf = udf(literal_eval_safe, ArrayType(DoubleType()))
        float_convert_udf = udf(literal_eval_safe, DoubleType())

        # Convert float type columns
        for col_name in float_type_cols:
            if col_name in df.columns:
                df = df.withColumn(col_name, float_convert_udf(F.col(col_name)))
                print("Converted column '{col_name}' to double type.")
            else:
                print("Column '{col_name}' not found in DataFrame. Skipping conversion.")

        # Convert array type columns
        for col_name in array_type_cols:
            if col_name in df.columns:
                df = df.withColumn(col_name, array_convert_udf(F.col(col_name)))
                print("Converted column '{col_name}' to array<double> type.")
            else:
                print("Column '{col_name}' not found in DataFrame. Skipping conversion.")

        print("Column types converted successfully.")
        return df

    except Exception as e:
        print("Failed to convert column types: {e}")
        raise e

def process_nested_columns(
    df: DataFrame,
    nested_columns: List[str],
    fixed_columns: List[str]
) -> DataFrame:
    """
    Processes a Spark DataFrame by extracting nested column keys, renaming columns, and selecting specific columns.

    This function performs the following operations:
    1. Extracts keys from specified nested columns and creates new columns for each key.
    2. Drops the original nested columns.
    3. Renames all columns by removing any dots (`.`) in their names.
    4. Selects a subset of columns, including fixed columns and the newly extracted columns.

    Args:
        df (DataFrame): The input Spark DataFrame to process.
        nested_columns (List[str]): A list of nested column names (as strings) to extract and flatten.
        fixed_columns (List[str]): A list of column names (as strings) to retain without modification.

    Returns:
        DataFrame: A new Spark DataFrame with extracted and renamed columns, containing only the specified subset of columns.

    Raises:
        Exception: If any error occurs during the processing steps.

    Example:
        >>> fixed_cols = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 
        ...           'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA']
        >>> nested_cols = ['MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED']
        >>> processed_df = process_nested_columns(currdf_spark, nested_cols, fixed_cols)
    """
    try:
        # Step 1: Extract keys from nested columns and create new columns
        for nested_col in nested_columns:
            print("Processing nested column: {nested_col}")

            # Extract the keys from the first row to determine the new column names
            first_row = df.select(nested_col).first()
            if first_row is None or first_row[nested_col] is None:
                print("No data found in column '{nested_col}'. Skipping extraction.")
                continue

            # Determine if the nested column is a MapType or StructType
            if isinstance(first_row[nested_col], dict):
                extracted_keys = first_row[nested_col].keys()
            else:
                # Attempt to get keys assuming StructType
                try:
                    extracted_keys = first_row[nested_col].asDict().keys()
                except AttributeError:
                    print("Unsupported nested column type for '{nested_col}'. Skipping extraction.")
                    continue

            print("Extracted keys from '{nested_col}': {list(extracted_keys)}")

            # Create a new column for each key
            for key in extracted_keys:
                # Optional: Clean key names if necessary (e.g., remove dots)
                clean_key = key.replace('.', '')
                new_col_name = f"{clean_key}"
                df = df.withColumn(new_col_name, F.col(nested_col).getItem(key))
                print("Added new column '{new_col_name}' from '{nested_col}'")

        # Step 2: Drop the original nested columns
        df = df.drop(*nested_columns)
        print("Dropped nested columns: {nested_columns}")

        # Step 3: Rename all columns by removing dots
        new_column_names = [c.replace('.', '') for c in df.columns]
        rename_mapping = dict(zip(df.columns, new_column_names))

        for old_col, new_col in rename_mapping.items():
            if old_col != new_col:
                df = df.withColumnRenamed(old_col, new_col)
                print("Renamed column '{old_col}' to '{new_col}'")

        print("Renamed all columns by removing dots.")

        # Step 4: Prepare the list of columns to select
        # Clean fixed_columns by removing dots, if any
        fixed_columns_clean = [c.replace('.', '') for c in fixed_columns]

        # Combine fixed columns with the extracted columns
        # Assuming extracted columns are already included via renaming
        # To ensure we select only existing columns
        existing_columns = [c for c in fixed_columns_clean if c in df.columns]
        extracted_columns = [c for c in df.columns if c not in fixed_columns_clean]
        columns_to_select = existing_columns + extracted_columns

        missing_columns = set(fixed_columns_clean) - set(existing_columns)
        if missing_columns:
            print("The following fixed columns are missing and will be skipped: {missing_columns}")

        # Select the specified columns
        df_selected = df.select(*columns_to_select)
        print("Selected the specified subset of columns.")

        return df_selected

    except Exception as e:
        print("Failed to process DataFrame: {e}")
        raise e
