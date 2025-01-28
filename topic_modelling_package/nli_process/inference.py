# topic_modelling_package/nli_process/inference.py

from typing import Iterator, List, Tuple, Dict, Any

from functools import partial
import pandas as pd
from loguru import logger

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType, StructType, StructField, TimestampType, MapType
from centralized_nlp_package.nli_utils import initialize_nli_infer_pipeline
from centralized_nlp_package.data_processing import sparkdf_apply_transformations


# Initialize the global nli_pipeline variable
nli_pipeline = None


def inference_summary(
    texts: List[str],
    inference_result: List[List[Dict[str, Any]]],
    labels: List[str],
    threshold: float = 0.8
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    """
    Summarizes inference results by aggregating scores and binary flags based on a threshold.
    
    Args:
        texts (List[str]): A list of text pairs in the format "text1</s></s>label.".
        inference_result (List[List[Dict[str, Any]]]): A list where each element corresponds to the inference results for a text pair.
        labels (List[str]): A list of label strings corresponding to each text pair.
        threshold (float, optional): The score threshold to determine binary flags. Defaults to 0.8.
    
    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
            - total_dict: A dictionary mapping each label to a list of binary flags indicating if the score exceeds the threshold.
            - score_dict: A dictionary mapping each label to a list of scores.
    
    Example:
        >>> texts = ["I love this product</s></s>positive.", "This is bad</s></s>negative."]
        >>> inference_result = [
        ...     [{"label": "entailment", "score": 0.9}, {"label": "contradiction", "score": 0.1}],
        ...     [{"label": "entailment", "score": 0.7}, {"label": "contradiction", "score": 0.3}]
        ... ]
        >>> labels = ["positive.", "negative."]
        >>> total, scores = inference_summary(texts, inference_result, labels, threshold=0.8)
        >>> print(total)
        {'positive.': [1], 'negative.': [0]}
        >>> print(scores)
        {'positive.': [0.9], 'negative.': [0.7]}
    """
    score_dict: Dict[str, List[float]] = {label+ '.': [] for label in labels}
    total_dict: Dict[str, List[int]] = {label+ '.': [] for label in labels}
    
    for text_pair, inference in zip(texts, inference_result):
        if not text_pair:
            logger.warning("Empty text pair encountered. Skipping.")
            continue
        try:
            text1, text2_label = text_pair.split('</s></s>')
        except ValueError as e:
            logger.error(f"Error splitting text pair '{text_pair}': {e}")
            continue
        
        for s in inference:
            if s['label'] == 'entailment':
                score =  s['score']
                total_flag = 1 if score > threshold else 0
                total_dict[text2_label].append(total_flag)
                score_dict[text2_label].append(score)
                logger.debug(
                    f"Label: {text2_label}, Score: {score}, "
                    f"Flag: {total_flag} (Threshold: {threshold})"
                )
    
    logger.info("Inference summary completed.")
    return total_dict, score_dict


def inference_run(
    iterator: Iterator[List[str]],
    nli_pipeline,
    max_length: int = 512,
    batch_size: int = 32,
) -> Iterator[pd.Series]:
    """
    Performs inference on batches of text pairs using the NLI pipeline and yields the results as Pandas Series.
    
    Args:
        iterator (Iterator[List[str]]): An iterator where each element is a list of text pairs.
        max_length (int, optional): Maximum token length for each text input. Defaults to 512.
        batch_size (int, optional): Number of samples per batch for inference. Defaults to 32.
        enable_quantization (bool, optional): Whether to enable model quantization for faster inference. Defaults to False.
    
    Yields:
        Iterator[pd.Series]: An iterator yielding Pandas Series containing inference results for each batch.
    
    Example:
        >>> from topic_modelling_package.nli_process.inference import inference_udf
        >>> texts = [["I love this product</s></s>positive.", "This is bad</s></s>negative."]]
        >>> results = inference_udf(iter(texts))
        >>> for res in results:
        ...     print(res)
        0    [{'label': 'entailment', 'score': 0.9}, {'label': 'contradiction', 'score': 0.1}]
        1    [{'label': 'entailment', 'score': 0.7}, {'label': 'contradiction', 'score': 0.3}]
        dtype: object
    """
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Flatten the list of text pairs in the batch
            batch = batch.tolist()
            flat_text_pairs = [pair for sublist in batch for pair in sublist]
            logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            if flat_text_pairs:
                # Perform inference in batch
                results = nli_pipeline(
                    flat_text_pairs,
                    padding=True,
                    top_k=None,
                    batch_size=batch_size,
                    truncation=True,
                    max_length=max_length
                )
                logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")
            else:
                results = []
                logger.warning(f"Batch {batch_num}: No text pairs to infer.")
            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                pairs = pairs.tolist()
                if pairs:
                    pair_length = len(pairs)
                    split_results.append(results[idx:idx + pair_length])
                    idx += pair_length
                    logger.debug(f"Batch {batch_num}: Split {pair_length} results for current row.")
                else:
                    split_results.append([])
                    logger.debug(f"Batch {batch_num}: No pairs in current row. Appended empty list.")
            
            yield pd.Series(split_results)
        except Exception as e:
            logger.error(f"Error in inference batch {batch_num}: {e}")
            # Yield empty results for this batch to continue processing
            yield pd.Series([[] for _ in batch])

def extract_inf(row, section_len, section, threshold):
    """
    Extracts sentiment information from a row of scores based on a specified threshold.

    This function processes a dictionary of sentiment scores, applying a threshold to determine binary classifications. It calculates counts, relative frequencies, and rounded scores for each sentiment type within a specified section. If the section length is zero, it handles the division by zero by assigning `None` or empty lists accordingly.

    Args:
        row (Dict[str, Any]):
            A dictionary where keys are sentiment types (e.g., 'happy', 'sad') and values are lists of sentiment scores (floats) for each instance.
        section_len (int):
            The total number of instances or sentences in the section being analyzed.
        section (str):
            The identifier for the current section (e.g., 'FILT_MD', 'FILT_QA').
        threshold (float):
            The threshold value above which a sentiment score is considered positive (e.g., 0.8).

    Returns:
        Dict[str, Any]:
            A dictionary containing the following keys for each sentiment type:
                - "{sentiment}_TOTAL_{section}" (List[float]): Binary list indicating whether each score exceeds the threshold.
                - "{sentiment}_COUNT_{section}" (float): Total count of scores exceeding the threshold.
                - "{sentiment}_REL_{section}" (float): Relative frequency of scores exceeding the threshold (count divided by section length).
                - "{sentiment}_SCORE_{section}" (List[float]): List of scores rounded to four decimal places.
            
            Additionally includes:
                - "len" (int): Total number of words in the input.
                - "raw_len" (int): Raw word count based on splitting by spaces.
                - "filt" (str): The original input text.

    Raises:
        None. The function handles potential errors internally and assigns `None` or empty lists where appropriate.

    Example:
        >>> row = {'happy': [0.9, 0.7, 0.85], 'sad': [0.2, 0.1, 0.05]}
        >>> extract_inf(row, section_len=3, section='FILT_MD', threshold=0.8)
        {
            'happy_TOTAL_FILT_MD': [1.0, 0.0, 1.0],
            'happy_COUNT_FILT_MD': 2.0,
            'happy_REL_FILT_MD': 0.6666666666666666,
            'happy_SCORE_FILT_MD': [0.9, 0.7, 0.85],
            'sad_TOTAL_FILT_MD': [0.0, 0.0, 0.0],
            'sad_COUNT_FILT_MD': 0.0,
            'sad_REL_FILT_MD': 0.0,
            'sad_SCORE_FILT_MD': [0.2, 0.1, 0.05],
            'len': 0,  # Assuming len is calculated elsewhere
            'raw_len': 0,  # Assuming raw_len is calculated elsewhere
            'filt': ''
        }
    """
    count_col = {}
    rel_col = {}
    score_col = {}
    total_col = {}

    # set_trace()
    for tp, score in row.items():
        if section_len != 0:
            score_binary = [float(1) if s > threshold else float(0) for s in score]
            total_col[f'{tp}_TOTAL_{section}'] = score_binary 
            count_col[f'{tp}_COUNT_{section}'] = float(sum(score_binary))
            rel_col[f'{tp}_REL_{section}'] = sum(score_binary) / section_len
            score_col[f'{tp}_SCORE_{section}'] = [round(s, 4) for s in score]
        else:
            count_col[f'{tp}_COUNT_{section}'] = None
            rel_col[f'{tp}_REL_{section}'] = None
            total_col[f'{tp}_TOTAL_{section}'] = []
            score_col[f'{tp}_SCORE_{section}'] = []
    # print(count_col.keys())

    return {**count_col, **rel_col, **score_col, **total_col}

def get_section_extract_udf(section, threshold):
    par_func = partial(extract_inf, section = section,threshold = threshold)
    return udf(par_func, MapType(StringType(), StringType()))

def processing_nested_columns(spark_df,
                              fixed_columns = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME','LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA', 'SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_QA'],
                               threshold=0.8):
    """
    Processes nested sentiment score columns in a Spark DataFrame and extracts relevant statistics.

    This function performs the following operations:
        1. Generates UDFs for each sentiment section ('FILT_MD', 'FILT_QA') based on the provided threshold.
        2. Applies these UDFs to extract sentiment information.
        3. Extracts individual sentiment statistics from the resulting nested columns.
        4. Renames columns to remove periods.
        5. Selects and reorders the final set of columns based on predefined fixed columns and extracted sentiment statistics.

    Args:
        spark_df (DataFrame):
            The input Spark DataFrame containing nested sentiment score columns (e.g., 'MD_FINAL_SCORE', 'QA_FINAL_SCORE').
        fixed_columns (List[str], optional):
            A list of columns to retain in the final DataFrame. Defaults to [
                'ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 
                'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD', 
                'LEN_FILT_QA', 'FILT_MD', 'FILT_QA', 
                'SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_QA'
            ].
        threshold (float, optional):
            The threshold value above which a sentiment score is considered positive. Defaults to 0.8.

    Returns:
        DataFrame:
            The processed Spark DataFrame with extracted sentiment statistics as individual columns, and with only the specified fixed columns and sentiment statistics retained.

    Raises:
        Exception:
            If there are issues during UDF application, column extraction, or DataFrame transformations.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = [
        ...     {
        ...         'ENTITY_ID': 'E1', 'CALL_ID': 'C1', 'VERSION_ID': 'V1', 'DATE': '2021-01-01',
        ...         'CALL_NAME': 'Call1', 'COMPANY_NAME': 'CompanyA', 'LEN_FILT_MD': 10, 
        ...         'LEN_FILT_QA': 8, 'FILT_MD': ['filter1', 'filter2'], 'FILT_QA': ['filter3'],
        ...         'SENT_LABELS_FILT_MD': [[1, 0, 1], [0, 1, 0]], 'SENT_LABELS_FILT_QA': [[1], [0]],
        ...         'MD_FINAL_SCORE': {'happy': [0.9, 0.7, 0.85], 'sad': [0.2, 0.1, 0.05]},
        ...         'QA_FINAL_SCORE': {'joyful': [0.95], 'angry': [0.1]}
        ...     }
        ... ]
        >>> spark_df = spark.createDataFrame(data)
        >>> processed_df = processing_nested_columns(spark_df, threshold=0.8)
        >>> processed_df.show(truncate=False)
        +---------+-------+----------+----------+----------+-------------+----------+----------+--------+--------+-----------------------+-----------------------+-------------------------+----------------------+------------------------+------------------------+-------------------------+--------------------------+
        |ENTITY_ID|CALL_ID|VERSION_ID|DATE      |CALL_NAME |COMPANY_NAME |LEN_FILT_MD|LEN_FILT_QA|FILT_MD |FILT_QA |SENT_LABELS_FILT_MD    |SENT_LABELS_FILT_QA    |happy_TOTAL_FILT_MD      |sad_TOTAL_FILT_MD    |happy_REL_FILT_MD       |sad_REL_FILT_MD       |happy_SCORE_FILT_MD     |sad_SCORE_FILT_MD         |
        +---------+-------+----------+----------+----------+-------------+----------+----------+--------+--------+-----------------------+-----------------------+-------------------------+----------------------+------------------------+------------------------+-------------------------+--------------------------+
        |E1       |C1     |V1        |2021-01-01|Call1     |CompanyA     |10        |8         |filter1 |filter3 |[1, 0, 1]              |[1, 0]                 |[1.0, 0.0, 1.0]          |[0.0, 0.0, 0.0]      |0.6666666666666666       |0.0                    |[0.9, 0.7, 0.85]        |[0.2, 0.1, 0.05]           |
        +---------+-------+----------+----------+----------+-------------+----------+----------+--------+--------+-----------------------+-----------------------+-------------------------+----------------------+------------------------+------------------------+-------------------------+--------------------------+
    """
    extract_tranformations = [(f"{ent}_FINAL_SCORE_EXTRACTED", [f"{ent}_FINAL_SCORE",f"LEN_FILT_{ent}"], get_section_extract_udf(f"FILT_{ent}", threshold)) for ent in ['MD', 'QA']]

    spark_df = sparkdf_apply_transformations(spark_df, extract_tranformations)
  
      # Extract the keys from the UDF output and create new columns
    md_final_score_extracted_cols = spark_df.select('MD_FINAL_SCORE_EXTRACTED').first().asDict()['MD_FINAL_SCORE_EXTRACTED'].keys()
    qa_final_score_extracted_cols = spark_df.select('QA_FINAL_SCORE_EXTRACTED').first().asDict()['QA_FINAL_SCORE_EXTRACTED'].keys()


    for col_name in md_final_score_extracted_cols:
        spark_df = spark_df.withColumn(col_name, col('MD_FINAL_SCORE_EXTRACTED').getItem(col_name))

    for col_name in qa_final_score_extracted_cols:
        spark_df = spark_df.withColumn(col_name, col('QA_FINAL_SCORE_EXTRACTED').getItem(col_name))
    spark_df = spark_df.drop('MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED')

    new_columns = [col.replace('.', '') for col in spark_df.columns]
    for old_col, new_col in zip(spark_df.columns, new_columns):
        spark_df = spark_df.withColumnRenamed(old_col, new_col)

    columns_filt =( fixed_columns + 
            [col.replace('.', '') for col in md_final_score_extracted_cols] + 
            [col.replace('.', '') for col in qa_final_score_extracted_cols] )

    spark_df = spark_df.select(*columns_filt)

    return spark_df