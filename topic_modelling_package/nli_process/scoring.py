from pyspark import DataFrame
from centralized_nlp_package.data_processing import create_spark_udf

def extract_matched_sentences(sentences, matches):
    if not sentences:
        return None
    return [sentence for sentence, match in zip(sentences, matches) if match == 1]
    
def extract_inf(row, section, section_len, threshold):
    count_col = {}
    rel_col = {}
    score_col = {}
    total_col = {}

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
    
    return {**count_col, **rel_col, **score_col, **total_col}

def extract_scores(df: DataFrame, patterns: list, new_suffix: str, original_suffix: str):
    try:
        extract_udf = create_spark_udf(extract_matched_sentences, 'arr[str]')
        matched_columns = [col for col in df.columns if any(pattern.match(col) for pattern in patterns)]
        
        for col_name in matched_columns:
            if "_TOTAL_FILT_MD" in col_name:
                new_col_name = col_name.replace("_TOTAL_FILT_MD", f"_{new_suffix}_FILT_MD")
                df = df.withColumn(new_col_name, extract_udf(col("FILT_MD"), col(col_name)))
            elif "_TOTAL_FILT_QA" in col_name:
                new_col_name = col_name.replace("_TOTAL_FILT_QA", f"_{new_suffix}_FILT_QA")
                df = df.withColumn(new_col_name, extract_udf(col("FILT_QA"), col(col_name)))
        logger.info("Scores extracted successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to extract scores: {e}")
        raise e