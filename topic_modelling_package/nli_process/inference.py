

def inference_summary(texts, inference_result, labels, threshold=0.8):
    score_dict = {tp + '.': [] for tp in labels}
    total_dict = {tp + '.': [] for tp in labels}
    
    for i, (text_pair, inference) in enumerate(zip(texts, inference_result)):
        if not text_pair:
            continue
        text1, text2_label = text_pair.split('</s></s>')
        for s in inference:
            if s['label'] == 'entailment':
                if s['score'] > threshold:
                    total_dict[text2_label].append(1)
                else:
                    total_dict[text2_label].append(0)
                score_dict[text2_label].append(s['score'])
    return total_dict, score_dict

summary_schema = StructType([
    StructField("total_dict", F.MapType(StringType(), ArrayType(IntegerType())), False),
    StructField("score_dict", F.MapType(StringType(), ArrayType(FloatType())), False)
])

def create_summary_udf(labels):
    from pyspark.sql.functions import udf
    import ast
    
    def summary_udf_func(texts, inference_result):
        return inference_summary(texts, inference_result, labels)
    
    return udf(summary_udf_func, summary_schema)

def create_inference_udf(spark, labels, enable_quantization=ENABLE_QUANTIZATION):
    nli_pipeline = None
    
    @pandas_udf(inference_schema, PandasUDFType.SCALAR)
    def inference_udf(texts: pd.Series) -> pd.Series:
        nonlocal nli_pipeline
        if nli_pipeline is None:
            nli_pipeline = initialize_nli_pipeline(enable_quantization=enable_quantization)
    
        # Prepare the texts for inference
        text_list = texts.tolist()
        flat_text_pairs = [pair for sublist in text_list for pair in sublist]
        logger.debug(f"Total text pairs to infer: {len(flat_text_pairs)}")
        
        # Perform inference in batch
        results = nli_pipeline(
            flat_text_pairs,
            padding=True,
            top_k=None,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=512
        )
        
        # Split results back to original rows
        split_results = []
        idx = 0
        for pairs in text_list:
            if len(pairs):
                split_results.append(results[idx:idx+len(pairs)])
                idx += len(pairs)
            else:
                split_results.append([])
    
        return pd.Series(split_results)
    
    return inference_udf