from centralized_nlp_package.nli_utils import initialize_nli_pipeline
import pandas as pd
from loguru import logger

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

def inference_udf(iterator, max_length=512, batch_size=32, enable_quantization=False):
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=enable_quantization)
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing  inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Flatten the list of text pairs in the batch
            flat_text_pairs = [pair for sublist in batch for pair in sublist]
            logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            if flat_text_pairs:
                # Perform inference in batch
                results = nli_pipeline(
                    flat_text_pairs,
                    padding=True,
                    top_k=None,
                    batch_size=batch_size,  # Adjusted batch size
                    truncation=True,
                    max_length=max_length
                )
                logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")
            else:
                results = []
            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                if pairs:
                    split_results.append(results[idx:idx+len(pairs)])
                    idx += len(pairs)
                else:
                    split_results.append([])
            
            yield pd.Series(split_results)
        except Exception as e:
            logger.error(f"Error in MD inference batch {batch_num}: {e}")
            # Yield empty results for this batch to continue processing
            yield pd.Series([[] for _ in batch])