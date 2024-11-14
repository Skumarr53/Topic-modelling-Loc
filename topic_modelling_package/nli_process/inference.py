# topic_modelling_package/nli_process/inference.py

from typing import Iterator, List, Tuple, Dict, Any

import pandas as pd
from loguru import logger

from centralized_nlp_package.nli_utils import initialize_nli_pipeline


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
    score_dict: Dict[str, List[float]] = {label: [] for label in labels}
    total_dict: Dict[str, List[int]] = {label: [] for label in labels}
    
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
            if s.get('label') == 'entailment':
                score = s.get('score', 0.0)
                total_flag = 1 if score > threshold else 0
                total_dict[text2_label].append(total_flag)
                score_dict[text2_label].append(score)
                logger.debug(
                    f"Label: {text2_label}, Score: {score}, "
                    f"Flag: {total_flag} (Threshold: {threshold})"
                )
    
    logger.info("Inference summary completed.")
    return total_dict, score_dict


def inference_udf(
    iterator: Iterator[List[str]],
    max_length: int = 512,
    batch_size: int = 32,
    enable_quantization: bool = False
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
    global nli_pipeline
    if nli_pipeline is None:
        logger.info("Initializing NLI pipeline.")
        nli_pipeline = initialize_nli_pipeline(enable_quantization=enable_quantization)
        logger.info("NLI pipeline initialized.")
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing inference batch {batch_num} with {len(batch)} rows.")
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
