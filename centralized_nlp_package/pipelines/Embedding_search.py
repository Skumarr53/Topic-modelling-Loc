# centralized_nlp_package/pipelines/input_preparation_pipeline2.py

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import ast
import gc

from ..utils.config import Config, get_config
from ..utils.logging_setup import setup_logging
from ..utils.helpers import format_date, construct_model_save_path
from ..data_access.snowflake_utils import read_from_snowflake
from ..preprocessing.text_processing import initialize_spacy_model, word_tokenize
from ..preprocessing.ngram_utils import get_model_ngrams
from ..embedding.embedding_utils import embed_text, nearest_neighbors
from ..visualization.umap_viz import umap_viz

def process_dataframe(currdf: pd.DataFrame, text_processor) -> pd.DataFrame:
    """
    Processes the dataframe by formatting columns and applying text processing.
    
    Args:
        currdf (pd.DataFrame): Input DataFrame.
        text_processor (spacy.Language): Initialized SpaCy model.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    logger.info("Processing dataframe.")
    currdf['CALL_ID'] = currdf['CALL_ID'].apply(lambda x: str(x))
    currdf["FILT_DATA"] = currdf.apply(lambda row: ast.literal_eval(row['FILT_MD']) + ast.literal_eval(row['FILT_QA']), axis=1)
    currdf = (
        currdf[['ENTITY_ID', 'FILT_DATA', 'COMPANY_NAME', 'CALL_NAME', 'UPLOAD_DT_UTC', 'EVENT_DATETIME_UTC', 'VERSION_ID', 'CALL_ID']]
        .sort_values(by='UPLOAD_DT_UTC')
        .drop_duplicates(subset=['ENTITY_ID', 'COMPANY_NAME', 'CALL_ID'], keep='first')
    )
    
    # Apply tokenization and lemmatization
    currdf['FILT_DATA'] = currdf['FILT_DATA'].apply(lambda x: word_tokenize(x, text_processor))
    logger.info("Dataframe processing completed.")
    return currdf

def run_pipeline2(config: Config) -> None:
    """
    Executes the second input preparation pipeline:
    - Retrieves and processes data.
    - Generates embeddings.
    - Finds nearest neighbors.
    - Creates UMAP visualizations.
    
    Args:
        config (Config): Configuration object containing all necessary settings.
    """
    setup_logging()
    logger.info("Starting Pipeline 2: Embedding Generation and Visualization")
    
    # Initialize SpaCy model
    nlp = initialize_spacy_model(config)
    
    # Retrieve data from Snowflake (similar to Pipeline 1)
    data_end_date = pd.Timestamp.now()
    data_start_date = data_end_date - pd.DateOffset(years=5)
    
    minDateNewQuery = format_date(data_start_date)
    maxDateNewQuery = format_date(data_end_date)
    logger.info(f"Querying data from {minDateNewQuery} to {maxDateNewQuery}")
    
    tsQuery = (
        f"SELECT FILT_DATA, ENTITY_ID, UPLOAD_DT_UTC, VERSION_ID, EVENT_DATETIME_UTC "
        f"FROM EDS_PROD.QUANT.YUJING_CT_TL_STG_1 "
        f"WHERE EVENT_DATETIME_UTC >= '{minDateNewQuery}' AND EVENT_DATETIME_UTC < '{maxDateNewQuery}';"
    )
    
    resultspkdf = read_from_snowflake(tsQuery, config)
    currdf = resultspkdf.sort_values(by='UPLOAD_DT_UTC').drop_duplicates(
        subset=['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep='first'
    )
    
    currdf['FILT_DATA'] = currdf['FILT_DATA'].apply(ast.literal_eval)
    currdf = process_dataframe(currdf, nlp)
    
    # Assume 'seed' DataFrame is provided or derived from currdf
    # For illustration, creating a sample seed DataFrame
    seed = pd.DataFrame({
        'label': ['topic1', 'topic2'],
        'match': ['sentence one', 'sentence two']
    })
    
    # Load Word2Vec model
    model_save_path = construct_model_save_path(
        config.word2vec.model_save_path,
        min_year=minDateNewQuery[2:4],
        min_month=minDateNewQuery[5:7],
        max_year=maxDateNewQuery[2:4],
        max_month=maxDateNewQuery[5:7]
    )
    model = Word2Vec.load(str(model_save_path))
    logger.info(f"Loaded Word2Vec model from {model_save_path}")
    
    # Embed seed words
    seed['embed'] = seed['match'].apply(lambda x: embed_text(word_tokenize(x, nlp), model))
    seed = seed[seed['embed'].notna()]
    seed['seed'] = True
    logger.info(f"Number of seed embeddings: {len(seed)}")
    
    # Generate UMAP visualization if sufficient seeds
    if len(seed.index) >= 8:
        umap_save_path = Path(config.visualization.umap_save_path).with_suffix('.html')
        umap_viz(seed, marker_size=8, save_to=str(umap_save_path))
        logger.info(f"UMAP visualization saved to {umap_save_path}")
    
    logger.info("Pipeline 2 completed successfully.")

if __name__ == "__main__":
    config = get_config()
    run_pipeline2(config)
