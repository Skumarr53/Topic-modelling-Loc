# centralized_nlp_package/pipelines/input_preparation_pipeline1.py

import pandas as pd
import itertools
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dask.distributed import Client
from loguru import logger
from pathlib import Path
import ast
import gc

from ..utils.config import Config, get_config
from ..utils.logging_setup import setup_logging
from ..utils.helpers import format_date, construct_model_save_path
from ..data_access.snowflake_utils import read_from_snowflake
from ..preprocessing.text_processing import initialize_spacy_model, word_tokenize
from ..embedding.word2vec_model import train_word2vec, save_model

def run_pipeline1(config: Config) -> None:
    """
    Executes the first input preparation pipeline:
    - Retrieves data from Snowflake.
    - Processes and cleans the data.
    - Trains and saves a Word2Vec model.

    Args:
        config (Config): Configuration object containing all necessary settings.
    """
    setup_logging()
    logger.info("Starting Pipeline 1: Model Inputs Preparation")
    
    # Initialize Dask client
    client = Client(n_workers=config.dask.n_workers)
    logger.info(f"Dask client initialized with {config.dask.n_workers} workers.")
    
    # Retrieve data from Snowflake
    data_end_date = datetime.now()
    data_start_date = data_end_date - relativedelta(years=5)
    
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
    feed = list(itertools.chain.from_iterable(currdf['FILT_DATA'].tolist()))
    logger.info(f"Number of sentences across all documents: {len(feed)}")
    
    gc.collect()
    
    # Train Word2Vec model
    model_params = {
        'vector_size': config.word2vec.vector_size,
        'window': config.word2vec.window,
        'min_count': config.word2vec.min_count,
        'workers': config.word2vec.workers,
        'epochs': config.word2vec.epochs
    }
    
    model = train_word2vec(feed, config.word2vec.gen_bigram, model_params)
    
    # Save the model
    min_year = minDateNewQuery[2:4]
    min_month = minDateNewQuery[5:7]
    max_year = maxDateNewQuery[2:4]
    max_month = maxDateNewQuery[5:7]
    
    model_save_path = construct_model_save_path(
        config.word2vec.model_save_path,
        min_year=min_year,
        min_month=min_month,
        max_year=max_year,
        max_month=max_month
    )
    
    save_model(model, str(model_save_path))
    
    logger.info(f"Word2Vec model saved to {model_save_path}")
    
    client.close()
    logger.info("Dask client closed. Pipeline 1 completed successfully.")

if __name__ == "__main__":
    config = get_config()
    run_pipeline1(config)
