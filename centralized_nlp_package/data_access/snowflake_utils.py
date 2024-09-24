# centralized_nlp_package/data_access/snowflake_utils.py

import pandas as pd
from snowflake.connector import connect
from typing import Any
from pathlib import Path
from ..utils.config import Config
from loguru import logger

def read_from_snowflake(query: str, config: Config) -> pd.DataFrame:
    """
    Executes a SQL query on Snowflake and returns the result as a pandas DataFrame.

    Args:
        query (str): The SQL query to execute.
        config (Config): Configuration object containing Snowflake credentials.

    Returns:
        pd.DataFrame: Query result.
    """
    logger.info("Establishing connection to Snowflake.")
    conn = connect(
        user=config.snowflake.user,
        password=config.snowflake.password,
        account=config.snowflake.account,
        warehouse=config.snowflake.warehouse,
        database=config.snowflake.database,
        schema=config.snowflake.schema
    )
    try:
        logger.debug(f"Executing query: {query}")
        df = pd.read_sql(query, conn)
        logger.info("Query executed successfully.")
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise
    finally:
        conn.close()
        logger.info("Snowflake connection closed.")
    return df
