# centralized_nlp_package/data_access/snowflake_utils.py

import pandas as pd
from snowflake.connector import connect
from typing import Any
from pathlib import Path
from loguru import logger
from centralized_nlp_package.utils.config import Config
from centralized_nlp_package.utils.logging_setup import setup_logging
from pyspark.sql import SparkSession

setup_logging()

def get_snowflake_connection():
    """
    Establish a connection to Snowflake using configuration settings.
    
    Returns:
        conn: A Snowflake connection object
    """
    snowflake_config = {
        'user': config.snowflake.user,
        'password': config.snowflake.password,
        'account': config.snowflake.account,
        'warehouse': config.snowflake.warehouse,
        'database': config.snowflake.database,
        'schema': config.snowflake.schema
    }
    
    conn = connect(**snowflake_config)
    return conn

def read_from_snowflake(query: str) -> pd.DataFrame:
    """
    Executes a SQL query on Snowflake and returns the result as a pandas DataFrame.

    Args:
        query (str): The SQL query to execute.
        config (Config): Configuration object containing Snowflake credentials.

    Returns:
        pd.DataFrame: Query result.
    """
    logger.info("Establishing connection to Snowflake.")
    conn = get_snowflake_connection()
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


def write_to_snowflake(df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
    """
    Writes a pandas DataFrame to a Snowflake table.

    Args:
        df (pd.DataFrame): The DataFrame to write to Snowflake.
        table_name (str): The target table name in Snowflake.
        config (Config): Configuration object containing Snowflake credentials.
        if_exists (str): Behavior if the table already exists:
                         'fail', 'replace', or 'append'. Default is 'append'.

    Returns:
        None
    """
    logger.info("Establishing connection to Snowflake.")
    conn = get_snowflake_connection()

    try:
        logger.info(f"Writing DataFrame to Snowflake table: {table_name}")
        df.to_sql(
            table_name,
            con=conn,
            if_exists=if_exists,
            index=False,
            method='multi'  # Use multi-row inserts for efficiency
        )
        logger.info(f"DataFrame written successfully to {table_name}.")
    except Exception as e:
        logger.error(f"Error writing DataFrame to Snowflake: {e}")
        raise
    finally:
        conn.close()
        logger.info("Snowflake connection closed.")


def get_snowflake_options():
    """
    Returns a dictionary of Snowflake options.
    """
    snowflake_options = {
        'sfURL': f'{config.snowflake.account}.snowflakecomputing.com',
        'sfUser': config.snowflake.user,
        'sfPassword': config.snowflake.password,
        'sfDatabase': config.snowflake.database,
        'sfSchema': config.snowflake.schema,
        'sfWarehouse': config.snowflake.warehouse,
        'sfRole': config.snowflake.role  # Optional if needed
    }
    return  snowflake_options

def read_from_snowflake_spark(query: str, config: Config, spark: SparkSession) -> 'pyspark.sql.DataFrame':
    """
    Executes a SQL query on Snowflake and returns the result as a Spark DataFrame.

    Args:
        query (str): The SQL query to execute.
        config (Config): Configuration object containing Snowflake credentials.
        spark (SparkSession): The active Spark session.

    Returns:
        pyspark.sql.DataFrame: Query result.
    """
    logger.info("Reading data from Snowflake using Spark.")

    snowflake_options = get_snowflake_options()

    try:
        logger.debug(f"Executing query: {query}")
        df_spark = spark.read.format("snowflake") \
            .options(**snowflake_options) \
            .option("query", query) \
            .load()
        logger.info("Query executed successfully and Spark DataFrame created.")
    except Exception as e:
        logger.error(f"Error executing query on Snowflake: {e}")
        raise

    return df_spark

def write_to_snowflake_spark(df: 'pyspark.sql.DataFrame', table_name: str, spark: SparkSession, mode: str = 'append') -> None:
    """
    Writes a Spark DataFrame to a Snowflake table.

    Args:
        df (pyspark.sql.DataFrame): The Spark DataFrame to write to Snowflake.
        table_name (str): The target table name in Snowflake.
        spark (SparkSession): The active Spark session.
        mode (str): Specifies the behavior if the table already exists:
                    'append', 'overwrite', 'error', or 'ignore'. Default is 'append'.

    Returns:
        None
    """
    logger.info(f"Writing Spark DataFrame to Snowflake table: {table_name}.")
    snowflake_options = get_snowflake_options()

    try:
        df.write.format("snowflake") \
            .options(**snowflake_options) \
            .option("dbtable", table_name) \
            .mode(mode) \
            .save()
        logger.info(f"DataFrame written successfully to {table_name}.")
    except Exception as e:
        logger.error(f"Error writing Spark DataFrame to Snowflake: {e}")
        raise
