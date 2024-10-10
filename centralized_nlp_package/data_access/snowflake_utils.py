# centralized_nlp_package/data_access/snowflake_utils.py
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import pandas as pd
from snowflake.connector import connect
from typing import Any
from pathlib import Path
from loguru import logger
from centralized_nlp_package.utils.config import config
from centralized_nlp_package.utils.logging_setup import setup_logging
from cryptography.fernet import Fernet
from pyspark.sql import SparkSession


setup_logging()

load_dotenv()


ENV = os.getenv('ENVIRONMENT', 'development')


def encrypt_message(message):
    """It encrypts data passed as a parameter to the method. 
    The outcome of this encryption is known as a Fernet token which is basically the ciphertext.

    Parameters:
    argument1 (str): value that needs to be encrypted

    Returns:
    str: encrypted value 

    """
    encoded_message = message.encode()
    fernet_obj= Fernet(os.getenv('FERNET_KEY'))
    encrypted_message = fernet_obj.encrypt(encoded_message)
    return encrypted_message

def decrypt_message(encrypted_message):
    """This method decrypts the Fernet token passed as a parameter to the method. 
    On successful decryption the original plaintext is obtained as a result

    Parameters:
    argument1 (str): encrypted values that needs to be decrypted

    Returns:
    str: decrypted value 

    """
    fernet_obj= Fernet(os.getenv('FERNET_KEY'))    
    decrypted_message = fernet_obj.decrypt(encrypted_message)
    return decrypted_message.decode()


def get_snowflake_connection():
    """
    Establish a connection to Snowflake using configuration settings.
    
    Returns:
        conn: A Snowflake connection object
    """
    snowflake_config = {
        'user': decrypt_message(config.lib_config.development.snowflake.user),
        'password': decrypt_message(config.lib_config.development.snowflake.password),
        'account': config.lib_config.development.snowflake.account,
        'database': config.lib_config.development.snowflake.database,
        'schema': config.lib_config.development.snowflake.schema,
        'timezone': "spark",
        'role': config.lib_config.development.snowflake.role  # Optional if needed
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
        'sfURL': f'{config.lib_config.development.snowflake.account}.snowflakecomputing.com',
        'sfUser': decrypt_message(config.lib_config.development.snowflake.user),
        'sfPassword': decrypt_message(config.lib_config.development.snowflake.password),
        'sfDatabase': config.lib_config.development.snowflake.database,
        'sfSchema': config.lib_config.development.snowflake.schema,
        "sfTimezone": "spark",
        'sfRole': config.lib_config.development.snowflake.role  # Optional if needed
    }
    return  snowflake_options

def read_from_snowflake_spark(query: str, spark: SparkSession) -> 'pyspark.sql.DataFrame':
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