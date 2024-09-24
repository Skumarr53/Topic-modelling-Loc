# centralized_nlp_package/tests/test_data_access.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from centralized_nlp_package.data_access.snowflake_utils import read_from_snowflake
from centralized_nlp_package.utils.config import Config, SnowflakeConfig

@pytest.fixture
def mock_config():
    return Config(
        snowflake=SnowflakeConfig(
            user="test_user",
            password="test_pass",
            account="test_account",
            warehouse="test_warehouse",
            database="test_db",
            schema="test_schema"
        ),
        dask=None,
        word2vec=None,
        preprocessing=None
    )

@patch('centralized_nlp_package.data_access.snowflake_utils.connect')
def test_read_from_snowflake(mock_connect, mock_config):
    # Mock the connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    # Mock the pandas.read_sql function
    sample_data = {'FILT_DATA': ['data1', 'data2'], 'ENTITY_ID': [1, 2]}
    mock_cursor.read_sql.return_value = pd.DataFrame(sample_data)

    # Define a sample query
    query = "SELECT FILT_DATA, ENTITY_ID FROM test_table;"

    # Call the function
    df = read_from_snowflake(query, mock_config)

    # Assertions
    mock_connect.assert_called_once_with(
        user=mock_config.snowflake.user,
        password=mock_config.snowflake.password,
        account=mock_config.snowflake.account,
        warehouse=mock_config.snowflake.warehouse,
        database=mock_config.snowflake.database,
        schema=mock_config.snowflake.schema
    )
    mock_conn.cursor.assert_called_once()
    mock_cursor.read_sql.assert_called_once_with(query, mock_conn)
    assert isinstance(df, pd.DataFrame)
    assert df.equals(pd.DataFrame(sample_data))
