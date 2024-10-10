# centralized_nlp_package/utils/helpers.py
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from dateutil.relativedelta import relativedelta
from centralized_nlp_package.utils.config import config
from centralized_nlp_package.utils.logging_setup import setup_logging

setup_logging()

def load_file(file_path: str) -> str:
    """
    Loads a SQL query from an external .sql file.
    
    Args:
        file_path (str): Path to the SQL file.
    
    Returns:
        str: The SQL query as a string.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")

def get_date_range(years_back: int = 0, months_back: int = 0) -> Tuple[str, str]:
    """
    Calculate the date range for the query based on months or years.

    Args:
        months_back (int): The number of months to go back from the current date (default is 0).
        years_back (int): The number of years to go back from the current year (default is 0).

    Returns:
        Tuple[str, str]: A tuple containing the minimum and maximum dates in the format 'YYYY-MM-DD'.
    """
    end_date = datetime.now()
    
    # Calculate start date based on months_back
    if months_back > 0:
        start_date = end_date - relativedelta(months=months_back)
    else:
        start_date = end_date

    # Calculate start date based on years_back
    if years_back > 0:
        start_year = end_date.year - years_back
        min_date = f"{start_year}-{end_date.month:02d}-01"
    else:
        min_date = f"{start_date.year}-{start_date.month:02d}-01"

    # Max date is always the start of the current month
    max_date = f"{end_date.year}-{end_date.month:02d}-01"

    return f"'{min_date}'", f"'{max_date}'"


def format_date(date: datetime) -> str:
    """
    Formats a datetime object to 'YYYY-MM-DD' string.

    Args:
        date (datetime): The date to format.

    Returns:
        str: Formatted date string.
    """
    formatted_date = date.strftime('%Y-%m-%d')
    logger.debug(f"Formatted date: {formatted_date}")
    return formatted_date

def construct_model_save_path(template: str, **kwargs) -> Path:
    """
    Constructs the model save path using the provided template and dynamic components.

    Args:
        template (str): Template string with placeholders.
        **kwargs: Variable number of keyword arguments to replace placeholders in the template.

    Returns:
        Path: Constructed file path.
    """
    # Find placeholders in the template (in the form {key})
    placeholders = re.findall(r"\{(\w+)\}", template)

    if not all(placeholder in kwargs for placeholder in placeholders):
        logger.error(f"Template placeholders, not matches with parameters provided.")
        raise ValueError(f"Template placeholders do not match the provided parameters.")
    
    # Construct the path string
    path_str = template.format(**kwargs)
    path = Path(path_str)
    logger.debug(f"Constructed model save path: {path}")
    return path


def query_constructor(query_identifier: str, **kwargs) -> str:
    """
    Constructs a query string with the provided parameters.
    
    Args:
        query_identifier (str): The identifier for the query from the configuration.
        *params: Parameters to replace placeholders in the query.
        config (DictConfig): The Hydra configuration containing query mappings.

    Returns:
        str: The constructed query string with parameters substituted.
    """
    # Load the base query string from Hydra config
    base_query = OmegaConf.select(config.queries.query_mapping, query_identifier)
    
    # Find placeholders in the query string (in the form :paramX)
    placeholders = re.findall(r"{.*}", base_query)

    if not all(placeholder in kwargs for placeholder in placeholders):
        logger.error(f"Query placeholders do not match the provided parameters.")
        raise ValueError(f"Queryplaceholders do not match the provided parameters.")
    
    # Replace placeholders with provided parameters
    base_query = base_query.format(**kwargs)
    
    return base_query