# centralized_nlp_package/utils/helpers.py

from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger

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

def construct_model_save_path(template: str, min_year: str, min_month: str, max_year: str, max_month: str) -> Path:
    """
    Constructs the model save path using the provided template and date components.

    Args:
        template (str): Template string with placeholders.
        min_year (str): Minimum year.
        min_month (str): Minimum month.
        max_year (str): Maximum year.
        max_month (str): Maximum month.

    Returns:
        Path: Constructed file path.
    """
    path_str = template.format(min_year=min_year, min_month=min_month, max_year=max_year, max_month=max_month)
    path = Path(path_str)
    logger.debug(f"Constructed model save path: {path}")
    return path
