# centralized_nlp_package/utils/helpers.py

from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger
from datetime  import 

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
