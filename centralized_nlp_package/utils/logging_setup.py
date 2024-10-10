# centralized_nlp_package/utils/logging_setup.py

from loguru import logger
from pathlib import Path
import os, sys

def setup_logging(log_file_path: str = "logs/log_file.log", log_level: str = "INFO") -> None:
    """
    Configures Loguru logger.

    Args:
        log_level (str): Logging level (e.g., "INFO", "DEBUG").
    """
    # Ensure the logs directory exists
    log_directory = Path(log_file_path).parent
    os.makedirs(log_directory, exist_ok=True)

    # Remove the default logger to prevent duplicate logs
    logger.remove()

    # Add a console sink with colored output
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<10}</level> | <level>{message}</level>",
        diagnose=True,  # To include detailed information about where the log is coming from
        backtrace=True,  # Provides a backtrace for error messages
        enqueue=True  # Makes the logging calls thread-safe
    )

    # Add a file sink with daily rotation and retention policy
    logger.add(
        log_file_path,
        rotation="1 day",       # Rotate log file daily
        retention="7 days",     # Keep logs for 7 days
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<10} | {message}",
        diagnose=True, 
        backtrace=True

    )