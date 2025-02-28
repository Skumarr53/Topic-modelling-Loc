# topic_modelling_package/utils/logging.py

from loguru import logger
from pathlib import Path
import os, sys


def setup_logging(env: str  = "dev"):
    """
    Set up the Loguru logger with console and file handlers.
    
    Args:
        log_file_path (str): Path to the log file. Defaults to "logs/log_file.log".
        log_level (str): Logging level. Defaults to "INFO".
    """
    # Remove any existing handlers (useful when setting up logging multiple times in tests or notebooks)
    logger.remove()

    if env == "prod":
        log_level = "INFO"        
    else:
        log_level = "DEBUG"

    # # Ensure log directory exists
    # log_directory = Path(log_file_path).parent
    # os.makedirs(log_directory, exist_ok=True)
    
    # Console Handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:7}</level> | "
            "<cyan>{file}:{line}</cyan> | "
            "{message}"
        ),
        diagnose=True,  # To include detailed information about where the log is coming from
        backtrace=True,  # Provides a backtrace for error messages
        enqueue=True  # Makes the logging calls thread-safe
    )
        
    # # File Handler with Rotation and Retention
    # logger.add(log_file_path, level=log_level, format="{time} | {level:10} | {message}", rotation="10 MB", retention="7 days", compression="zip")
    logger.info("Logging setup completed.")