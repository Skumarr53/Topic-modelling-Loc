# centralized_nlp_package/utils/logging_setup.py

from loguru import logger
import sys

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configures Loguru logger.

    Args:
        log_level (str): Logging level (e.g., "INFO", "DEBUG").
    """
    logger.remove()  # Remove the default logger
    logger.add(sys.stderr, level=log_level, format="<green>{time}</green> | <level>{level}</level> | <level>{message}</level>")
    logger.add("logs/{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days", level=log_level, format="{time} {level} {message}")
