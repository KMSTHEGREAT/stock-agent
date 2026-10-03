"""
Logging configuration for the stock agent.
"""
import logging
import logging.handlers
from config.settings import LOGGING


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # Already configured
    
    logger.setLevel(getattr(logging, LOGGING["level"]))
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        LOGGING["file"],
        maxBytes=LOGGING["max_bytes"],
        backupCount=LOGGING["backup_count"]
    )
    file_handler.setFormatter(logging.Formatter(LOGGING["format"]))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOGGING["format"]))
    logger.addHandler(console_handler)
    
    return logger