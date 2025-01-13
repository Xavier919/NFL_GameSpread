import logging
import sys
from pathlib import Path

def setup_logger(name: str, clear_existing_handlers: bool = False) -> logging.Logger:
    """Setup logger with consistent configuration"""
    logger = logging.getLogger(name)
    
    # Clear existing handlers if requested
    if clear_existing_handlers:
        logger.handlers.clear()
    
    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger