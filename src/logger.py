"""
Logging configuration for the HieQue framework
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

def get_logger(name: str, 
               level: str = None, 
               log_file: str = None,
               format_string: str = None) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    # Get level from environment or use default
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_file = log_file or os.getenv("LOG_FILE", "logs/hieque.log")
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            # Ensure log directory exists
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, log to console only
            logger.warning(f"Failed to set up file logging: {e}")
    
    return logger


def setup_logging(config: dict = None):
    """
    Setup logging configuration for the entire application
    
    Args:
        config: Dictionary with logging configuration
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": os.getenv("LOG_FILE", "logs/hieque.log"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console_level": "INFO",
        "file_level": "DEBUG"
    }
    
    # Update with provided config
    default_config.update(config)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, default_config["level"].upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(default_config["format"])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, default_config["console_level"].upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if default_config["log_file"]:
        try:
            log_path = Path(default_config["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(getattr(logging, default_config["file_level"].upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Failed to set up file logging: {e}")
    
    return root_logger


class HieQueLogger:
    """
    Enhanced logger class with additional functionality
    """
    
    def __init__(self, name: str, **kwargs):
        self.logger = get_logger(name, **kwargs)
        self.name = name
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(f"PERFORMANCE: {operation} took {duration:.3f}s", **kwargs)
    
    def query(self, query_text: str, method: str, results_count: int, duration: float):
        """Log query execution details"""
        self.logger.info(
            f"QUERY: '{query_text[:100]}...' | Method: {method} | "
            f"Results: {results_count} | Duration: {duration:.3f}s"
        )
    
    def document_processing(self, filename: str, pages: int, duration: float):
        """Log document processing details"""
        self.logger.info(
            f"DOCUMENT: {filename} | Pages: {pages} | Duration: {duration:.3f}s"
        )
    
    def clustering(self, n_components: int, n_documents: int, duration: float):
        """Log clustering operation details"""
        self.logger.info(
            f"CLUSTERING: {n_components} components | {n_documents} documents | "
            f"Duration: {duration:.3f}s"
        )


# Convenience function for quick logging setup
def quick_logger(name: str = None) -> logging.Logger:
    """Quick setup for a logger with default configuration"""
    if name is None:
        name = "hieque"
    return get_logger(name)

