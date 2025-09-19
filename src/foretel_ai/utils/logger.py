"""
Professional logging system for ForeTel.AI application.

This module provides a centralized logging configuration with support for
file and console logging, log rotation, and different log levels.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from ..config.settings import config, LogLevel

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)

class ForeTelLogger:
    """Professional logging system for ForeTel.AI."""
    
    _instance: Optional['ForeTelLogger'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ForeTelLogger':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logging system."""
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path(config.logging.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(config.logging.level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler with rotation
        if config.logging.log_to_file:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=config.logging.log_file_path,
                maxBytes=config.logging.max_log_size_mb * 1024 * 1024,
                backupCount=config.logging.backup_count,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                fmt=config.logging.format,
                datefmt=config.logging.date_format
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(config.logging.level.value)
            root_logger.addHandler(file_handler)
        
        # Console handler with colors
        if config.logging.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                fmt=config.logging.format,
                datefmt=config.logging.date_format
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(config.logging.console_level.value)
            root_logger.addHandler(console_handler)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        return logging.getLogger(name)
    
    @staticmethod
    def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None) -> None:
        """Log function calls for debugging."""
        logger = logging.getLogger(__name__)
        kwargs = kwargs or {}
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
    
    @staticmethod
    def log_performance(operation: str, duration: float, **metadata) -> None:
        """Log performance metrics."""
        logger = logging.getLogger("performance")
        logger.info(f"Operation '{operation}' completed in {duration:.4f}s", extra=metadata)
    
    @staticmethod
    def log_model_metrics(model_name: str, metrics: dict) -> None:
        """Log model performance metrics."""
        logger = logging.getLogger("models")
        logger.info(f"Model '{model_name}' metrics: {metrics}")
    
    @staticmethod
    def log_user_action(action: str, user_id: str = "anonymous", **context) -> None:
        """Log user actions for analytics."""
        logger = logging.getLogger("user_actions")
        logger.info(f"User '{user_id}' performed '{action}'", extra=context)
    
    @staticmethod
    def log_error_with_context(error: Exception, context: dict = None) -> None:
        """Log errors with additional context."""
        logger = logging.getLogger("errors")
        context = context or {}
        logger.error(f"Error: {str(error)}", extra=context, exc_info=True)

# Initialize the logging system
logger_instance = ForeTelLogger()

# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return ForeTelLogger.get_logger(name)

def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None) -> None:
    """Log function calls."""
    ForeTelLogger.log_function_call(func_name, args, kwargs)

def log_performance(operation: str, duration: float, **metadata) -> None:
    """Log performance metrics."""
    ForeTelLogger.log_performance(operation, duration, **metadata)

def log_model_metrics(model_name: str, metrics: dict) -> None:
    """Log model metrics."""
    ForeTelLogger.log_model_metrics(model_name, metrics)

def log_user_action(action: str, user_id: str = "anonymous", **context) -> None:
    """Log user actions."""
    ForeTelLogger.log_user_action(action, user_id, **context)

def log_error_with_context(error: Exception, context: dict = None) -> None:
    """Log errors with context."""
    ForeTelLogger.log_error_with_context(error, context)

# Module logger
logger = get_logger(__name__)
logger.info("ForeTel.AI logging system initialized")

__all__ = [
    "get_logger",
    "log_function_call", 
    "log_performance",
    "log_model_metrics",
    "log_user_action",
    "log_error_with_context",
    "ForeTelLogger"
]
