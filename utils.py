"""
Utility functions for ForeTel.AI application
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data_file(file_path: str) -> bool:
    """
    Validate if a data file exists and is readable.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        # Try to read the file
        df = pd.read_csv(file_path, nrows=1)
        return len(df.columns) > 0
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return False

def validate_model_file(file_path: str) -> bool:
    """
    Validate if a model file exists and is loadable.
    
    Args:
        file_path (str): Path to the model file
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        # Try to load the model
        joblib.load(file_path)
        return True
    except Exception as e:
        logger.error(f"Error validating model {file_path}: {e}")
        return False

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division by zero
        
    Returns:
        float: Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, bool]:
    """
    Validate if a DataFrame has required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        Dict[str, bool]: Dictionary with column names as keys and validation status as values
    """
    validation_results = {}
    for col in required_columns:
        validation_results[col] = col in df.columns
    
    return validation_results

def clean_numeric_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric data by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (List[str]): List of numeric columns to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    for col in columns:
        if col in df_cleaned.columns:
            # Handle missing values
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            
            # Handle outliers using IQR method
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
    
    return df_cleaned

def create_directories(directories: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories (List[str]): List of directory paths to create
    """
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

def log_model_performance(model_name: str, metrics: Dict[str, float]) -> None:
    """
    Log model performance metrics.
    
    Args:
        model_name (str): Name of the model
        metrics (Dict[str, float]): Dictionary of performance metrics
    """
    logger.info(f"Model: {model_name}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

def handle_file_upload_error(error: Exception, file_name: str) -> str:
    """
    Handle file upload errors and return user-friendly message.
    
    Args:
        error (Exception): The exception that occurred
        file_name (str): Name of the file being uploaded
        
    Returns:
        str: User-friendly error message
    """
    error_messages = {
        'FileNotFoundError': f"File '{file_name}' not found. Please check the file path.",
        'PermissionError': f"Permission denied accessing '{file_name}'. Please check file permissions.",
        'UnicodeDecodeError': f"Unable to read '{file_name}'. Please ensure it's a valid text file.",
        'pd.errors.EmptyDataError': f"File '{file_name}' is empty or contains no data.",
        'pd.errors.ParserError': f"Unable to parse '{file_name}'. Please check file format."
    }
    
    error_type = type(error).__name__
    return error_messages.get(error_type, f"An unexpected error occurred while processing '{file_name}': {str(error)}")

def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount (float): Amount to format
        
    Returns:
        str: Formatted currency string
    """
    try:
        return f"${amount:,.2f}"
    except (TypeError, ValueError):
        return "$0.00"

def format_percentage(value: float) -> str:
    """
    Format a number as percentage.
    
    Args:
        value (float): Value to format (0-1 range)
        
    Returns:
        str: Formatted percentage string
    """
    try:
        return f"{value * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"
