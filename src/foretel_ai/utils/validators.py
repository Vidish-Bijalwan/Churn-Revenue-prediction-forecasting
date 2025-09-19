"""
Data validation utilities for ForeTel.AI application.

This module provides comprehensive data validation, schema checking,
and data quality assessment functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class DataValidator:
    """
    Comprehensive data validation system for telecom datasets.
    
    Validates data quality, schema compliance, and business rules.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.required_columns = config.data.required_columns
        self.validation_rules = self._setup_validation_rules()
        logger.info("DataValidator initialized")
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Validate a complete dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        result = self.comprehensive_validation(df)
        
        if result.errors:
            logger.error(f"Dataset validation failed: {result.errors}")
            return False
        
        if result.warnings:
            logger.warning(f"Dataset validation warnings: {result.warnings}")
        
        logger.info("Dataset validation passed")
        return True
    
    def comprehensive_validation(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive dataset validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        metrics = {}
        
        # Basic structure validation
        structure_result = self._validate_structure(df)
        errors.extend(structure_result['errors'])
        warnings.extend(structure_result['warnings'])
        metrics.update(structure_result['metrics'])
        
        # Data quality validation
        quality_result = self._validate_data_quality(df)
        errors.extend(quality_result['errors'])
        warnings.extend(quality_result['warnings'])
        metrics.update(quality_result['metrics'])
        
        # Business rules validation
        business_result = self._validate_business_rules(df)
        errors.extend(business_result['errors'])
        warnings.extend(business_result['warnings'])
        metrics.update(business_result['metrics'])
        
        # Statistical validation
        stats_result = self._validate_statistics(df)
        warnings.extend(stats_result['warnings'])
        metrics.update(stats_result['metrics'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_file_upload(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate uploaded file before processing.
        
        Args:
            file_path: Path to uploaded file
            
        Returns:
            ValidationResult for the file
        """
        errors = []
        warnings = []
        metrics = {}
        
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return ValidationResult(False, errors, warnings, metrics)
        
        # Check file extension
        if file_path.suffix.lower() not in config.security.allowed_file_extensions:
            errors.append(f"Invalid file extension: {file_path.suffix}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        metrics['file_size_mb'] = file_size_mb
        
        if file_size_mb > config.security.max_file_size_mb:
            errors.append(f"File too large: {file_size_mb:.2f}MB > {config.security.max_file_size_mb}MB")
        
        # Try to read file structure
        try:
            if file_path.suffix.lower() == '.csv':
                df_sample = pd.read_csv(file_path, nrows=100)
                metrics['columns'] = df_sample.columns.tolist()
                metrics['sample_rows'] = len(df_sample)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df_sample = pd.read_excel(file_path, nrows=100)
                metrics['columns'] = df_sample.columns.tolist()
                metrics['sample_rows'] = len(df_sample)
        except Exception as e:
            errors.append(f"Cannot read file: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules for telecom data."""
        return {
            'age_range': (18, 100),
            'tenure_range': (0, 120),
            'spending_range': (0, 1000),
            'usage_ranges': {
                'DataUsageGB': (0, 1000),
                'CallUsageMin': (0, 10000),
                'SMSUsage': (0, 10000)
            },
            'rating_range': (1, 5),
            'required_dtypes': {
                'Age': ['int64', 'float64'],
                'TenureMonths': ['int64', 'float64'],
                'MonthlySpending': ['float64', 'int64']
            }
        }
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset structure."""
        errors = []
        warnings = []
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': df.columns.tolist()
        }
        
        # Check minimum rows
        if len(df) < 100:
            warnings.append(f"Dataset has only {len(df)} rows, recommend at least 100")
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            errors.append(f"Duplicate column names: {duplicate_columns}")
        
        # Check column naming conventions
        invalid_column_names = [col for col in df.columns if not col.replace('_', '').isalnum()]
        if invalid_column_names:
            warnings.append(f"Non-standard column names: {invalid_column_names}")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        errors = []
        warnings = []
        metrics = {}
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        metrics['missing_values'] = missing_counts.to_dict()
        metrics['missing_percentages'] = missing_percentages.to_dict()
        
        # Check for columns with excessive missing values
        high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
        if high_missing_cols:
            warnings.append(f"Columns with >50% missing values: {high_missing_cols}")
        
        # Check for completely empty columns
        empty_cols = missing_percentages[missing_percentages == 100].index.tolist()
        if empty_cols:
            errors.append(f"Completely empty columns: {empty_cols}")
        
        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        metrics['duplicate_rows'] = duplicate_count
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
        
        # Data type consistency
        for col in df.columns:
            if col in self.validation_rules.get('required_dtypes', {}):
                expected_dtypes = self.validation_rules['required_dtypes'][col]
                if str(df[col].dtype) not in expected_dtypes:
                    warnings.append(f"Column '{col}' has unexpected dtype: {df[col].dtype}")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business-specific rules."""
        errors = []
        warnings = []
        metrics = {}
        
        # Age validation
        if 'Age' in df.columns:
            age_min, age_max = self.validation_rules['age_range']
            invalid_ages = df[(df['Age'] < age_min) | (df['Age'] > age_max)]
            if len(invalid_ages) > 0:
                warnings.append(f"Found {len(invalid_ages)} records with invalid ages")
                metrics['invalid_age_count'] = len(invalid_ages)
        
        # Tenure validation
        if 'TenureMonths' in df.columns:
            tenure_min, tenure_max = self.validation_rules['tenure_range']
            invalid_tenure = df[(df['TenureMonths'] < tenure_min) | (df['TenureMonths'] > tenure_max)]
            if len(invalid_tenure) > 0:
                warnings.append(f"Found {len(invalid_tenure)} records with invalid tenure")
                metrics['invalid_tenure_count'] = len(invalid_tenure)
        
        # Spending validation
        if 'MonthlySpending' in df.columns:
            spending_min, spending_max = self.validation_rules['spending_range']
            invalid_spending = df[(df['MonthlySpending'] < spending_min) | (df['MonthlySpending'] > spending_max)]
            if len(invalid_spending) > 0:
                warnings.append(f"Found {len(invalid_spending)} records with invalid spending")
                metrics['invalid_spending_count'] = len(invalid_spending)
        
        # Usage validation
        for col, (min_val, max_val) in self.validation_rules['usage_ranges'].items():
            if col in df.columns:
                invalid_usage = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(invalid_usage) > 0:
                    warnings.append(f"Found {len(invalid_usage)} records with invalid {col}")
                    metrics[f'invalid_{col.lower()}_count'] = len(invalid_usage)
        
        # Rating validation
        rating_cols = ['NetworkQuality', 'ServiceRating']
        rating_min, rating_max = self.validation_rules['rating_range']
        for col in rating_cols:
            if col in df.columns:
                invalid_ratings = df[(df[col] < rating_min) | (df[col] > rating_max)]
                if len(invalid_ratings) > 0:
                    warnings.append(f"Found {len(invalid_ratings)} records with invalid {col}")
                    metrics[f'invalid_{col.lower()}_count'] = len(invalid_ratings)
        
        # Customer ID uniqueness
        if 'CustomerID' in df.columns:
            duplicate_ids = df['CustomerID'].duplicated().sum()
            if duplicate_ids > 0:
                errors.append(f"Found {duplicate_ids} duplicate Customer IDs")
                metrics['duplicate_customer_ids'] = duplicate_ids
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties."""
        warnings = []
        metrics = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            col_stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            metrics[f'{col}_stats'] = col_stats
            
            # Check for extreme skewness
            if abs(col_stats['skewness']) > 3:
                warnings.append(f"Column '{col}' has extreme skewness: {col_stats['skewness']:.2f}")
            
            # Check for extreme kurtosis
            if abs(col_stats['kurtosis']) > 10:
                warnings.append(f"Column '{col}' has extreme kurtosis: {col_stats['kurtosis']:.2f}")
            
            # Check for potential outliers (values beyond 3 standard deviations)
            if col_stats['std'] > 0:
                outliers = df[abs((df[col] - col_stats['mean']) / col_stats['std']) > 3]
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / len(df)) * 100
                    if outlier_percentage > 5:  # More than 5% outliers
                        warnings.append(f"Column '{col}' has {outlier_percentage:.1f}% potential outliers")
        
        return {'warnings': warnings, 'metrics': metrics}

class ModelValidator:
    """Validator for machine learning models and predictions."""
    
    def __init__(self):
        """Initialize model validator."""
        logger.info("ModelValidator initialized")
    
    def validate_prediction_input(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate input data for predictions.
        
        Args:
            data: Input data dictionary
            
        Returns:
            ValidationResult for the input
        """
        errors = []
        warnings = []
        metrics = {}
        
        # Check required fields for churn prediction
        churn_required = ['Age', 'TenureMonths', 'MonthlySpending', 'NetworkQuality']
        missing_churn = [field for field in churn_required if field not in data]
        if missing_churn:
            errors.append(f"Missing required fields for churn prediction: {missing_churn}")
        
        # Validate data types and ranges
        if 'Age' in data:
            try:
                age = float(data['Age'])
                if not (18 <= age <= 100):
                    warnings.append(f"Age {age} is outside typical range (18-100)")
            except (ValueError, TypeError):
                errors.append("Age must be a valid number")
        
        if 'TenureMonths' in data:
            try:
                tenure = float(data['TenureMonths'])
                if not (0 <= tenure <= 120):
                    warnings.append(f"Tenure {tenure} months is outside typical range (0-120)")
            except (ValueError, TypeError):
                errors.append("TenureMonths must be a valid number")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

__all__ = ['DataValidator', 'ModelValidator', 'ValidationResult']
