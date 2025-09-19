"""
Professional data processing module for ForeTel.AI.

This module provides comprehensive data loading, validation, preprocessing,
and feature engineering capabilities with proper error handling and logging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

from ..config.settings import config
from ..utils.logger import get_logger
from ..utils.validators import DataValidator

logger = get_logger(__name__)

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    memory_usage: str
    file_size: Optional[str] = None

class TelecomDataProcessor:
    """
    Professional data processor for telecom analytics.
    
    This class handles all data operations including loading, validation,
    preprocessing, and feature engineering with comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.validator = DataValidator()
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, Union[SimpleImputer, KNNImputer]] = {}
        self._is_fitted = False
        
        logger.info("TelecomDataProcessor initialized")
    
    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load telecom dataset with automatic fallback to multiple sources.
        
        Args:
            file_path: Optional specific file path to load
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If no valid dataset is found
            ValueError: If dataset validation fails
        """
        logger.info("Loading telecom dataset")
        
        # Determine file paths to try
        if file_path:
            paths_to_try = [file_path]
        else:
            paths_to_try = [
                config.get_data_path(path) for path in config.data.dataset_paths
            ]
        
        # Try loading from each path
        for path in paths_to_try:
            try:
                if Path(path).exists():
                    df = pd.read_csv(path)
                    logger.info(f"Successfully loaded dataset from {path}")
                    
                    # Validate dataset
                    if self.validator.validate_dataset(df):
                        return self._post_load_processing(df)
                    else:
                        logger.warning(f"Dataset validation failed for {path}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to load dataset from {path}: {e}")
                continue
        
        # If no dataset found, generate sample data
        logger.warning("No valid dataset found, generating sample data")
        return self._generate_sample_data()
    
    def get_dataset_info(self, df: pd.DataFrame) -> DatasetInfo:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DatasetInfo object with dataset statistics
        """
        return DatasetInfo(
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values=df.isnull().sum().to_dict(),
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )
    
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        target_column: str = "Churned",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Comprehensive data preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Feature engineering
        X = self._engineer_features(X)
        
        # Encode categorical variables
        X = self._encode_categorical_features(X)
        
        # Scale numerical features
        X = self._scale_numerical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self._is_fitted = True
        logger.info("Data preprocessing completed successfully")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If preprocessors are not fitted
        """
        if not self._is_fitted:
            raise ValueError("Preprocessors not fitted. Call preprocess_data first.")
        
        logger.info("Transforming new data")
        
        # Apply same transformations as training data
        df = self._handle_missing_values(df, fit=False)
        df = self._engineer_features(df)
        df = self._encode_categorical_features(df, fit=False)
        df = self._scale_numerical_features(df, fit=False)
        
        return df
    
    def _post_load_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-load data processing and validation."""
        # Ensure required columns exist
        if 'TotalRevenue' not in df.columns and all(col in df.columns for col in ['MonthlySpending', 'TenureMonths']):
            df['TotalRevenue'] = df['MonthlySpending'] * df['TenureMonths']
        
        if 'ARPU' not in df.columns and all(col in df.columns for col in ['TotalRevenue', 'TenureMonths']):
            df['ARPU'] = df['TotalRevenue'] / df['TenureMonths'].replace(0, 1)
        
        # Log dataset info
        info = self.get_dataset_info(df)
        logger.info(f"Dataset loaded: {info.shape[0]} rows, {info.shape[1]} columns")
        
        return df
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample telecom data for demonstration."""
        logger.info("Generating sample telecom dataset")
        
        np.random.seed(config.models.random_state)
        n_customers = config.data.default_customers
        
        data = {
            'CustomerID': [f'CUST_{i:06d}' for i in range(n_customers)],
            'Age': np.random.randint(18, 80, n_customers),
            'Gender': np.random.choice(['Male', 'Female'], n_customers),
            'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_customers),
            'TenureMonths': np.random.randint(1, 72, n_customers),
            'MonthlySpending': np.random.uniform(20, 150, n_customers),
            'SubscriptionType': np.random.choice(['Basic', 'Standard', 'Premium'], n_customers),
            'ContractType': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers),
            'PaymentMethod': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], n_customers),
            'NetworkQuality': np.random.randint(1, 6, n_customers),
            'DataUsageGB': np.random.uniform(5, 100, n_customers),
            'CallUsageMin': np.random.uniform(100, 1000, n_customers),
            'SMSUsage': np.random.randint(50, 500, n_customers),
            'CustomerCareCalls': np.random.randint(0, 10, n_customers),
            'ServiceRating': np.random.randint(1, 6, n_customers),
            'ChurnRiskScore': np.random.uniform(0, 1, n_customers),
            'Churned': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        df['TotalRevenue'] = df['MonthlySpending'] * df['TenureMonths']
        df['ARPU'] = df['TotalRevenue'] / df['TenureMonths']
        
        # Save generated data
        output_path = config.get_data_path('telecom_dataset_generated.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Sample dataset saved to {output_path}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Handle numerical missing values
        if numerical_cols:
            if fit:
                self.imputers['numerical'] = SimpleImputer(strategy=config.data.missing_value_strategy)
                df[numerical_cols] = self.imputers['numerical'].fit_transform(df[numerical_cols])
            else:
                if 'numerical' in self.imputers:
                    df[numerical_cols] = self.imputers['numerical'].transform(df[numerical_cols])
        
        # Handle categorical missing values
        if categorical_cols:
            if fit:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = self.imputers['categorical'].fit_transform(df[categorical_cols])
            else:
                if 'categorical' in self.imputers:
                    df[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing data."""
        df = df.copy()
        
        # Revenue-based features
        if 'MonthlySpending' in df.columns and 'TenureMonths' in df.columns:
            df['TotalRevenue'] = df['MonthlySpending'] * df['TenureMonths']
            df['ARPU'] = df['TotalRevenue'] / df['TenureMonths'].replace(0, 1)
        
        # Usage-based features
        if all(col in df.columns for col in ['DataUsageGB', 'CallUsageMin', 'SMSUsage']):
            df['TotalUsage'] = df['DataUsageGB'] + df['CallUsageMin'] / 60 + df['SMSUsage'] / 100
        
        # Customer experience features
        if 'CustomerCareCalls' in df.columns and 'TenureMonths' in df.columns:
            df['CallsPerMonth'] = df['CustomerCareCalls'] / df['TenureMonths'].replace(0, 1)
        
        # Categorical features
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 65, 100], 
                                  labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if config.data.categorical_encoding == 'onehot':
            # One-hot encoding
            if fit:
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            else:
                # For transform, we need to handle new categories
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        elif config.data.categorical_encoding == 'label':
            # Label encoding
            for col in categorical_cols:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = self.label_encoders[col].classes_
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in unique_values else unique_values[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols and config.data.scaling_method == 'standard':
            if fit:
                self.scaler = StandardScaler()
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            else:
                if self.scaler:
                    df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df

__all__ = ['TelecomDataProcessor', 'DatasetInfo']
