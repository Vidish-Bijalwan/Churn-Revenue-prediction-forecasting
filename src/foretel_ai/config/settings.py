"""
Configuration settings for ForeTel.AI application.

This module provides centralized configuration management for the entire application,
including database settings, model parameters, UI configuration, and feature flags.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"

class LogLevel(Enum):
    """Logging levels enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ModelType(Enum):
    """Model types enumeration."""
    CHURN_PREDICTION = "churn_prediction"
    REVENUE_FORECASTING = "revenue_forecasting"

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///telecom_analytics.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables."""
        return cls(
            url=os.getenv("DATABASE_URL", cls.url),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DB_POOL_SIZE", cls.pool_size)),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", cls.max_overflow))
        )

@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    churn_model_path: str = "enhanced_churn_model.pkl"
    revenue_model_path: str = "revenue_forecasting.pkl"
    scaler_path: str = "scaler.pkl"
    imputer_path: str = "imputer.pkl"
    train_columns_path: str = "train_columns.pkl"
    
    # Model parameters
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Hyperparameter tuning
    optuna_trials: int = 100
    optuna_timeout: Optional[int] = 3600  # 1 hour
    
    # Feature engineering
    churn_features: List[str] = field(default_factory=lambda: [
        "Age", "MonthlyIncome", "MonthlySpending", "NetworkQuality",
        "DataUsageGB", "CallUsageMin", "SMSUsage", "CustomerCareCalls",
        "TenureMonths", "ServiceRating"
    ])
    
    revenue_features: List[str] = field(default_factory=lambda: [
        "MonthlyIncome", "MonthlySpending", "DataUsageGB", "CallUsageMin",
        "SMSUsage", "TenureMonths", "ServiceRating"
    ])

@dataclass
class UIConfig:
    """User interface configuration."""
    page_title: str = "ForeTel.AI - Telecom Analytics Platform"
    page_icon: str = "📊"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Theme colors
    primary_color: str = "#1e3a8a"
    secondary_color: str = "#4CAF50"
    background_color: str = "#ffffff"
    text_color: str = "#333333"
    
    # Chart settings
    chart_height: int = 400
    chart_width: int = 600
    
    # Data display
    max_rows_display: int = 1000
    pagination_size: int = 50

@dataclass
class DataConfig:
    """Data processing configuration."""
    # File paths
    dataset_paths: List[str] = field(default_factory=lambda: [
        "telecom_dataset.csv",
        "telecom_dataset_generated.csv",
        "telecom_dataset_preprocessed.csv"
    ])
    
    # Data generation
    default_customers: int = 10000
    min_customers: int = 1000
    max_customers: int = 100000
    
    # Data validation
    required_columns: List[str] = field(default_factory=lambda: [
        "CustomerID", "Age", "Gender", "Region", "TenureMonths",
        "MonthlySpending", "SubscriptionType", "ContractType"
    ])
    
    # Data cleaning
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    missing_value_strategy: str = "median"  # mean, median, mode, forward_fill
    
    # Feature engineering
    categorical_encoding: str = "onehot"  # onehot, label, target
    scaling_method: str = "standard"  # standard, minmax, robust

@dataclass
class SecurityConfig:
    """Security and validation configuration."""
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".csv", ".xlsx", ".json", ".parquet"
    ])
    max_file_size_mb: int = 100
    max_upload_files: int = 5
    
    # API rate limiting
    api_rate_limit: int = 100  # requests per minute
    api_burst_limit: int = 200
    
    # Session management
    session_timeout: int = 3600  # 1 hour in seconds
    max_concurrent_sessions: int = 100

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    log_to_file: bool = True
    log_file_path: str = "logs/foretel_ai.log"
    max_log_size_mb: int = 10
    backup_count: int = 5
    
    # Console logging
    log_to_console: bool = True
    console_level: LogLevel = LogLevel.INFO

@dataclass
class AppConfig:
    """Main application configuration."""
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    version: str = "2.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    models: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_chat_assistant: bool = True
    enable_document_insights: bool = True
    enable_real_time_predictions: bool = True
    enable_model_explanations: bool = True
    enable_data_export: bool = True
    
    # Performance settings
    cache_timeout: int = 3600  # 1 hour
    max_memory_usage_mb: int = 2048  # 2GB
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories if they don't exist
        for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, ASSETS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        return cls()
    
    def get_model_path(self, model_type: ModelType) -> Path:
        """Get the full path for a model file."""
        model_files = {
            ModelType.CHURN_PREDICTION: self.models.churn_model_path,
            ModelType.REVENUE_FORECASTING: self.models.revenue_model_path
        }
        return MODELS_DIR / model_files[model_type]
    
    def get_data_path(self, filename: str) -> Path:
        """Get the full path for a data file."""
        return DATA_DIR / filename
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

# Global configuration instance
config = AppConfig.load_from_env()

# Export commonly used paths
__all__ = [
    "config",
    "BASE_DIR",
    "DATA_DIR", 
    "MODELS_DIR",
    "LOGS_DIR",
    "ASSETS_DIR",
    "AppConfig",
    "ModelType",
    "LogLevel"
]
