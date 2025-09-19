# ForeTel.AI API Reference

## Overview

ForeTel.AI provides a comprehensive API for telecom analytics, churn prediction, and revenue forecasting. This document outlines the main classes, methods, and usage patterns.

## Core Components

### Configuration Management

#### `AppConfig`
Main configuration class for the entire application.

```python
from foretel_ai.config.settings import config

# Access configuration
database_url = config.database.url
model_path = config.get_model_path(ModelType.CHURN_PREDICTION)
```

**Key Properties:**
- `database`: Database configuration
- `models`: ML model settings
- `ui`: User interface configuration
- `data`: Data processing settings
- `security`: Security and validation settings
- `logging`: Logging configuration

### Data Processing

#### `TelecomDataProcessor`
Handles all data operations including loading, validation, and preprocessing.

```python
from foretel_ai.data.processor import TelecomDataProcessor

processor = TelecomDataProcessor()

# Load dataset
df = processor.load_dataset()

# Preprocess for training
X_train, X_test, y_train, y_test = processor.preprocess_data(df, "Churned")

# Transform new data
new_data_processed = processor.transform_new_data(new_df)
```

**Key Methods:**
- `load_dataset(file_path=None)`: Load telecom dataset with fallback options
- `preprocess_data(df, target_column, test_size, random_state)`: Full preprocessing pipeline
- `transform_new_data(df)`: Transform new data using fitted preprocessors
- `get_dataset_info(df)`: Get comprehensive dataset statistics

### Model Management

#### `ChurnPredictor`
Advanced churn prediction using XGBoost with SHAP explanations.

```python
from foretel_ai.models.churn_model import ChurnPredictor

predictor = ChurnPredictor()

# Train model
metrics = predictor.train(df, optimize_hyperparameters=True)

# Make predictions
prediction = predictor.predict({
    'Age': 35,
    'TenureMonths': 24,
    'MonthlySpending': 75.0,
    'NetworkQuality': 4
})

# Get feature importance
importance = predictor.get_feature_importance(plot=True)
```

**Key Methods:**
- `train(df, target_column, optimize_hyperparameters, save_model)`: Train the model
- `predict(data, explain=True)`: Make predictions with explanations
- `get_feature_importance(plot=False)`: Get feature importance scores
- `save_model(file_path=None)`: Save trained model
- `load_model(file_path=None)`: Load pre-trained model

#### `RevenueForecaster`
Ensemble revenue forecasting with multiple algorithms.

```python
from foretel_ai.models.revenue_model import RevenueForecaster

forecaster = RevenueForecaster()

# Train ensemble
metrics = forecaster.train(df, optimize_hyperparameters=True)

# Make forecast
forecast = forecaster.predict({
    'MonthlyIncome': 3500,
    'MonthlySpending': 75.0,
    'TenureMonths': 24
}, forecast_period="monthly")

# Get model importance
importance = forecaster.get_model_importance()
```

**Key Methods:**
- `train(df, target_column, optimize_hyperparameters, include_time_series, save_model)`: Train ensemble
- `predict(data, include_confidence, forecast_period)`: Generate forecasts
- `get_model_importance()`: Get feature importance from all models
- `save_model(file_path=None)`: Save trained ensemble
- `load_model(file_path=None)`: Load pre-trained ensemble

### Data Validation

#### `DataValidator`
Comprehensive data validation and quality assessment.

```python
from foretel_ai.utils.validators import DataValidator

validator = DataValidator()

# Validate dataset
is_valid = validator.validate_dataset(df)

# Comprehensive validation
result = validator.comprehensive_validation(df)
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
print(f"Metrics: {result.metrics}")

# Validate file upload
file_result = validator.validate_file_upload("data.csv")
```

**Key Methods:**
- `validate_dataset(df)`: Quick dataset validation
- `comprehensive_validation(df)`: Detailed validation with metrics
- `validate_file_upload(file_path)`: Validate uploaded files

### UI Components

#### `UIComponents`
Professional UI components for Streamlit applications.

```python
from foretel_ai.ui.components import UIComponents

ui = UIComponents()

# Create page header
ui.page_header("Dashboard", "Analytics Overview", "📊")

# Create metric cards
ui.metric_card("Total Customers", "10,000", "↗️ 5%", icon="👥")

# Create data table
ui.data_table(df, "Customer Data", searchable=True, downloadable=True)

# Create charts
fig = ui.create_gauge_chart(85, "Customer Satisfaction", 0, 100)
```

**Key Methods:**
- `page_header(title, subtitle, icon)`: Professional page headers
- `metric_card(title, value, delta, delta_color, icon)`: KPI metric cards
- `data_table(df, title, searchable, downloadable, max_rows)`: Interactive tables
- `create_gauge_chart(value, title, min_val, max_val)`: Gauge visualizations
- `create_feature_importance_chart(importance_dict, title)`: Feature importance plots

### Logging System

#### `ForeTelLogger`
Professional logging with file rotation and colored console output.

```python
from foretel_ai.utils.logger import get_logger, log_performance, log_model_metrics

logger = get_logger(__name__)

# Basic logging
logger.info("Process started")
logger.error("An error occurred")

# Performance logging
log_performance("data_processing", 2.5, rows_processed=10000)

# Model metrics logging
log_model_metrics("ChurnPredictor", {"accuracy": 0.94, "precision": 0.91})
```

**Key Functions:**
- `get_logger(name)`: Get logger instance
- `log_performance(operation, duration, **metadata)`: Log performance metrics
- `log_model_metrics(model_name, metrics)`: Log model performance
- `log_user_action(action, user_id, **context)`: Log user interactions

## Data Structures

### Prediction Results

#### `ChurnPredictionResult`
```python
@dataclass
class ChurnPredictionResult:
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str  # "Very Low", "Low", "Medium", "High", "Very High"
    confidence: float
    feature_importance: Dict[str, float]
```

#### `RevenueForecast`
```python
@dataclass
class RevenueForecast:
    customer_id: str
    predicted_revenue: float
    confidence_interval: Tuple[float, float]
    forecast_period: str
    model_confidence: float
    contributing_factors: Dict[str, float]
```

### Model Metrics

#### `ModelMetrics`
```python
@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
```

#### `EnsembleMetrics`
```python
@dataclass
class EnsembleMetrics:
    mse: float
    rmse: float
    mae: float
    r2_score: float
    mape: float
    individual_scores: Dict[str, float]
```

## Usage Examples

### Complete Churn Prediction Pipeline

```python
from foretel_ai.data.processor import TelecomDataProcessor
from foretel_ai.models.churn_model import ChurnPredictor
from foretel_ai.utils.validators import DataValidator

# Initialize components
processor = TelecomDataProcessor()
predictor = ChurnPredictor()
validator = DataValidator()

# Load and validate data
df = processor.load_dataset()
if validator.validate_dataset(df):
    # Train model
    metrics = predictor.train(df, optimize_hyperparameters=True)
    print(f"Model accuracy: {metrics.accuracy:.3f}")
    
    # Make prediction
    prediction = predictor.predict({
        'Age': 35,
        'TenureMonths': 24,
        'MonthlySpending': 75.0,
        'NetworkQuality': 4,
        'DataUsageGB': 25.0,
        'CallUsageMin': 500,
        'SMSUsage': 150,
        'CustomerCareCalls': 2,
        'ServiceRating': 4
    })
    
    print(f"Churn probability: {prediction.churn_probability:.1%}")
    print(f"Risk level: {prediction.risk_level}")
```

### Revenue Forecasting with Confidence Intervals

```python
from foretel_ai.models.revenue_model import RevenueForecaster

forecaster = RevenueForecaster()

# Train ensemble model
metrics = forecaster.train(df, include_time_series=True)
print(f"Model R²: {metrics.r2_score:.3f}")

# Generate forecast
forecast = forecaster.predict({
    'MonthlyIncome': 3500,
    'MonthlySpending': 75.0,
    'TenureMonths': 24,
    'DataUsageGB': 25.0,
    'ServiceRating': 4
}, include_confidence=True)

print(f"Predicted revenue: ${forecast.predicted_revenue:.2f}")
print(f"Confidence interval: ${forecast.confidence_interval[0]:.2f} - ${forecast.confidence_interval[1]:.2f}")
```

### Batch Processing

```python
import pandas as pd

# Load batch data
batch_df = pd.read_csv("customer_batch.csv")

# Validate batch data
if validator.validate_dataset(batch_df):
    # Batch churn predictions
    churn_predictions = predictor.predict(batch_df)
    
    # Batch revenue forecasts
    revenue_forecasts = forecaster.predict(batch_df)
    
    # Create results DataFrame
    results = pd.DataFrame([
        {
            'CustomerID': pred.customer_id,
            'ChurnProbability': pred.churn_probability,
            'RiskLevel': pred.risk_level,
            'PredictedRevenue': forecast.predicted_revenue,
            'RevenueConfidence': forecast.model_confidence
        }
        for pred, forecast in zip(churn_predictions, revenue_forecasts)
    ])
    
    # Save results
    results.to_csv("prediction_results.csv", index=False)
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///telecom_analytics.db

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true

# Model Settings
RANDOM_STATE=42
TEST_SIZE=0.2

# UI Settings
DEBUG=false
```

### Custom Configuration

```python
from foretel_ai.config.settings import AppConfig, ModelConfig

# Create custom configuration
custom_config = AppConfig(
    models=ModelConfig(
        random_state=123,
        test_size=0.3,
        optuna_trials=200
    ),
    debug=True
)
```

## Error Handling

All components include comprehensive error handling:

```python
try:
    prediction = predictor.predict(data)
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
except Exception as e:
    logger.error(f"Prediction failed: {e}")
```

## Performance Optimization

- **Caching**: Models and data are cached for faster subsequent operations
- **Parallel Processing**: Multi-core processing for batch operations
- **Memory Management**: Efficient memory usage with data streaming
- **Model Optimization**: Hyperparameter tuning for optimal performance

## Security Considerations

- **Input Validation**: All inputs are validated before processing
- **File Upload Limits**: Size and type restrictions on uploaded files
- **Data Sanitization**: Automatic cleaning of potentially harmful data
- **Logging**: Comprehensive audit trails for all operations
