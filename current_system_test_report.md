
# ForeTel.AI Current System Test Report
Generated: 2025-09-19 23:11:32
Duration: 3.00 seconds

## Summary
- **Total Tests**: 45
- **Passed**: 38 ✅
- **Failed**: 7 ❌
- **Success Rate**: 84.4%

## Test Results

### File
- **File: app.py**: ✅ PASS
  - Size: 32612 bytes
- **File: churn_predictor.py**: ✅ PASS
  - Size: 6278 bytes
- **File: revenue_forecasting_model.py**: ✅ PASS
  - Size: 8735 bytes
- **File: dataset_generator.py**: ✅ PASS
  - Size: 4554 bytes
- **File: config.py**: ✅ PASS
  - Size: 4013 bytes
- **File: .env**: ✅ PASS
  - Size: 476 bytes
- **File: Requirements.txt**: ✅ PASS
  - Size: 308 bytes
- **File: telecom_dataset_generated.csv**: ✅ PASS
  - Size: 1930887 bytes

### Directory
- **Directory: models**: ✅ PASS
  - Contains 5 items
- **Directory: logs**: ✅ PASS
  - Contains 0 items
- **Directory: docs**: ✅ PASS
  - Contains 2 items

### Dataset Load
- **Dataset Load: telecom_dataset_generated.csv**: ✅ PASS
  - Shape: 10000 rows × 17 columns

### General
- **Dataset Schema**: ❌ FAIL
  - Missing columns: ['MonthlyIncome', 'Churned', 'TotalRevenue']
- **Data Quality**: ✅ PASS
  - Nulls: 0, Duplicates: 0
- **Data Statistics**: ✅ PASS
  - Age range: 18-90
- **Churn Model**: ✅ PASS
  - Loaded successfully, Size: 261602 bytes
- **Churn Model Type**: ✅ PASS
  - Type: XGBClassifier
- **Revenue Model**: ✅ PASS
  - Loaded successfully, Size: 83253232 bytes
- **Revenue Model Type**: ✅ PASS
  - Type: dict
- **Scaler**: ✅ PASS
  - Loaded successfully, Size: 1983 bytes
- **Scaler Type**: ✅ PASS
  - Type: StandardScaler
- **Imputer**: ✅ PASS
  - Loaded successfully, Size: 1439 bytes
- **Imputer Type**: ✅ PASS
  - Type: SimpleImputer
- **Train Columns**: ✅ PASS
  - Loaded successfully, Size: 949 bytes
- **Train Columns Type**: ✅ PASS
  - Type: Index
- **Churn Predictions**: ❌ FAIL
  - Prediction error: Feature shape mismatch, expected: 27, got 5
- **Revenue Predictions**: ❌ FAIL
  - Prediction error: 'dict' object has no attribute 'predict'
- **App Import**: ✅ PASS
  - app.py can be imported
- **Streamlit Import**: ✅ PASS
  - Streamlit version available
- **Streamlit Server**: ✅ PASS
  - Running at http://localhost:8501
- **Streamlit Health**: ✅ PASS
  - Health endpoint responding
- **Environment File**: ✅ PASS
  - Contains 15 variables
- **Config Module**: ✅ PASS
  - Loaded successfully
- **Model Load Time**: ✅ PASS
  - 4.38ms
- **Prediction Time**: ❌ FAIL
  - Error: Feature shape mismatch, expected: 27, got 5

### Module
- **Module: pandas**: ✅ PASS
  - Available
- **Module: numpy**: ✅ PASS
  - Available
- **Module: plotly**: ✅ PASS
  - Available
- **Module: sklearn**: ✅ PASS
  - Available

### Env Var
- **Env Var: MONGODB_URI**: ✅ PASS
  - Present
- **Env Var: ENVIRONMENT**: ✅ PASS
  - Present
- **Env Var: DEBUG**: ✅ PASS
  - Present

### Config
- **Config: UI_CSS**: ❌ FAIL
  - Missing
- **Config: MONGODB_URI**: ❌ FAIL
  - Missing
- **Config: CHATBOT_RESPONSES**: ❌ FAIL
  - Missing

## System Status
✅ System is mostly functional (84.4% success rate).
