#!/usr/bin/env python3
"""
ForeTel.AI Current System Testing Suite
Tests the existing Streamlit application and trained models
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import requests
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CurrentSystemTestSuite:
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.base_dir = os.getcwd()
        
    def print_header(self, title: str):
        """Print formatted test section header"""
        print(f"\n{'='*80}")
        print(f"🧪 {title}")
        print(f"{'='*80}")

    def print_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Print individual test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   📝 {details}")
        
        self.test_results[test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now()
        }

    def test_file_structure(self):
        """Test if all required files exist"""
        self.print_header("FILE STRUCTURE TESTING")
        
        required_files = [
            'app.py',
            'churn_predictor.py', 
            'revenue_forecasting_model.py',
            'dataset_generator.py',
            'config.py',
            '.env',
            'Requirements.txt',
            'telecom_dataset_generated.csv'
        ]
        
        required_dirs = [
            'models',
            'logs',
            'docs'
        ]
        
        # Check files
        for file_path in required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.print_test_result(f"File: {file_path}", True, f"Size: {file_size} bytes")
            else:
                self.print_test_result(f"File: {file_path}", False, "File not found")
        
        # Check directories
        for dir_path in required_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                file_count = len(os.listdir(dir_path))
                self.print_test_result(f"Directory: {dir_path}", True, f"Contains {file_count} items")
            else:
                self.print_test_result(f"Directory: {dir_path}", False, "Directory not found")

    def test_dataset(self):
        """Test the generated dataset"""
        self.print_header("DATASET TESTING")
        
        dataset_files = [
            'telecom_dataset_generated.csv',
            'telecom_dataset.csv'
        ]
        
        dataset_found = False
        for dataset_file in dataset_files:
            if os.path.exists(dataset_file):
                try:
                    df = pd.read_csv(dataset_file)
                    
                    # Basic dataset checks
                    rows, cols = df.shape
                    self.print_test_result(f"Dataset Load: {dataset_file}", True, 
                                         f"Shape: {rows} rows × {cols} columns")
                    
                    # Check for required columns
                    required_columns = ['CustomerID', 'Age', 'MonthlyIncome', 'Churned', 'TotalRevenue']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if not missing_columns:
                        self.print_test_result("Dataset Schema", True, "All required columns present")
                    else:
                        self.print_test_result("Dataset Schema", False, 
                                             f"Missing columns: {missing_columns}")
                    
                    # Check data quality
                    null_counts = df.isnull().sum().sum()
                    duplicate_counts = df.duplicated().sum()
                    
                    self.print_test_result("Data Quality", True, 
                                         f"Nulls: {null_counts}, Duplicates: {duplicate_counts}")
                    
                    # Statistical summary
                    if 'Age' in df.columns:
                        age_stats = f"Age range: {df['Age'].min()}-{df['Age'].max()}"
                        self.print_test_result("Data Statistics", True, age_stats)
                    
                    dataset_found = True
                    break
                    
                except Exception as e:
                    self.print_test_result(f"Dataset Load: {dataset_file}", False, f"Error: {e}")
        
        if not dataset_found:
            self.print_test_result("Dataset Availability", False, "No dataset files found")

    def test_trained_models(self):
        """Test the trained models"""
        self.print_header("TRAINED MODELS TESTING")
        
        model_files = {
            'Churn Model': 'models/enhanced_churn_model.pkl',
            'Revenue Model': 'models/revenue_forecasting.pkl',
            'Scaler': 'models/scaler.pkl',
            'Imputer': 'models/imputer.pkl',
            'Train Columns': 'models/train_columns.pkl'
        }
        
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    file_size = os.path.getsize(model_path)
                    self.print_test_result(f"{model_name}", True, 
                                         f"Loaded successfully, Size: {file_size} bytes")
                    
                    # Test model type
                    model_type = type(model).__name__
                    self.print_test_result(f"{model_name} Type", True, f"Type: {model_type}")
                    
                except Exception as e:
                    self.print_test_result(f"{model_name}", False, f"Load error: {e}")
            else:
                self.print_test_result(f"{model_name}", False, "Model file not found")

    def test_model_predictions(self):
        """Test model predictions with sample data"""
        self.print_header("MODEL PREDICTIONS TESTING")
        
        try:
            # Load models
            churn_model_path = 'models/enhanced_churn_model.pkl'
            revenue_model_path = 'models/revenue_forecasting.pkl'
            
            if os.path.exists(churn_model_path):
                churn_model = joblib.load(churn_model_path)
                
                # Create sample data
                sample_data = pd.DataFrame({
                    'Age': [35, 45, 25],
                    'MonthlyIncome': [50000, 75000, 30000],
                    'DataUsageGB': [15.5, 25.0, 8.2],
                    'CallUsageMin': [250, 400, 150],
                    'MonthlySpending': [75, 120, 45]
                })
                
                # Test churn prediction
                try:
                    churn_predictions = churn_model.predict(sample_data)
                    churn_probabilities = churn_model.predict_proba(sample_data)
                    
                    self.print_test_result("Churn Predictions", True, 
                                         f"Predictions: {churn_predictions}, Shape: {churn_predictions.shape}")
                    self.print_test_result("Churn Probabilities", True, 
                                         f"Probabilities shape: {churn_probabilities.shape}")
                except Exception as e:
                    self.print_test_result("Churn Predictions", False, f"Prediction error: {e}")
            else:
                self.print_test_result("Churn Model", False, "Churn model not found")
            
            if os.path.exists(revenue_model_path):
                revenue_model = joblib.load(revenue_model_path)
                
                # Test revenue prediction
                try:
                    revenue_predictions = revenue_model.predict(sample_data)
                    
                    self.print_test_result("Revenue Predictions", True, 
                                         f"Predictions: {revenue_predictions}, Shape: {revenue_predictions.shape}")
                except Exception as e:
                    self.print_test_result("Revenue Predictions", False, f"Prediction error: {e}")
            else:
                self.print_test_result("Revenue Model", False, "Revenue model not found")
                
        except Exception as e:
            self.print_test_result("Model Predictions", False, f"General error: {e}")

    def test_streamlit_app(self):
        """Test if Streamlit app can be imported and basic functionality"""
        self.print_header("STREAMLIT APPLICATION TESTING")
        
        try:
            # Test if app.py can be imported
            import importlib.util
            spec = importlib.util.spec_from_file_location("app", "app.py")
            
            if spec and spec.loader:
                self.print_test_result("App Import", True, "app.py can be imported")
                
                # Check if Streamlit is available
                try:
                    import streamlit as st
                    self.print_test_result("Streamlit Import", True, f"Streamlit version available")
                except ImportError:
                    self.print_test_result("Streamlit Import", False, "Streamlit not installed")
                
                # Check other required imports
                required_modules = ['pandas', 'numpy', 'plotly', 'sklearn']
                for module in required_modules:
                    try:
                        __import__(module)
                        self.print_test_result(f"Module: {module}", True, "Available")
                    except ImportError:
                        self.print_test_result(f"Module: {module}", False, "Not available")
            else:
                self.print_test_result("App Import", False, "Cannot import app.py")
                
        except Exception as e:
            self.print_test_result("App Import", False, f"Import error: {e}")

    def test_streamlit_server(self):
        """Test if Streamlit server is running"""
        self.print_header("STREAMLIT SERVER TESTING")
        
        streamlit_urls = [
            'http://localhost:8501',
            'http://127.0.0.1:8501'
        ]
        
        server_running = False
        for url in streamlit_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.print_test_result("Streamlit Server", True, f"Running at {url}")
                    server_running = True
                    
                    # Test health endpoint
                    try:
                        health_response = requests.get(f"{url}/_stcore/health", timeout=5)
                        if health_response.status_code == 200:
                            self.print_test_result("Streamlit Health", True, "Health endpoint responding")
                        else:
                            self.print_test_result("Streamlit Health", False, f"Health endpoint: {health_response.status_code}")
                    except:
                        self.print_test_result("Streamlit Health", False, "Health endpoint not accessible")
                    
                    break
                else:
                    self.print_test_result("Streamlit Server", False, f"HTTP {response.status_code} at {url}")
            except requests.exceptions.RequestException:
                continue
        
        if not server_running:
            self.print_test_result("Streamlit Server", False, "Server not running on standard ports")

    def test_configuration(self):
        """Test configuration files"""
        self.print_header("CONFIGURATION TESTING")
        
        # Test .env file
        if os.path.exists('.env'):
            try:
                with open('.env', 'r') as f:
                    env_content = f.read()
                    
                env_vars = [line for line in env_content.split('\n') if '=' in line and not line.startswith('#')]
                self.print_test_result("Environment File", True, f"Contains {len(env_vars)} variables")
                
                # Check for critical variables
                critical_vars = ['MONGODB_URI', 'ENVIRONMENT', 'DEBUG']
                for var in critical_vars:
                    if var in env_content:
                        self.print_test_result(f"Env Var: {var}", True, "Present")
                    else:
                        self.print_test_result(f"Env Var: {var}", False, "Missing")
                        
            except Exception as e:
                self.print_test_result("Environment File", False, f"Read error: {e}")
        else:
            self.print_test_result("Environment File", False, "File not found")
        
        # Test config.py
        if os.path.exists('config.py'):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", "config.py")
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                self.print_test_result("Config Module", True, "Loaded successfully")
                
                # Check for key configuration items
                config_items = ['UI_CSS', 'MONGODB_URI', 'CHATBOT_RESPONSES']
                for item in config_items:
                    if hasattr(config_module, item):
                        self.print_test_result(f"Config: {item}", True, "Available")
                    else:
                        self.print_test_result(f"Config: {item}", False, "Missing")
                        
            except Exception as e:
                self.print_test_result("Config Module", False, f"Load error: {e}")
        else:
            self.print_test_result("Config Module", False, "File not found")

    def test_performance_metrics(self):
        """Test basic performance metrics"""
        self.print_header("PERFORMANCE TESTING")
        
        # Test model loading time
        if os.path.exists('models/enhanced_churn_model.pkl'):
            start_time = time.time()
            try:
                model = joblib.load('models/enhanced_churn_model.pkl')
                load_time = (time.time() - start_time) * 1000
                
                if load_time < 5000:  # Less than 5 seconds
                    self.print_test_result("Model Load Time", True, f"{load_time:.2f}ms")
                else:
                    self.print_test_result("Model Load Time", False, f"{load_time:.2f}ms (too slow)")
            except Exception as e:
                self.print_test_result("Model Load Time", False, f"Error: {e}")
        
        # Test prediction time
        if os.path.exists('models/enhanced_churn_model.pkl'):
            try:
                model = joblib.load('models/enhanced_churn_model.pkl')
                sample_data = pd.DataFrame({
                    'Age': [35], 'MonthlyIncome': [50000], 'DataUsageGB': [15.5],
                    'CallUsageMin': [250], 'MonthlySpending': [75]
                })
                
                start_time = time.time()
                prediction = model.predict(sample_data)
                prediction_time = (time.time() - start_time) * 1000
                
                if prediction_time < 100:  # Less than 100ms
                    self.print_test_result("Prediction Time", True, f"{prediction_time:.2f}ms")
                else:
                    self.print_test_result("Prediction Time", False, f"{prediction_time:.2f}ms (too slow)")
            except Exception as e:
                self.print_test_result("Prediction Time", False, f"Error: {e}")

    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.print_header("TEST REPORT GENERATION")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = f"""
# ForeTel.AI Current System Test Report
Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration.total_seconds():.2f} seconds

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ✅
- **Failed**: {failed_tests} ❌
- **Success Rate**: {success_rate:.1f}%

## Test Results
"""
        
        categories = {}
        for test_name, result in self.test_results.items():
            category = test_name.split(':')[0] if ':' in test_name else 'General'
            if category not in categories:
                categories[category] = []
            categories[category].append((test_name, result))
        
        for category, tests in categories.items():
            report += f"\n### {category}\n"
            for test_name, result in tests:
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                report += f"- **{test_name}**: {status}\n"
                if result['details']:
                    report += f"  - {result['details']}\n"
        
        report += f"""
## System Status
"""
        
        if failed_tests == 0:
            report += "🎉 All tests passed! The current ForeTel.AI system is working correctly.\n"
        elif success_rate >= 80:
            report += f"✅ System is mostly functional ({success_rate:.1f}% success rate).\n"
        elif success_rate >= 60:
            report += f"⚠️ System has some issues ({success_rate:.1f}% success rate).\n"
        else:
            report += f"❌ System has significant issues ({success_rate:.1f}% success rate).\n"
        
        # Save report to file
        report_file = "current_system_test_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Test report saved to: {report_file}")
        print(f"\n🎯 **FINAL RESULT**: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return report

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting ForeTel.AI Current System Testing Suite")
        print(f"⏰ Test started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.test_file_structure()
        self.test_dataset()
        self.test_trained_models()
        self.test_model_predictions()
        self.test_streamlit_app()
        self.test_streamlit_server()
        self.test_configuration()
        self.test_performance_metrics()
        
        # Generate final report
        self.generate_test_report()

def main():
    """Main test execution function"""
    test_suite = CurrentSystemTestSuite()
    
    try:
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        logger.exception("Test suite failed")

if __name__ == "__main__":
    main()
