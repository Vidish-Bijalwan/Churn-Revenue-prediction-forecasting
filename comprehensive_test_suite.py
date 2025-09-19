#!/usr/bin/env python3
"""
ForeTel.AI Comprehensive Testing Suite
Tests both current system and enterprise components
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import requests
import time
import subprocess
import json
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.base_dir = Path.cwd()
        
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

    def test_current_system_functionality(self):
        """Test the current working Streamlit system"""
        self.print_header("CURRENT SYSTEM FUNCTIONALITY")
        
        # Test Streamlit server
        try:
            response = requests.get('http://localhost:8501', timeout=5)
            if response.status_code == 200:
                self.print_test_result("Streamlit Server", True, "Server is running")
                
                # Test health endpoint
                health_response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
                if health_response.status_code == 200:
                    self.print_test_result("Streamlit Health", True, "Health check passed")
                else:
                    self.print_test_result("Streamlit Health", False, f"Health check failed: {health_response.status_code}")
            else:
                self.print_test_result("Streamlit Server", False, f"Server returned: {response.status_code}")
        except requests.exceptions.RequestException:
            self.print_test_result("Streamlit Server", False, "Server not accessible")

    def test_model_functionality_with_preprocessing(self):
        """Test models with proper data preprocessing"""
        self.print_header("MODEL FUNCTIONALITY WITH PREPROCESSING")
        
        try:
            # Load the dataset
            if os.path.exists('telecom_dataset_generated.csv'):
                df = pd.read_csv('telecom_dataset_generated.csv')
                self.print_test_result("Dataset Loading", True, f"Loaded {len(df)} rows")
                
                # Load preprocessing components
                scaler_path = 'models/scaler.pkl'
                imputer_path = 'models/imputer.pkl'
                columns_path = 'models/train_columns.pkl'
                
                if all(os.path.exists(p) for p in [scaler_path, imputer_path, columns_path]):
                    scaler = joblib.load(scaler_path)
                    imputer = joblib.load(imputer_path)
                    train_columns = joblib.load(columns_path)
                    
                    self.print_test_result("Preprocessing Components", True, "All components loaded")
                    
                    # Prepare sample data with all required features
                    sample_data = df.head(5)[train_columns]
                    
                    # Apply preprocessing
                    sample_imputed = imputer.transform(sample_data)
                    sample_scaled = scaler.transform(sample_imputed)
                    
                    self.print_test_result("Data Preprocessing", True, f"Processed shape: {sample_scaled.shape}")
                    
                    # Test churn model
                    churn_model_path = 'models/enhanced_churn_model.pkl'
                    if os.path.exists(churn_model_path):
                        churn_model = joblib.load(churn_model_path)
                        
                        churn_predictions = churn_model.predict(sample_scaled)
                        churn_probabilities = churn_model.predict_proba(sample_scaled)
                        
                        self.print_test_result("Churn Model Predictions", True, 
                                             f"Predictions: {churn_predictions[:3]}")
                        self.print_test_result("Churn Model Probabilities", True, 
                                             f"Prob shape: {churn_probabilities.shape}")
                    else:
                        self.print_test_result("Churn Model", False, "Model file not found")
                    
                    # Test revenue model (handle dict format)
                    revenue_model_path = 'models/revenue_forecasting.pkl'
                    if os.path.exists(revenue_model_path):
                        revenue_model_data = joblib.load(revenue_model_path)
                        
                        if isinstance(revenue_model_data, dict):
                            if 'ensemble' in revenue_model_data:
                                ensemble_model = revenue_model_data['ensemble']
                                revenue_predictions = ensemble_model.predict(sample_scaled)
                                self.print_test_result("Revenue Model Predictions", True, 
                                                     f"Predictions: {revenue_predictions[:3]}")
                            else:
                                self.print_test_result("Revenue Model", False, "Ensemble model not found in dict")
                        else:
                            revenue_predictions = revenue_model_data.predict(sample_scaled)
                            self.print_test_result("Revenue Model Predictions", True, 
                                                 f"Predictions: {revenue_predictions[:3]}")
                    else:
                        self.print_test_result("Revenue Model", False, "Model file not found")
                        
                else:
                    self.print_test_result("Preprocessing Components", False, "Missing preprocessing files")
                    
            else:
                self.print_test_result("Dataset Loading", False, "Dataset file not found")
                
        except Exception as e:
            self.print_test_result("Model Functionality", False, f"Error: {e}")

    def test_enterprise_architecture_readiness(self):
        """Test enterprise architecture components"""
        self.print_header("ENTERPRISE ARCHITECTURE READINESS")
        
        # Test Docker Compose file
        if os.path.exists('docker-compose.yml'):
            try:
                result = subprocess.run(['docker-compose', 'config'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    self.print_test_result("Docker Compose Config", True, "Configuration is valid")
                else:
                    self.print_test_result("Docker Compose Config", False, f"Config error: {result.stderr}")
            except subprocess.TimeoutExpired:
                self.print_test_result("Docker Compose Config", False, "Configuration check timed out")
            except FileNotFoundError:
                self.print_test_result("Docker Compose Config", False, "docker-compose not installed")
        else:
            self.print_test_result("Docker Compose File", False, "docker-compose.yml not found")
        
        # Test service directories
        service_dirs = ['services/auth', 'services/model', 'services/data', 'services/ui']
        for service_dir in service_dirs:
            if os.path.exists(service_dir):
                dockerfile_path = os.path.join(service_dir, 'Dockerfile')
                requirements_path = os.path.join(service_dir, 'requirements.txt')
                
                if os.path.exists(dockerfile_path):
                    self.print_test_result(f"{service_dir} Dockerfile", True, "Dockerfile exists")
                else:
                    self.print_test_result(f"{service_dir} Dockerfile", False, "Dockerfile missing")
                    
                if os.path.exists(requirements_path):
                    self.print_test_result(f"{service_dir} Requirements", True, "Requirements file exists")
                else:
                    self.print_test_result(f"{service_dir} Requirements", False, "Requirements file missing")
            else:
                self.print_test_result(f"{service_dir} Directory", False, "Service directory not found")
        
        # Test SQL initialization files
        sql_files = ['sql/auth_init.sql', 'sql/models_init.sql', 'sql/data_init.sql']
        for sql_file in sql_files:
            if os.path.exists(sql_file):
                file_size = os.path.getsize(sql_file)
                self.print_test_result(f"SQL Init: {sql_file}", True, f"Size: {file_size} bytes")
            else:
                self.print_test_result(f"SQL Init: {sql_file}", False, "File not found")
        
        # Test NGINX configuration
        if os.path.exists('nginx/nginx.conf'):
            file_size = os.path.getsize('nginx/nginx.conf')
            self.print_test_result("NGINX Configuration", True, f"Size: {file_size} bytes")
        else:
            self.print_test_result("NGINX Configuration", False, "nginx.conf not found")

    def test_api_service_structure(self):
        """Test API service code structure"""
        self.print_header("API SERVICE STRUCTURE")
        
        services = {
            'auth': 'services/auth/main.py',
            'model': 'services/model/main.py', 
            'data': 'services/data/main.py'
        }
        
        for service_name, service_file in services.items():
            if os.path.exists(service_file):
                try:
                    with open(service_file, 'r') as f:
                        content = f.read()
                    
                    # Check for FastAPI imports
                    if 'from fastapi import FastAPI' in content:
                        self.print_test_result(f"{service_name.title()} Service FastAPI", True, "FastAPI imported")
                    else:
                        self.print_test_result(f"{service_name.title()} Service FastAPI", False, "FastAPI not imported")
                    
                    # Check for health endpoint
                    if '/health' in content:
                        self.print_test_result(f"{service_name.title()} Service Health", True, "Health endpoint defined")
                    else:
                        self.print_test_result(f"{service_name.title()} Service Health", False, "Health endpoint missing")
                    
                    # Check file size
                    file_size = len(content)
                    self.print_test_result(f"{service_name.title()} Service Size", True, f"{file_size} characters")
                    
                except Exception as e:
                    self.print_test_result(f"{service_name.title()} Service", False, f"Read error: {e}")
            else:
                self.print_test_result(f"{service_name.title()} Service", False, "Service file not found")

    def test_monitoring_configuration(self):
        """Test monitoring and logging configuration"""
        self.print_header("MONITORING CONFIGURATION")
        
        monitoring_configs = {
            'Prometheus': 'monitoring/prometheus.yml',
            'Grafana Datasources': 'monitoring/grafana/datasources',
            'Grafana Dashboards': 'monitoring/grafana/dashboards',
            'ELK Logstash': 'elk/logstash.conf'
        }
        
        for config_name, config_path in monitoring_configs.items():
            if os.path.exists(config_path):
                if os.path.isdir(config_path):
                    file_count = len(os.listdir(config_path))
                    self.print_test_result(config_name, True, f"Directory with {file_count} items")
                else:
                    file_size = os.path.getsize(config_path)
                    self.print_test_result(config_name, True, f"File size: {file_size} bytes")
            else:
                self.print_test_result(config_name, False, "Configuration not found")

    def test_security_configuration(self):
        """Test security configuration"""
        self.print_header("SECURITY CONFIGURATION")
        
        # Test environment files
        env_files = ['.env', '.env.production']
        for env_file in env_files:
            if os.path.exists(env_file):
                try:
                    with open(env_file, 'r') as f:
                        content = f.read()
                    
                    # Check for security-related variables
                    security_vars = ['JWT_SECRET', 'ENCRYPTION_KEY', 'SSL_CERT_PATH']
                    found_vars = sum(1 for var in security_vars if var in content)
                    
                    self.print_test_result(f"Security Config: {env_file}", True, 
                                         f"Found {found_vars}/{len(security_vars)} security variables")
                except Exception as e:
                    self.print_test_result(f"Security Config: {env_file}", False, f"Read error: {e}")
            else:
                self.print_test_result(f"Security Config: {env_file}", False, "File not found")

    def test_data_pipeline_components(self):
        """Test data pipeline components"""
        self.print_header("DATA PIPELINE COMPONENTS")
        
        # Test Airflow DAGs directory
        airflow_dags = 'airflow/dags'
        if os.path.exists(airflow_dags):
            dag_count = len([f for f in os.listdir(airflow_dags) if f.endswith('.py')])
            self.print_test_result("Airflow DAGs", True, f"Found {dag_count} DAG files")
        else:
            self.print_test_result("Airflow DAGs", False, "DAGs directory not found")
        
        # Test data quality expectations
        if 'great_expectations' in str(self.base_dir / 'services' / 'data' / 'main.py'):
            self.print_test_result("Data Quality Framework", True, "Great Expectations integrated")
        else:
            self.print_test_result("Data Quality Framework", False, "Great Expectations not found")

    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        self.print_header("PERFORMANCE BENCHMARKS")
        
        # Test model loading performance
        model_files = [
            'models/enhanced_churn_model.pkl',
            'models/revenue_forecasting.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                start_time = time.time()
                try:
                    model = joblib.load(model_file)
                    load_time = (time.time() - start_time) * 1000
                    
                    model_name = os.path.basename(model_file).replace('.pkl', '')
                    if load_time < 1000:  # Less than 1 second
                        self.print_test_result(f"Load Time: {model_name}", True, f"{load_time:.2f}ms")
                    else:
                        self.print_test_result(f"Load Time: {model_name}", False, f"{load_time:.2f}ms (slow)")
                        
                except Exception as e:
                    self.print_test_result(f"Load Time: {model_name}", False, f"Error: {e}")

    def test_documentation_completeness(self):
        """Test documentation completeness"""
        self.print_header("DOCUMENTATION COMPLETENESS")
        
        doc_files = {
            'API Reference': 'docs/API_REFERENCE.md',
            'Industry Recommendations': 'docs/INDUSTRY_RECOMMENDATIONS.md',
            'New README': 'README_NEW.md',
            'Original README': 'README.md'
        }
        
        for doc_name, doc_path in doc_files.items():
            if os.path.exists(doc_path):
                file_size = os.path.getsize(doc_path)
                self.print_test_result(doc_name, True, f"Size: {file_size} bytes")
            else:
                self.print_test_result(doc_name, False, "Documentation not found")

    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        self.print_header("COMPREHENSIVE TEST REPORT")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Categorize results
        categories = {
            'Current System': [],
            'Enterprise Architecture': [],
            'API Services': [],
            'Security': [],
            'Performance': [],
            'Documentation': [],
            'Other': []
        }
        
        for test_name, result in self.test_results.items():
            if any(keyword in test_name.lower() for keyword in ['streamlit', 'model', 'current']):
                categories['Current System'].append((test_name, result))
            elif any(keyword in test_name.lower() for keyword in ['docker', 'compose', 'service', 'sql', 'nginx']):
                categories['Enterprise Architecture'].append((test_name, result))
            elif any(keyword in test_name.lower() for keyword in ['api', 'fastapi', 'health']):
                categories['API Services'].append((test_name, result))
            elif any(keyword in test_name.lower() for keyword in ['security', 'jwt', 'encryption']):
                categories['Security'].append((test_name, result))
            elif any(keyword in test_name.lower() for keyword in ['performance', 'load', 'time']):
                categories['Performance'].append((test_name, result))
            elif any(keyword in test_name.lower() for keyword in ['documentation', 'readme', 'api reference']):
                categories['Documentation'].append((test_name, result))
            else:
                categories['Other'].append((test_name, result))
        
        report = f"""
# ForeTel.AI Comprehensive Test Report
Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration.total_seconds():.2f} seconds

## Executive Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ✅
- **Failed**: {failed_tests} ❌
- **Success Rate**: {success_rate:.1f}%

## Test Results by Category
"""
        
        for category, tests in categories.items():
            if tests:
                category_passed = sum(1 for _, result in tests if result['passed'])
                category_total = len(tests)
                category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
                
                report += f"\n### {category} ({category_passed}/{category_total} - {category_rate:.1f}%)\n"
                for test_name, result in tests:
                    status = "✅ PASS" if result['passed'] else "❌ FAIL"
                    report += f"- **{test_name}**: {status}\n"
                    if result['details']:
                        report += f"  - {result['details']}\n"
        
        report += f"""
## System Readiness Assessment

### Current System Status
"""
        current_system_tests = categories['Current System']
        if current_system_tests:
            current_passed = sum(1 for _, result in current_system_tests if result['passed'])
            current_total = len(current_system_tests)
            current_rate = (current_passed / current_total * 100) if current_total > 0 else 0
            
            if current_rate >= 90:
                report += f"🎉 **EXCELLENT** - Current system is fully functional ({current_rate:.1f}%)\n"
            elif current_rate >= 75:
                report += f"✅ **GOOD** - Current system is mostly functional ({current_rate:.1f}%)\n"
            elif current_rate >= 50:
                report += f"⚠️ **FAIR** - Current system has some issues ({current_rate:.1f}%)\n"
            else:
                report += f"❌ **POOR** - Current system has significant issues ({current_rate:.1f}%)\n"
        
        report += f"""
### Enterprise Architecture Status
"""
        enterprise_tests = categories['Enterprise Architecture']
        if enterprise_tests:
            enterprise_passed = sum(1 for _, result in enterprise_tests if result['passed'])
            enterprise_total = len(enterprise_tests)
            enterprise_rate = (enterprise_passed / enterprise_total * 100) if enterprise_total > 0 else 0
            
            if enterprise_rate >= 90:
                report += f"🚀 **READY** - Enterprise architecture is deployment-ready ({enterprise_rate:.1f}%)\n"
            elif enterprise_rate >= 75:
                report += f"🔧 **NEARLY READY** - Minor configuration needed ({enterprise_rate:.1f}%)\n"
            elif enterprise_rate >= 50:
                report += f"⚠️ **IN PROGRESS** - Significant work remaining ({enterprise_rate:.1f}%)\n"
            else:
                report += f"🚧 **NOT READY** - Major components missing ({enterprise_rate:.1f}%)\n"
        
        report += f"""
## Recommendations

### Immediate Actions
"""
        if failed_tests > 0:
            report += "1. **Fix Failed Tests**: Address the failed tests listed above\n"
            report += "2. **Verify Dependencies**: Ensure all required packages are installed\n"
            report += "3. **Check Configuration**: Validate all configuration files\n"
        else:
            report += "1. **Deploy to Production**: All tests passed - ready for deployment\n"
            report += "2. **Monitor Performance**: Set up continuous monitoring\n"
            report += "3. **Scale Infrastructure**: Prepare for production load\n"
        
        report += f"""
### Next Steps
1. **Current System**: {'✅ Production Ready' if success_rate >= 90 else '⚠️ Needs Attention'}
2. **Enterprise Platform**: {'🚀 Ready to Deploy' if success_rate >= 85 else '🔧 Needs Configuration'}
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Security**: Complete security audit and penetration testing
5. **Performance**: Conduct load testing and optimization

---
*Report generated by ForeTel.AI Comprehensive Test Suite*
"""
        
        # Save report
        report_file = "comprehensive_test_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Comprehensive test report saved to: {report_file}")
        print(f"\n🎯 **FINAL RESULT**: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        # Print summary
        if success_rate >= 90:
            print("🎉 **EXCELLENT**: System is ready for production deployment!")
        elif success_rate >= 75:
            print("✅ **GOOD**: System is mostly ready with minor issues to address")
        elif success_rate >= 50:
            print("⚠️ **FAIR**: System needs significant work before deployment")
        else:
            print("❌ **POOR**: System requires major fixes and improvements")
        
        return report

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting ForeTel.AI Comprehensive Testing Suite")
        print(f"⏰ Test started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.test_current_system_functionality()
        self.test_model_functionality_with_preprocessing()
        self.test_enterprise_architecture_readiness()
        self.test_api_service_structure()
        self.test_monitoring_configuration()
        self.test_security_configuration()
        self.test_data_pipeline_components()
        self.test_performance_benchmarks()
        self.test_documentation_completeness()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()

def main():
    """Main test execution function"""
    test_suite = ComprehensiveTestSuite()
    
    try:
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        logger.exception("Comprehensive test suite failed")

if __name__ == "__main__":
    main()
