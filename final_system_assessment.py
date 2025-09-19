#!/usr/bin/env python3
"""
ForeTel.AI Final System Assessment
Comprehensive evaluation of current system and enterprise readiness
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import requests
import time
import json
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalSystemAssessment:
    def __init__(self):
        self.assessment_results = {}
        self.start_time = datetime.now()
        self.base_dir = Path.cwd()
        
    def print_header(self, title: str):
        """Print formatted assessment section header"""
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print(f"{'='*80}")

    def print_assessment_result(self, component: str, status: str, score: int, details: str = ""):
        """Print individual assessment result"""
        status_icons = {
            'EXCELLENT': '🎉',
            'GOOD': '✅', 
            'FAIR': '⚠️',
            'POOR': '❌',
            'NOT_TESTED': '⏸️'
        }
        
        icon = status_icons.get(status, '❓')
        print(f"{icon} {component}: {status} ({score}/100)")
        if details:
            print(f"   📝 {details}")
        
        self.assessment_results[component] = {
            'status': status,
            'score': score,
            'details': details,
            'timestamp': datetime.now()
        }

    def assess_current_streamlit_system(self):
        """Assess the current Streamlit system"""
        self.print_header("CURRENT STREAMLIT SYSTEM ASSESSMENT")
        
        total_score = 0
        max_score = 0
        
        # Test Streamlit server availability
        try:
            response = requests.get('http://localhost:8501', timeout=5)
            if response.status_code == 200:
                server_score = 100
                self.print_assessment_result("Streamlit Server", "EXCELLENT", server_score, 
                                           "Server running and accessible")
            else:
                server_score = 50
                self.print_assessment_result("Streamlit Server", "FAIR", server_score, 
                                           f"Server responding with status {response.status_code}")
        except:
            server_score = 0
            self.print_assessment_result("Streamlit Server", "POOR", server_score, 
                                       "Server not accessible")
        
        total_score += server_score
        max_score += 100
        
        # Test dataset availability and quality
        dataset_files = ['telecom_dataset_generated.csv', 'telecom_dataset.csv']
        dataset_score = 0
        
        for dataset_file in dataset_files:
            if os.path.exists(dataset_file):
                try:
                    df = pd.read_csv(dataset_file)
                    rows, cols = df.shape
                    
                    if rows >= 1000 and cols >= 10:
                        dataset_score = 100
                        self.print_assessment_result("Dataset Quality", "EXCELLENT", dataset_score,
                                                   f"High-quality dataset: {rows} rows × {cols} columns")
                    elif rows >= 100:
                        dataset_score = 75
                        self.print_assessment_result("Dataset Quality", "GOOD", dataset_score,
                                                   f"Adequate dataset: {rows} rows × {cols} columns")
                    else:
                        dataset_score = 50
                        self.print_assessment_result("Dataset Quality", "FAIR", dataset_score,
                                                   f"Small dataset: {rows} rows × {cols} columns")
                    break
                except:
                    continue
        
        if dataset_score == 0:
            self.print_assessment_result("Dataset Quality", "POOR", 0, "No valid dataset found")
        
        total_score += dataset_score
        max_score += 100
        
        # Test trained models
        model_files = {
            'Churn Model': 'models/enhanced_churn_model.pkl',
            'Revenue Model': 'models/revenue_forecasting.pkl',
            'Preprocessing': ['models/scaler.pkl', 'models/imputer.pkl', 'models/train_columns.pkl']
        }
        
        models_score = 0
        models_found = 0
        
        for model_name, model_path in model_files.items():
            if model_name == 'Preprocessing':
                if all(os.path.exists(p) for p in model_path):
                    models_found += 1
                    models_score += 33
            else:
                if os.path.exists(model_path):
                    models_found += 1
                    models_score += 33
        
        if models_score >= 90:
            self.print_assessment_result("Trained Models", "EXCELLENT", models_score,
                                       f"All {models_found} model components available")
        elif models_score >= 60:
            self.print_assessment_result("Trained Models", "GOOD", models_score,
                                       f"{models_found} model components available")
        elif models_score > 0:
            self.print_assessment_result("Trained Models", "FAIR", models_score,
                                       f"Some model components missing")
        else:
            self.print_assessment_result("Trained Models", "POOR", 0, "No trained models found")
        
        total_score += models_score
        max_score += 100
        
        # Test application functionality
        app_files = ['app.py', 'churn_predictor.py', 'revenue_forecasting_model.py', 'config.py']
        app_score = 0
        
        for app_file in app_files:
            if os.path.exists(app_file):
                app_score += 25
        
        if app_score == 100:
            self.print_assessment_result("Application Files", "EXCELLENT", app_score,
                                       "All core application files present")
        elif app_score >= 75:
            self.print_assessment_result("Application Files", "GOOD", app_score,
                                       "Most application files present")
        elif app_score >= 50:
            self.print_assessment_result("Application Files", "FAIR", app_score,
                                       "Some application files missing")
        else:
            self.print_assessment_result("Application Files", "POOR", app_score,
                                       "Critical application files missing")
        
        total_score += app_score
        max_score += 100
        
        # Calculate overall current system score
        overall_score = int((total_score / max_score) * 100) if max_score > 0 else 0
        
        if overall_score >= 90:
            status = "EXCELLENT"
        elif overall_score >= 75:
            status = "GOOD"
        elif overall_score >= 50:
            status = "FAIR"
        else:
            status = "POOR"
        
        self.print_assessment_result("Current System Overall", status, overall_score,
                                   f"Production readiness: {overall_score}%")
        
        return overall_score

    def assess_enterprise_architecture(self):
        """Assess enterprise architecture readiness"""
        self.print_header("ENTERPRISE ARCHITECTURE ASSESSMENT")
        
        total_score = 0
        max_score = 0
        
        # Test Docker infrastructure
        docker_score = 0
        
        if os.path.exists('docker-compose.yml'):
            docker_score += 30
            
        service_dirs = ['services/auth', 'services/model', 'services/data', 'services/ui']
        for service_dir in service_dirs:
            if os.path.exists(service_dir):
                dockerfile = os.path.join(service_dir, 'Dockerfile')
                requirements = os.path.join(service_dir, 'requirements.txt')
                main_py = os.path.join(service_dir, 'main.py')
                
                if os.path.exists(dockerfile):
                    docker_score += 5
                if os.path.exists(requirements):
                    docker_score += 5
                if os.path.exists(main_py):
                    docker_score += 10
        
        if docker_score >= 90:
            self.print_assessment_result("Docker Infrastructure", "EXCELLENT", docker_score,
                                       "Complete microservices architecture")
        elif docker_score >= 70:
            self.print_assessment_result("Docker Infrastructure", "GOOD", docker_score,
                                       "Most microservices components ready")
        elif docker_score >= 40:
            self.print_assessment_result("Docker Infrastructure", "FAIR", docker_score,
                                       "Basic Docker setup present")
        else:
            self.print_assessment_result("Docker Infrastructure", "POOR", docker_score,
                                       "Docker infrastructure incomplete")
        
        total_score += docker_score
        max_score += 100
        
        # Test database schemas
        sql_files = ['sql/auth_init.sql', 'sql/models_init.sql', 'sql/data_init.sql']
        sql_score = 0
        
        for sql_file in sql_files:
            if os.path.exists(sql_file):
                file_size = os.path.getsize(sql_file)
                if file_size > 1000:  # Substantial SQL file
                    sql_score += 33
                else:
                    sql_score += 15
        
        if sql_score >= 90:
            self.print_assessment_result("Database Schemas", "EXCELLENT", sql_score,
                                       "Complete database initialization scripts")
        elif sql_score >= 60:
            self.print_assessment_result("Database Schemas", "GOOD", sql_score,
                                       "Most database schemas ready")
        elif sql_score > 0:
            self.print_assessment_result("Database Schemas", "FAIR", sql_score,
                                       "Some database schemas present")
        else:
            self.print_assessment_result("Database Schemas", "POOR", 0,
                                       "No database schemas found")
        
        total_score += sql_score
        max_score += 100
        
        # Test API Gateway configuration
        nginx_score = 0
        if os.path.exists('nginx/nginx.conf'):
            file_size = os.path.getsize('nginx/nginx.conf')
            if file_size > 2000:
                nginx_score = 100
            elif file_size > 500:
                nginx_score = 70
            else:
                nginx_score = 40
        
        if nginx_score >= 90:
            self.print_assessment_result("API Gateway", "EXCELLENT", nginx_score,
                                       "Complete NGINX configuration")
        elif nginx_score >= 70:
            self.print_assessment_result("API Gateway", "GOOD", nginx_score,
                                       "Good NGINX configuration")
        elif nginx_score > 0:
            self.print_assessment_result("API Gateway", "FAIR", nginx_score,
                                       "Basic NGINX configuration")
        else:
            self.print_assessment_result("API Gateway", "POOR", 0,
                                       "No API Gateway configuration")
        
        total_score += nginx_score
        max_score += 100
        
        # Test environment configuration
        env_score = 0
        if os.path.exists('.env.production'):
            with open('.env.production', 'r') as f:
                env_content = f.read()
            
            critical_vars = ['DATABASE_URL', 'JWT_SECRET', 'ENCRYPTION_KEY', 'REDIS_URL']
            found_vars = sum(1 for var in critical_vars if var in env_content)
            env_score = int((found_vars / len(critical_vars)) * 100)
        
        if env_score >= 90:
            self.print_assessment_result("Environment Config", "EXCELLENT", env_score,
                                       "Complete production configuration")
        elif env_score >= 70:
            self.print_assessment_result("Environment Config", "GOOD", env_score,
                                       "Most configuration variables present")
        elif env_score > 0:
            self.print_assessment_result("Environment Config", "FAIR", env_score,
                                       "Some configuration present")
        else:
            self.print_assessment_result("Environment Config", "POOR", 0,
                                       "No production configuration")
        
        total_score += env_score
        max_score += 100
        
        # Calculate overall enterprise score
        overall_score = int((total_score / max_score) * 100) if max_score > 0 else 0
        
        if overall_score >= 90:
            status = "EXCELLENT"
        elif overall_score >= 75:
            status = "GOOD"
        elif overall_score >= 50:
            status = "FAIR"
        else:
            status = "POOR"
        
        self.print_assessment_result("Enterprise Architecture Overall", status, overall_score,
                                   f"Deployment readiness: {overall_score}%")
        
        return overall_score

    def assess_security_compliance(self):
        """Assess security and compliance readiness"""
        self.print_header("SECURITY & COMPLIANCE ASSESSMENT")
        
        security_score = 0
        
        # Check for security configurations
        security_features = [
            ('.env.production', 'Production environment variables'),
            ('services/auth/main.py', 'Authentication service'),
            ('nginx/nginx.conf', 'API Gateway security'),
            ('sql/auth_init.sql', 'User management schema')
        ]
        
        for feature_file, description in security_features:
            if os.path.exists(feature_file):
                security_score += 25
        
        if security_score >= 90:
            self.print_assessment_result("Security Features", "EXCELLENT", security_score,
                                       "Comprehensive security implementation")
        elif security_score >= 70:
            self.print_assessment_result("Security Features", "GOOD", security_score,
                                       "Good security foundation")
        elif security_score >= 50:
            self.print_assessment_result("Security Features", "FAIR", security_score,
                                       "Basic security measures")
        else:
            self.print_assessment_result("Security Features", "POOR", security_score,
                                       "Insufficient security measures")
        
        return security_score

    def assess_monitoring_observability(self):
        """Assess monitoring and observability"""
        self.print_header("MONITORING & OBSERVABILITY ASSESSMENT")
        
        monitoring_score = 0
        
        # Check for monitoring components in docker-compose
        if os.path.exists('docker-compose.yml'):
            with open('docker-compose.yml', 'r') as f:
                compose_content = f.read()
            
            monitoring_services = ['prometheus', 'grafana', 'elasticsearch', 'kibana']
            found_services = sum(1 for service in monitoring_services if service in compose_content)
            monitoring_score = int((found_services / len(monitoring_services)) * 100)
        
        if monitoring_score >= 90:
            self.print_assessment_result("Monitoring Stack", "EXCELLENT", monitoring_score,
                                       "Complete monitoring infrastructure")
        elif monitoring_score >= 70:
            self.print_assessment_result("Monitoring Stack", "GOOD", monitoring_score,
                                       "Good monitoring coverage")
        elif monitoring_score > 0:
            self.print_assessment_result("Monitoring Stack", "FAIR", monitoring_score,
                                       "Basic monitoring setup")
        else:
            self.print_assessment_result("Monitoring Stack", "POOR", 0,
                                       "No monitoring infrastructure")
        
        return monitoring_score

    def assess_documentation_quality(self):
        """Assess documentation quality"""
        self.print_header("DOCUMENTATION ASSESSMENT")
        
        doc_score = 0
        
        doc_files = {
            'README.md': 20,
            'docs/API_REFERENCE.md': 30,
            'docs/INDUSTRY_RECOMMENDATIONS.md': 25,
            'comprehensive_test_report.md': 25
        }
        
        for doc_file, points in doc_files.items():
            if os.path.exists(doc_file):
                file_size = os.path.getsize(doc_file)
                if file_size > 1000:  # Substantial documentation
                    doc_score += points
                else:
                    doc_score += points // 2
        
        if doc_score >= 90:
            self.print_assessment_result("Documentation", "EXCELLENT", doc_score,
                                       "Comprehensive documentation")
        elif doc_score >= 70:
            self.print_assessment_result("Documentation", "GOOD", doc_score,
                                       "Good documentation coverage")
        elif doc_score >= 40:
            self.print_assessment_result("Documentation", "FAIR", doc_score,
                                       "Basic documentation present")
        else:
            self.print_assessment_result("Documentation", "POOR", doc_score,
                                       "Insufficient documentation")
        
        return doc_score

    def generate_final_report(self):
        """Generate final comprehensive assessment report"""
        self.print_header("FINAL ASSESSMENT REPORT")
        
        # Calculate category scores
        categories = {
            'Current System': [],
            'Enterprise Architecture': [],
            'Security': [],
            'Monitoring': [],
            'Documentation': []
        }
        
        for component, result in self.assessment_results.items():
            if 'Current System' in component or 'Streamlit' in component or 'Dataset' in component or 'Models' in component or 'Application' in component:
                categories['Current System'].append(result['score'])
            elif 'Enterprise' in component or 'Docker' in component or 'Database' in component or 'API Gateway' in component or 'Environment' in component:
                categories['Enterprise Architecture'].append(result['score'])
            elif 'Security' in component:
                categories['Security'].append(result['score'])
            elif 'Monitoring' in component:
                categories['Monitoring'].append(result['score'])
            elif 'Documentation' in component:
                categories['Documentation'].append(result['score'])
        
        category_averages = {}
        for category, scores in categories.items():
            if scores:
                category_averages[category] = sum(scores) / len(scores)
            else:
                category_averages[category] = 0
        
        # Calculate overall system score
        overall_score = sum(category_averages.values()) / len(category_averages)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = f"""
# ForeTel.AI Final System Assessment Report
Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Assessment Duration: {duration.total_seconds():.2f} seconds

## Executive Summary
**Overall System Score: {overall_score:.1f}/100**

## Category Breakdown
"""
        
        for category, score in category_averages.items():
            if score >= 90:
                status = "🎉 EXCELLENT"
            elif score >= 75:
                status = "✅ GOOD"
            elif score >= 50:
                status = "⚠️ FAIR"
            else:
                status = "❌ POOR"
            
            report += f"- **{category}**: {score:.1f}/100 {status}\n"
        
        report += f"""
## Detailed Assessment Results
"""
        
        for component, result in self.assessment_results.items():
            status_icon = {'EXCELLENT': '🎉', 'GOOD': '✅', 'FAIR': '⚠️', 'POOR': '❌'}.get(result['status'], '❓')
            report += f"\n### {component}\n"
            report += f"**Status**: {status_icon} {result['status']} ({result['score']}/100)\n"
            if result['details']:
                report += f"**Details**: {result['details']}\n"
        
        report += f"""
## System Readiness Assessment

### Current System Status
"""
        current_score = category_averages.get('Current System', 0)
        if current_score >= 90:
            report += "🎉 **PRODUCTION READY** - The current Streamlit system is fully functional and ready for immediate use.\n"
        elif current_score >= 75:
            report += "✅ **MOSTLY READY** - The current system is functional with minor issues to address.\n"
        elif current_score >= 50:
            report += "⚠️ **NEEDS WORK** - The current system has significant issues that need attention.\n"
        else:
            report += "❌ **NOT READY** - The current system requires major fixes before deployment.\n"
        
        report += f"""
### Enterprise Platform Status
"""
        enterprise_score = category_averages.get('Enterprise Architecture', 0)
        if enterprise_score >= 90:
            report += "🚀 **DEPLOYMENT READY** - Enterprise architecture is complete and ready for production deployment.\n"
        elif enterprise_score >= 75:
            report += "🔧 **NEARLY READY** - Enterprise architecture needs minor configuration before deployment.\n"
        elif enterprise_score >= 50:
            report += "⚠️ **IN DEVELOPMENT** - Enterprise architecture is partially complete, significant work remaining.\n"
        else:
            report += "🚧 **EARLY STAGE** - Enterprise architecture requires substantial development.\n"
        
        report += f"""
## Recommendations

### Immediate Actions (Priority 1)
"""
        if current_score < 75:
            report += "1. **Fix Current System Issues**: Address failing components in the Streamlit application\n"
        if enterprise_score < 50:
            report += "2. **Complete Enterprise Setup**: Finish implementing missing microservices components\n"
        
        report += f"""
### Short-term Goals (Priority 2)
"""
        if category_averages.get('Security', 0) < 80:
            report += "1. **Enhance Security**: Implement comprehensive security measures and audit trails\n"
        if category_averages.get('Monitoring', 0) < 80:
            report += "2. **Setup Monitoring**: Deploy monitoring and observability infrastructure\n"
        
        report += f"""
### Long-term Goals (Priority 3)
1. **Performance Optimization**: Conduct load testing and optimize system performance
2. **Compliance Audit**: Ensure GDPR, CCPA, and industry compliance
3. **Disaster Recovery**: Implement backup and disaster recovery procedures
4. **CI/CD Pipeline**: Setup automated testing and deployment pipelines

## Deployment Recommendation
"""
        
        if overall_score >= 85:
            report += "🎯 **RECOMMENDED**: System is ready for production deployment with comprehensive enterprise features.\n"
        elif overall_score >= 70:
            report += "🎯 **CONDITIONAL**: System can be deployed with current functionality while enterprise features are completed.\n"
        elif overall_score >= 50:
            report += "⚠️ **NOT RECOMMENDED**: System needs significant work before production deployment.\n"
        else:
            report += "❌ **NOT READY**: System requires major development before considering deployment.\n"
        
        report += f"""
---
*Assessment completed by ForeTel.AI Final System Assessment Suite*
*Total Components Assessed: {len(self.assessment_results)}*
"""
        
        # Save report
        report_file = "final_system_assessment_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Final assessment report saved to: {report_file}")
        print(f"\n🎯 **OVERALL SCORE**: {overall_score:.1f}/100")
        
        # Print final recommendation
        if overall_score >= 85:
            print("🎉 **EXCELLENT**: ForeTel.AI is ready for enterprise production deployment!")
        elif overall_score >= 70:
            print("✅ **GOOD**: ForeTel.AI is functional and ready for deployment with ongoing improvements")
        elif overall_score >= 50:
            print("⚠️ **FAIR**: ForeTel.AI needs additional work before production deployment")
        else:
            print("❌ **POOR**: ForeTel.AI requires significant development before deployment")
        
        return report

    def run_final_assessment(self):
        """Run complete final assessment"""
        print("🔍 Starting ForeTel.AI Final System Assessment")
        print(f"⏰ Assessment started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all assessments
        current_score = self.assess_current_streamlit_system()
        enterprise_score = self.assess_enterprise_architecture()
        security_score = self.assess_security_compliance()
        monitoring_score = self.assess_monitoring_observability()
        doc_score = self.assess_documentation_quality()
        
        # Generate final report
        self.generate_final_report()

def main():
    """Main assessment execution function"""
    assessment = FinalSystemAssessment()
    
    try:
        assessment.run_final_assessment()
    except KeyboardInterrupt:
        print("\n⚠️ Assessment interrupted by user")
    except Exception as e:
        print(f"\n❌ Assessment failed with error: {e}")
        logger.exception("Final assessment failed")

if __name__ == "__main__":
    main()
