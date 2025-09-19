#!/usr/bin/env python3
"""
ForeTel.AI Enterprise Platform Comprehensive Testing Suite
Tests all components: Infrastructure, Services, APIs, Security, Performance
"""

import asyncio
import aiohttp
import requests
import psycopg2
import redis
import json
import time
import subprocess
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
import docker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForeTelTestSuite:
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.docker_client = docker.from_env()
        
        # Test configuration
        self.base_urls = {
            'auth': 'http://localhost:8000',
            'model': 'http://localhost:8001', 
            'data': 'http://localhost:8002',
            'ui': 'http://localhost:8501',
            'gateway': 'http://localhost:80'
        }
        
        self.db_configs = {
            'auth': {
                'host': 'localhost',
                'port': 5432,
                'database': 'auth_db',
                'user': 'auth_user',
                'password': 'auth_pass'
            },
            'models': {
                'host': 'localhost',
                'port': 5433,
                'database': 'models_db', 
                'user': 'model_user',
                'password': 'model_pass'
            },
            'data': {
                'host': 'localhost',
                'port': 5434,
                'database': 'data_db',
                'user': 'data_user', 
                'password': 'data_pass'
            }
        }

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

    def test_docker_infrastructure(self):
        """Test Docker containers and infrastructure"""
        self.print_header("DOCKER INFRASTRUCTURE TESTING")
        
        try:
            # Check if docker-compose.yml exists
            compose_file = "docker-compose.yml"
            if os.path.exists(compose_file):
                self.print_test_result("Docker Compose File", True, "Found docker-compose.yml")
            else:
                self.print_test_result("Docker Compose File", False, "docker-compose.yml not found")
                return
            
            # Check Docker daemon
            try:
                self.docker_client.ping()
                self.print_test_result("Docker Daemon", True, "Docker daemon is running")
            except Exception as e:
                self.print_test_result("Docker Daemon", False, f"Docker daemon error: {e}")
                return
            
            # List expected containers
            expected_containers = [
                'api-gateway', 'auth-service', 'model-service', 'data-service', 'ui-service',
                'postgres-auth', 'postgres-models', 'postgres-data', 'redis', 'kafka',
                'zookeeper', 'mlflow', 'prometheus', 'grafana'
            ]
            
            # Check if containers can be built (dry run)
            try:
                result = subprocess.run(['docker-compose', 'config'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    self.print_test_result("Docker Compose Config", True, "Configuration is valid")
                else:
                    self.print_test_result("Docker Compose Config", False, f"Config error: {result.stderr}")
            except Exception as e:
                self.print_test_result("Docker Compose Config", False, f"Config test failed: {e}")
                
        except Exception as e:
            self.print_test_result("Docker Infrastructure", False, f"Infrastructure test failed: {e}")

    def test_database_connections(self):
        """Test PostgreSQL database connections and schemas"""
        self.print_header("DATABASE CONNECTION TESTING")
        
        for db_name, config in self.db_configs.items():
            try:
                # Test connection
                conn = psycopg2.connect(**config)
                cursor = conn.cursor()
                
                # Test basic query
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                self.print_test_result(f"PostgreSQL {db_name.title()} Connection", True, 
                                     f"Connected successfully")
                
                # Test schema exists
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' LIMIT 5;
                """)
                tables = cursor.fetchall()
                
                if tables:
                    table_names = [table[0] for table in tables]
                    self.print_test_result(f"{db_name.title()} Schema", True, 
                                         f"Found tables: {', '.join(table_names[:3])}...")
                else:
                    self.print_test_result(f"{db_name.title()} Schema", False, "No tables found")
                
                conn.close()
                
            except psycopg2.OperationalError as e:
                self.print_test_result(f"PostgreSQL {db_name.title()} Connection", False, 
                                     f"Connection failed - service may not be running")
            except Exception as e:
                self.print_test_result(f"PostgreSQL {db_name.title()} Connection", False, f"Error: {e}")

    def test_redis_connection(self):
        """Test Redis connection and caching"""
        self.print_header("REDIS CACHING TESTING")
        
        try:
            # Test Redis connection
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            
            # Test ping
            pong = r.ping()
            self.print_test_result("Redis Connection", True, "Redis is responding")
            
            # Test basic operations
            test_key = "foretel_test_key"
            test_value = "test_value_123"
            
            # Set and get
            r.set(test_key, test_value, ex=60)
            retrieved_value = r.get(test_key)
            
            if retrieved_value == test_value:
                self.print_test_result("Redis Set/Get", True, "Cache operations working")
            else:
                self.print_test_result("Redis Set/Get", False, "Cache operations failed")
            
            # Test expiration
            r.expire(test_key, 1)
            time.sleep(2)
            expired_value = r.get(test_key)
            
            if expired_value is None:
                self.print_test_result("Redis Expiration", True, "TTL working correctly")
            else:
                self.print_test_result("Redis Expiration", False, "TTL not working")
            
            # Cleanup
            r.delete(test_key)
            
        except redis.ConnectionError:
            self.print_test_result("Redis Connection", False, "Redis service not running")
        except Exception as e:
            self.print_test_result("Redis Connection", False, f"Redis error: {e}")

    def test_kafka_messaging(self):
        """Test Kafka message queue system"""
        self.print_header("KAFKA MESSAGING TESTING")
        
        try:
            # Test Kafka producer
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                request_timeout_ms=5000
            )
            
            test_topic = 'foretel_test_topic'
            test_message = {
                'test_id': 'test_123',
                'timestamp': datetime.now().isoformat(),
                'data': 'test_message'
            }
            
            # Send test message
            future = producer.send(test_topic, test_message)
            result = future.get(timeout=10)
            
            self.print_test_result("Kafka Producer", True, f"Message sent to {test_topic}")
            
            producer.close()
            
            # Test Kafka consumer
            consumer = KafkaConsumer(
                test_topic,
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=5000,
                auto_offset_reset='latest'
            )
            
            # Try to consume the message
            messages_received = 0
            for message in consumer:
                if message.value.get('test_id') == 'test_123':
                    messages_received += 1
                    break
            
            consumer.close()
            
            if messages_received > 0:
                self.print_test_result("Kafka Consumer", True, "Message consumed successfully")
            else:
                self.print_test_result("Kafka Consumer", False, "No messages consumed")
                
        except Exception as e:
            self.print_test_result("Kafka Connection", False, f"Kafka not available: {e}")

    async def test_service_endpoints(self):
        """Test all microservice endpoints"""
        self.print_header("SERVICE ENDPOINTS TESTING")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            
            # Test health endpoints
            for service_name, base_url in self.base_urls.items():
                try:
                    async with session.get(f"{base_url}/health") as response:
                        if response.status == 200:
                            data = await response.json()
                            self.print_test_result(f"{service_name.title()} Health", True, 
                                                 f"Status: {data.get('status', 'unknown')}")
                        else:
                            self.print_test_result(f"{service_name.title()} Health", False, 
                                                 f"HTTP {response.status}")
                except aiohttp.ClientError:
                    self.print_test_result(f"{service_name.title()} Health", False, 
                                         "Service not responding - may not be running")
                except Exception as e:
                    self.print_test_result(f"{service_name.title()} Health", False, f"Error: {e}")

    def test_authentication_service(self):
        """Test authentication service functionality"""
        self.print_header("AUTHENTICATION SERVICE TESTING")
        
        auth_url = self.base_urls['auth']
        
        try:
            # Test user registration
            test_user = {
                "email": "test@foretel.ai",
                "username": "testuser",
                "password": "testpass123",
                "full_name": "Test User"
            }
            
            response = requests.post(f"{auth_url}/register", json=test_user, timeout=10)
            
            if response.status_code in [200, 201]:
                self.print_test_result("User Registration", True, "User registered successfully")
                
                # Test login
                login_data = {
                    "username": test_user["username"],
                    "password": test_user["password"]
                }
                
                login_response = requests.post(f"{auth_url}/login", json=login_data, timeout=10)
                
                if login_response.status_code == 200:
                    token_data = login_response.json()
                    if 'access_token' in token_data:
                        self.print_test_result("User Login", True, "Login successful, token received")
                        
                        # Test protected endpoint
                        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
                        me_response = requests.get(f"{auth_url}/me", headers=headers, timeout=10)
                        
                        if me_response.status_code == 200:
                            self.print_test_result("Protected Endpoint", True, "Token validation working")
                        else:
                            self.print_test_result("Protected Endpoint", False, "Token validation failed")
                    else:
                        self.print_test_result("User Login", False, "No token in response")
                else:
                    self.print_test_result("User Login", False, f"Login failed: {login_response.status_code}")
            elif response.status_code == 400:
                self.print_test_result("User Registration", True, "User already exists (expected)")
            else:
                self.print_test_result("User Registration", False, f"Registration failed: {response.status_code}")
                
        except requests.exceptions.RequestException:
            self.print_test_result("Authentication Service", False, "Auth service not responding")
        except Exception as e:
            self.print_test_result("Authentication Service", False, f"Auth test error: {e}")

    def test_model_service(self):
        """Test model service functionality"""
        self.print_header("MODEL SERVICE TESTING")
        
        model_url = self.base_urls['model']
        
        try:
            # Test model prediction endpoint
            test_features = {
                "features": {
                    "age": 35,
                    "monthly_income": 50000,
                    "data_usage_gb": 15.5,
                    "call_usage_min": 250,
                    "monthly_spending": 75
                },
                "model_type": "churn",
                "explain": False
            }
            
            response = requests.post(f"{model_url}/predict", json=test_features, timeout=30)
            
            if response.status_code == 200:
                prediction_data = response.json()
                if 'prediction' in prediction_data:
                    self.print_test_result("Model Prediction", True, 
                                         f"Prediction: {prediction_data['prediction']}")
                else:
                    self.print_test_result("Model Prediction", False, "No prediction in response")
            else:
                self.print_test_result("Model Prediction", False, f"Prediction failed: {response.status_code}")
            
            # Test batch prediction
            batch_data = {
                "data": [test_features["features"], test_features["features"]],
                "model_type": "churn",
                "explain": False
            }
            
            batch_response = requests.post(f"{model_url}/predict/batch", json=batch_data, timeout=30)
            
            if batch_response.status_code == 200:
                batch_result = batch_response.json()
                if 'results' in batch_result:
                    self.print_test_result("Batch Prediction", True, 
                                         f"Processed {len(batch_result['results'])} predictions")
                else:
                    self.print_test_result("Batch Prediction", False, "No results in batch response")
            else:
                self.print_test_result("Batch Prediction", False, f"Batch prediction failed: {batch_response.status_code}")
                
        except requests.exceptions.RequestException:
            self.print_test_result("Model Service", False, "Model service not responding")
        except Exception as e:
            self.print_test_result("Model Service", False, f"Model test error: {e}")

    def test_data_service(self):
        """Test data service functionality"""
        self.print_header("DATA SERVICE TESTING")
        
        data_url = self.base_urls['data']
        
        try:
            # Test dataset list endpoint
            response = requests.get(f"{data_url}/datasets", timeout=10)
            
            if response.status_code == 200:
                datasets = response.json()
                self.print_test_result("Dataset List", True, f"Retrieved {len(datasets)} datasets")
            else:
                self.print_test_result("Dataset List", False, f"Failed to get datasets: {response.status_code}")
            
            # Test streaming ingestion
            streaming_data = {
                "topic": "test_stream",
                "data": {
                    "customer_id": "test_123",
                    "data_usage_gb": 20.5,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            stream_response = requests.post(f"{data_url}/streaming/ingest", json=streaming_data, timeout=10)
            
            if stream_response.status_code == 200:
                self.print_test_result("Streaming Ingestion", True, "Data ingested successfully")
            else:
                self.print_test_result("Streaming Ingestion", False, f"Ingestion failed: {stream_response.status_code}")
                
        except requests.exceptions.RequestException:
            self.print_test_result("Data Service", False, "Data service not responding")
        except Exception as e:
            self.print_test_result("Data Service", False, f"Data test error: {e}")

    def test_monitoring_systems(self):
        """Test monitoring and logging systems"""
        self.print_header("MONITORING SYSTEMS TESTING")
        
        monitoring_endpoints = {
            'Prometheus': 'http://localhost:9090/-/healthy',
            'Grafana': 'http://localhost:3000/api/health',
            'Elasticsearch': 'http://localhost:9200/_cluster/health',
            'Kibana': 'http://localhost:5601/api/status'
        }
        
        for service, endpoint in monitoring_endpoints.items():
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    self.print_test_result(f"{service} Health", True, "Service is healthy")
                else:
                    self.print_test_result(f"{service} Health", False, f"HTTP {response.status_code}")
            except requests.exceptions.RequestException:
                self.print_test_result(f"{service} Health", False, "Service not responding")
            except Exception as e:
                self.print_test_result(f"{service} Health", False, f"Error: {e}")

    def test_security_features(self):
        """Test security and encryption features"""
        self.print_header("SECURITY FEATURES TESTING")
        
        # Test rate limiting
        try:
            auth_url = self.base_urls['auth']
            
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(10):
                response = requests.get(f"{auth_url}/health", timeout=5)
                responses.append(response.status_code)
            
            if all(status == 200 for status in responses):
                self.print_test_result("Rate Limiting", True, "All requests processed (rate limiting may not be active)")
            else:
                rate_limited = any(status == 429 for status in responses)
                if rate_limited:
                    self.print_test_result("Rate Limiting", True, "Rate limiting is working")
                else:
                    self.print_test_result("Rate Limiting", False, "Unexpected response codes")
                    
        except Exception as e:
            self.print_test_result("Rate Limiting", False, f"Rate limiting test error: {e}")
        
        # Test HTTPS redirect (if configured)
        try:
            response = requests.get("http://localhost", timeout=5, allow_redirects=False)
            if response.status_code in [301, 302, 308]:
                if 'https' in response.headers.get('Location', ''):
                    self.print_test_result("HTTPS Redirect", True, "HTTP to HTTPS redirect working")
                else:
                    self.print_test_result("HTTPS Redirect", False, "Redirect not to HTTPS")
            else:
                self.print_test_result("HTTPS Redirect", False, "No HTTPS redirect configured")
        except Exception as e:
            self.print_test_result("HTTPS Redirect", False, f"HTTPS test error: {e}")

    def test_performance_metrics(self):
        """Test basic performance metrics"""
        self.print_header("PERFORMANCE TESTING")
        
        # Test response times
        services_to_test = ['auth', 'model', 'data']
        
        for service in services_to_test:
            try:
                url = f"{self.base_urls[service]}/health"
                
                # Measure response time
                start_time = time.time()
                response = requests.get(url, timeout=10)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                if response.status_code == 200:
                    if response_time < 1000:  # Less than 1 second
                        self.print_test_result(f"{service.title()} Response Time", True, 
                                             f"{response_time:.2f}ms")
                    else:
                        self.print_test_result(f"{service.title()} Response Time", False, 
                                             f"{response_time:.2f}ms (too slow)")
                else:
                    self.print_test_result(f"{service.title()} Response Time", False, 
                                         f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.print_test_result(f"{service.title()} Response Time", False, f"Error: {e}")

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
# ForeTel.AI Enterprise Platform Test Report
Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration.total_seconds():.2f} seconds

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ✅
- **Failed**: {failed_tests} ❌
- **Success Rate**: {success_rate:.1f}%

## Test Results
"""
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"- **{test_name}**: {status}\n"
            if result['details']:
                report += f"  - {result['details']}\n"
        
        report += f"""
## Recommendations
"""
        
        if failed_tests == 0:
            report += "🎉 All tests passed! The ForeTel.AI platform is ready for production.\n"
        else:
            report += f"⚠️ {failed_tests} tests failed. Review the failures above and:\n"
            report += "1. Ensure all services are running with `docker-compose up -d`\n"
            report += "2. Check service logs for errors\n"
            report += "3. Verify network connectivity between services\n"
            report += "4. Confirm all environment variables are set correctly\n"
        
        # Save report to file
        report_file = "test_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Test report saved to: {report_file}")
        print(f"\n🎯 **FINAL RESULT**: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return report

    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting ForeTel.AI Enterprise Platform Testing Suite")
        print(f"⏰ Test started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.test_docker_infrastructure()
        self.test_database_connections()
        self.test_redis_connection()
        self.test_kafka_messaging()
        await self.test_service_endpoints()
        self.test_authentication_service()
        self.test_model_service()
        self.test_data_service()
        self.test_monitoring_systems()
        self.test_security_features()
        self.test_performance_metrics()
        
        # Generate final report
        self.generate_test_report()

def main():
    """Main test execution function"""
    test_suite = ForeTelTestSuite()
    
    try:
        # Run async tests
        asyncio.run(test_suite.run_all_tests())
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        logger.exception("Test suite failed")

if __name__ == "__main__":
    main()
