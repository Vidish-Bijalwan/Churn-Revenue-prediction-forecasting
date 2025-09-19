
# ForeTel.AI Enterprise Platform Test Report
Generated: 2025-09-19 23:10:00
Duration: 2.10 seconds

## Summary
- **Total Tests**: 25
- **Passed**: 2 ✅
- **Failed**: 23 ❌
- **Success Rate**: 8.0%

## Test Results
- **Docker Compose File**: ✅ PASS
  - Found docker-compose.yml
- **Docker Daemon**: ✅ PASS
  - Docker daemon is running
- **Docker Compose Config**: ❌ FAIL
  - Config test failed: [Errno 2] No such file or directory: 'docker-compose'
- **PostgreSQL Auth Connection**: ❌ FAIL
  - Connection failed - service may not be running
- **PostgreSQL Models Connection**: ❌ FAIL
  - Connection failed - service may not be running
- **PostgreSQL Data Connection**: ❌ FAIL
  - Connection failed - service may not be running
- **Redis Connection**: ❌ FAIL
  - Redis service not running
- **Kafka Connection**: ❌ FAIL
  - Kafka not available: NoBrokersAvailable
- **Auth Health**: ❌ FAIL
  - Service not responding - may not be running
- **Model Health**: ❌ FAIL
  - Service not responding - may not be running
- **Data Health**: ❌ FAIL
  - Service not responding - may not be running
- **Ui Health**: ❌ FAIL
  - Service not responding - may not be running
- **Gateway Health**: ❌ FAIL
  - Service not responding - may not be running
- **Authentication Service**: ❌ FAIL
  - Auth service not responding
- **Model Service**: ❌ FAIL
  - Model service not responding
- **Data Service**: ❌ FAIL
  - Data service not responding
- **Prometheus Health**: ❌ FAIL
  - Service not responding
- **Grafana Health**: ❌ FAIL
  - Service not responding
- **Elasticsearch Health**: ❌ FAIL
  - Service not responding
- **Kibana Health**: ❌ FAIL
  - Service not responding
- **Rate Limiting**: ❌ FAIL
  - Rate limiting test error: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /health (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7285447447c0>: Failed to establish a new connection: [Errno 111] Connection refused'))
- **HTTPS Redirect**: ❌ FAIL
  - HTTPS test error: HTTPConnectionPool(host='localhost', port=80): Max retries exceeded with url: / (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x728544744af0>: Failed to establish a new connection: [Errno 111] Connection refused'))
- **Auth Response Time**: ❌ FAIL
  - Error: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /health (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x728544883790>: Failed to establish a new connection: [Errno 111] Connection refused'))
- **Model Response Time**: ❌ FAIL
  - Error: HTTPConnectionPool(host='localhost', port=8001): Max retries exceeded with url: /health (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x728544883ce0>: Failed to establish a new connection: [Errno 111] Connection refused'))
- **Data Response Time**: ❌ FAIL
  - Error: HTTPConnectionPool(host='localhost', port=8002): Max retries exceeded with url: /health (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7285448839b0>: Failed to establish a new connection: [Errno 111] Connection refused'))

## Recommendations
⚠️ 23 tests failed. Review the failures above and:
1. Ensure all services are running with `docker-compose up -d`
2. Check service logs for errors
3. Verify network connectivity between services
4. Confirm all environment variables are set correctly
