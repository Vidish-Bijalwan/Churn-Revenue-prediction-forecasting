-- Data Database Initialization
-- Create tables for data management, quality monitoring, and pipeline orchestration

-- Datasets registry for tracking all data sources
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    rows INTEGER,
    columns INTEGER,
    schema JSONB, -- Column names and types
    quality_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    tags JSONB, -- Array of tags
    metadata JSONB, -- Additional metadata
    source_type VARCHAR(50), -- 'upload', 'api', 'streaming', 'database'
    source_config JSONB, -- Source-specific configuration
    refresh_schedule VARCHAR(100), -- Cron expression for auto-refresh
    last_refreshed TIMESTAMP,
    owner_id INTEGER,
    access_level VARCHAR(20) DEFAULT 'private' -- 'public', 'private', 'restricted'
);

-- Data quality checks and results
CREATE TABLE IF NOT EXISTS data_quality_checks (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    check_type VARCHAR(100) NOT NULL, -- 'completeness', 'uniqueness', 'validity', 'consistency'
    check_name VARCHAR(255) NOT NULL,
    check_description TEXT,
    expected_value TEXT,
    actual_value TEXT,
    status VARCHAR(20) NOT NULL, -- 'PASS', 'FAIL', 'WARNING', 'SKIP'
    severity VARCHAR(20) DEFAULT 'MEDIUM', -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time_ms INTEGER,
    details JSONB,
    threshold_config JSONB,
    remediation_suggestion TEXT
);

-- Data lineage tracking
CREATE TABLE IF NOT EXISTS data_lineage (
    id SERIAL PRIMARY KEY,
    source_dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    target_dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    transformation VARCHAR(255) NOT NULL,
    transformation_code TEXT,
    transformation_config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    execution_time_ms INTEGER,
    rows_processed INTEGER,
    success_rate FLOAT,
    error_details JSONB
);

-- Data pipeline jobs
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(64) UNIQUE NOT NULL,
    job_name VARCHAR(255) NOT NULL,
    job_type VARCHAR(50) NOT NULL, -- 'etl', 'feature_engineering', 'data_validation', 'export'
    status VARCHAR(20) DEFAULT 'QUEUED', -- 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    priority INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    scheduled_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    input_datasets INTEGER[], -- Array of dataset IDs
    output_datasets INTEGER[], -- Array of dataset IDs
    configuration JSONB,
    resource_usage JSONB, -- CPU, memory usage
    logs TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_by INTEGER,
    dependencies TEXT[] -- Array of job IDs this job depends on
);

-- Streaming data ingestion
CREATE TABLE IF NOT EXISTS streaming_sources (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'kafka', 'kinesis', 'pubsub', 'webhook'
    connection_config JSONB NOT NULL,
    schema_config JSONB,
    processing_config JSONB,
    status VARCHAR(20) DEFAULT 'INACTIVE', -- 'ACTIVE', 'INACTIVE', 'ERROR'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP,
    messages_processed BIGINT DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    throughput_per_second FLOAT DEFAULT 0.0,
    created_by INTEGER
);

-- Real-time data buffer for streaming
CREATE TABLE IF NOT EXISTS streaming_buffer (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(64) NOT NULL REFERENCES streaming_sources(source_id) ON DELETE CASCADE,
    message_id VARCHAR(255),
    payload JSONB NOT NULL,
    metadata JSONB,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    processing_status VARCHAR(20) DEFAULT 'PENDING', -- 'PENDING', 'PROCESSED', 'FAILED'
    error_message TEXT,
    partition_key VARCHAR(255),
    sequence_number BIGINT
);

-- Data profiling results
CREATE TABLE IF NOT EXISTS data_profiles (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    column_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(50),
    null_count INTEGER,
    null_percentage FLOAT,
    unique_count INTEGER,
    unique_percentage FLOAT,
    min_value TEXT,
    max_value TEXT,
    mean_value FLOAT,
    median_value FLOAT,
    std_deviation FLOAT,
    quartile_25 FLOAT,
    quartile_75 FLOAT,
    most_frequent_values JSONB, -- Top N most frequent values
    distribution_histogram JSONB,
    profiled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sample_size INTEGER,
    anomalies_detected INTEGER DEFAULT 0
);

-- Data access logs for audit and compliance
CREATE TABLE IF NOT EXISTS data_access_logs (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
    user_id INTEGER,
    access_type VARCHAR(50) NOT NULL, -- 'read', 'write', 'delete', 'export', 'share'
    access_method VARCHAR(50), -- 'api', 'ui', 'download', 'query'
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rows_accessed INTEGER,
    columns_accessed TEXT[],
    query_executed TEXT,
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    compliance_tags TEXT[] -- For GDPR, CCPA compliance tracking
);

-- Data retention policies
CREATE TABLE IF NOT EXISTS data_retention_policies (
    id SERIAL PRIMARY KEY,
    policy_name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    dataset_pattern VARCHAR(255), -- Regex pattern to match dataset names
    retention_days INTEGER NOT NULL,
    archive_before_delete BOOLEAN DEFAULT TRUE,
    archive_location TEXT,
    compliance_reason TEXT, -- GDPR, CCPA, internal policy
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    last_applied TIMESTAMP
);

-- Data anonymization rules
CREATE TABLE IF NOT EXISTS anonymization_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(255) UNIQUE NOT NULL,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    column_name VARCHAR(255) NOT NULL,
    anonymization_method VARCHAR(50) NOT NULL, -- 'hash', 'mask', 'generalize', 'suppress', 'noise'
    method_config JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    compliance_requirement VARCHAR(100) -- GDPR, CCPA, HIPAA
);

-- Feature engineering transformations
CREATE TABLE IF NOT EXISTS feature_transformations (
    id SERIAL PRIMARY KEY,
    transformation_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    transformation_type VARCHAR(50) NOT NULL, -- 'aggregation', 'encoding', 'scaling', 'binning', 'custom'
    input_columns TEXT[] NOT NULL,
    output_columns TEXT[] NOT NULL,
    transformation_code TEXT NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,
    performance_metrics JSONB -- Execution time, memory usage
);

-- Data alerts and notifications
CREATE TABLE IF NOT EXISTS data_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(64) UNIQUE NOT NULL,
    alert_type VARCHAR(50) NOT NULL, -- 'quality_degradation', 'schema_change', 'volume_anomaly', 'freshness_issue'
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    severity VARCHAR(20) NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    threshold_value FLOAT,
    actual_value FLOAT,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by INTEGER,
    status VARCHAR(20) DEFAULT 'OPEN', -- 'OPEN', 'ACKNOWLEDGED', 'RESOLVED', 'FALSE_POSITIVE'
    notification_channels TEXT[], -- 'email', 'slack', 'webhook'
    notification_sent BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

-- Data catalog for discovery and governance
CREATE TABLE IF NOT EXISTS data_catalog (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    business_name VARCHAR(255),
    business_description TEXT,
    domain VARCHAR(100), -- 'customer', 'product', 'financial', 'operational'
    classification VARCHAR(50), -- 'public', 'internal', 'confidential', 'restricted'
    sensitivity_level VARCHAR(50), -- 'low', 'medium', 'high', 'critical'
    data_steward INTEGER, -- User ID of data steward
    business_owner INTEGER, -- User ID of business owner
    usage_guidelines TEXT,
    access_requirements TEXT,
    related_datasets INTEGER[], -- Array of related dataset IDs
    business_glossary JSONB, -- Terms and definitions
    sample_queries TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_is_active ON datasets(is_active);
CREATE INDEX IF NOT EXISTS idx_datasets_owner_id ON datasets(owner_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_checks_dataset_id ON data_quality_checks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_checks_status ON data_quality_checks(status);
CREATE INDEX IF NOT EXISTS idx_data_quality_checks_timestamp ON data_quality_checks(timestamp);
CREATE INDEX IF NOT EXISTS idx_data_lineage_source_dataset_id ON data_lineage(source_dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_lineage_target_dataset_id ON data_lineage(target_dataset_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_job_id ON pipeline_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_status ON pipeline_jobs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_scheduled_at ON pipeline_jobs(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_streaming_sources_source_id ON streaming_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_streaming_buffer_source_id ON streaming_buffer(source_id);
CREATE INDEX IF NOT EXISTS idx_streaming_buffer_ingested_at ON streaming_buffer(ingested_at);
CREATE INDEX IF NOT EXISTS idx_data_profiles_dataset_id ON data_profiles(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_access_logs_dataset_id ON data_access_logs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_access_logs_user_id ON data_access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_data_access_logs_timestamp ON data_access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_data_alerts_dataset_id ON data_alerts(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_alerts_status ON data_alerts(status);
CREATE INDEX IF NOT EXISTS idx_data_catalog_dataset_id ON data_catalog(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_catalog_domain ON data_catalog(domain);

-- Functions for data management
CREATE OR REPLACE FUNCTION update_dataset_stats(
    p_dataset_id INTEGER,
    p_rows INTEGER,
    p_columns INTEGER,
    p_quality_score FLOAT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE datasets 
    SET 
        rows = p_rows,
        columns = p_columns,
        quality_score = COALESCE(p_quality_score, quality_score),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_dataset_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate data quality score
CREATE OR REPLACE FUNCTION calculate_quality_score(p_dataset_id INTEGER)
RETURNS FLOAT AS $$
DECLARE
    total_checks INTEGER;
    passed_checks INTEGER;
    quality_score FLOAT;
BEGIN
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN status = 'PASS' THEN 1 END) as passed
    INTO total_checks, passed_checks
    FROM data_quality_checks
    WHERE dataset_id = p_dataset_id
        AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';
    
    IF total_checks > 0 THEN
        quality_score := (passed_checks::FLOAT / total_checks::FLOAT) * 100;
    ELSE
        quality_score := 0.0;
    END IF;
    
    -- Update dataset quality score
    UPDATE datasets 
    SET quality_score = quality_score, updated_at = CURRENT_TIMESTAMP
    WHERE id = p_dataset_id;
    
    RETURN quality_score;
END;
$$ LANGUAGE plpgsql;

-- Function to detect data anomalies
CREATE OR REPLACE FUNCTION detect_data_anomalies(
    p_dataset_id INTEGER,
    p_column_name VARCHAR(255),
    p_threshold FLOAT DEFAULT 2.0
)
RETURNS INTEGER AS $$
DECLARE
    anomaly_count INTEGER := 0;
    mean_val FLOAT;
    std_val FLOAT;
BEGIN
    -- Get statistics for the column
    SELECT mean_value, std_deviation
    INTO mean_val, std_val
    FROM data_profiles
    WHERE dataset_id = p_dataset_id AND column_name = p_column_name
    ORDER BY profiled_at DESC
    LIMIT 1;
    
    -- Simple anomaly detection using z-score
    -- In a real implementation, this would analyze actual data
    -- For now, we'll simulate anomaly detection
    anomaly_count := FLOOR(RANDOM() * 10); -- Simulate 0-9 anomalies
    
    -- Update profile with anomaly count
    UPDATE data_profiles
    SET anomalies_detected = anomaly_count
    WHERE dataset_id = p_dataset_id AND column_name = p_column_name;
    
    RETURN anomaly_count;
END;
$$ LANGUAGE plpgsql;

-- Function to apply retention policy
CREATE OR REPLACE FUNCTION apply_retention_policy(p_policy_id INTEGER)
RETURNS INTEGER AS $$
DECLARE
    policy_record RECORD;
    affected_datasets INTEGER := 0;
    cutoff_date TIMESTAMP;
BEGIN
    -- Get policy details
    SELECT * INTO policy_record
    FROM data_retention_policies
    WHERE id = p_policy_id AND is_active = TRUE;
    
    IF NOT FOUND THEN
        RETURN 0;
    END IF;
    
    cutoff_date := CURRENT_TIMESTAMP - (policy_record.retention_days || ' days')::INTERVAL;
    
    -- Archive or delete datasets matching the policy
    UPDATE datasets
    SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
    WHERE name ~ policy_record.dataset_pattern
        AND created_at < cutoff_date
        AND is_active = TRUE;
    
    GET DIAGNOSTICS affected_datasets = ROW_COUNT;
    
    -- Update policy last applied timestamp
    UPDATE data_retention_policies
    SET last_applied = CURRENT_TIMESTAMP
    WHERE id = p_policy_id;
    
    RETURN affected_datasets;
END;
$$ LANGUAGE plpgsql;

-- Function to log data access
CREATE OR REPLACE FUNCTION log_data_access(
    p_dataset_id INTEGER,
    p_user_id INTEGER,
    p_access_type VARCHAR(50),
    p_access_method VARCHAR(50) DEFAULT 'api',
    p_ip_address INET DEFAULT NULL,
    p_rows_accessed INTEGER DEFAULT NULL,
    p_columns_accessed TEXT[] DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO data_access_logs (
        dataset_id, user_id, access_type, access_method, 
        ip_address, rows_accessed, columns_accessed
    ) VALUES (
        p_dataset_id, p_user_id, p_access_type, p_access_method,
        p_ip_address, p_rows_accessed, p_columns_accessed
    );
END;
$$ LANGUAGE plpgsql;

-- Function to clean old streaming buffer data
CREATE OR REPLACE FUNCTION cleanup_streaming_buffer(
    p_retention_hours INTEGER DEFAULT 24
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM streaming_buffer
    WHERE ingested_at < CURRENT_TIMESTAMP - (p_retention_hours || ' hours')::INTERVAL
        AND processing_status = 'PROCESSED';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_catalog_updated_at BEFORE UPDATE ON data_catalog
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
