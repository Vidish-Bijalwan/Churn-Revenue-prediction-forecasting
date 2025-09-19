-- Models Database Initialization
-- Create tables for ML model management, versioning, and performance tracking

-- Model registry for tracking all trained models
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'churn', 'revenue', 'classification', 'regression'
    algorithm VARCHAR(100) NOT NULL, -- 'xgboost', 'random_forest', 'neural_network', etc.
    framework VARCHAR(50), -- 'scikit-learn', 'tensorflow', 'pytorch', 'h2o'
    file_path TEXT NOT NULL,
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    status VARCHAR(20) DEFAULT 'TRAINING', -- TRAINING, READY, DEPLOYED, ARCHIVED, FAILED
    description TEXT,
    tags TEXT[], -- Array of tags for categorization
    metadata JSONB, -- Model-specific metadata
    hyperparameters JSONB, -- Model hyperparameters
    training_config JSONB, -- Training configuration
    deployment_config JSONB -- Deployment configuration
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50), -- 'training', 'validation', 'test', 'production'
    dataset_id INTEGER, -- Reference to dataset used for evaluation
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_config JSONB,
    additional_metrics JSONB
);

-- Model training jobs
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(64) UNIQUE NOT NULL,
    model_id VARCHAR(64) REFERENCES model_registry(model_id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'QUEUED', -- QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    dataset_path TEXT,
    target_column VARCHAR(100),
    training_config JSONB,
    hyperparameter_config JSONB,
    resource_usage JSONB, -- CPU, memory, GPU usage
    logs TEXT,
    error_message TEXT,
    created_by INTEGER,
    priority INTEGER DEFAULT 5 -- 1 (highest) to 10 (lowest)
);

-- Model deployments
CREATE TABLE IF NOT EXISTS model_deployments (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(64) UNIQUE NOT NULL,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    environment VARCHAR(50) NOT NULL, -- 'development', 'staging', 'production'
    endpoint_url TEXT,
    status VARCHAR(20) DEFAULT 'DEPLOYING', -- DEPLOYING, ACTIVE, INACTIVE, FAILED
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_by INTEGER,
    configuration JSONB,
    resource_allocation JSONB, -- CPU, memory, replicas
    health_check_url TEXT,
    last_health_check TIMESTAMP,
    health_status VARCHAR(20), -- HEALTHY, UNHEALTHY, UNKNOWN
    auto_scaling_config JSONB
);

-- Model predictions log for monitoring
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(64) UNIQUE NOT NULL,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    deployment_id VARCHAR(64) REFERENCES model_deployments(deployment_id),
    input_features JSONB NOT NULL,
    prediction_result JSONB NOT NULL,
    confidence_score FLOAT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    client_ip INET,
    user_id INTEGER,
    batch_id VARCHAR(64), -- For batch predictions
    feedback_score FLOAT, -- User feedback on prediction quality
    actual_outcome JSONB, -- Actual outcome for model evaluation
    drift_score FLOAT -- Data drift score for this prediction
);

-- Model experiments for A/B testing
CREATE TABLE IF NOT EXISTS model_experiments (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'DRAFT', -- DRAFT, RUNNING, COMPLETED, PAUSED
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    traffic_split JSONB, -- Percentage allocation to each model
    success_metrics TEXT[], -- Array of metrics to track
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    configuration JSONB
);

-- Model experiment variants
CREATE TABLE IF NOT EXISTS experiment_variants (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(64) NOT NULL REFERENCES model_experiments(experiment_id) ON DELETE CASCADE,
    variant_name VARCHAR(100) NOT NULL,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    traffic_percentage FLOAT NOT NULL,
    is_control BOOLEAN DEFAULT FALSE,
    configuration JSONB
);

-- Feature store for managing model features
CREATE TABLE IF NOT EXISTS feature_groups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    features TEXT[] NOT NULL, -- Array of feature names
    feature_types JSONB, -- Feature name to type mapping
    source_table VARCHAR(255),
    source_query TEXT,
    refresh_frequency VARCHAR(50), -- 'real-time', 'hourly', 'daily', 'weekly'
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1
);

-- Feature values for real-time serving
CREATE TABLE IF NOT EXISTS feature_values (
    id SERIAL PRIMARY KEY,
    feature_group_id INTEGER NOT NULL REFERENCES feature_groups(id) ON DELETE CASCADE,
    entity_id VARCHAR(255) NOT NULL, -- Customer ID, product ID, etc.
    feature_values JSONB NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Model monitoring alerts
CREATE TABLE IF NOT EXISTS model_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(64) UNIQUE NOT NULL,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- 'performance_degradation', 'data_drift', 'concept_drift', 'error_rate'
    severity VARCHAR(20) NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message TEXT NOT NULL,
    threshold_value FLOAT,
    actual_value FLOAT,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by INTEGER,
    status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE
    metadata JSONB,
    notification_sent BOOLEAN DEFAULT FALSE
);

-- Model approval workflow
CREATE TABLE IF NOT EXISTS model_approvals (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    approval_stage VARCHAR(50) NOT NULL, -- 'data_validation', 'model_validation', 'security_review', 'business_approval'
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, APPROVED, REJECTED
    approver_id INTEGER,
    approved_at TIMESTAMP,
    comments TEXT,
    requirements_met JSONB, -- Checklist of requirements
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model lineage for tracking data and model dependencies
CREATE TABLE IF NOT EXISTS model_lineage (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
    parent_model_id VARCHAR(64) REFERENCES model_registry(model_id),
    dataset_id INTEGER, -- Reference to training dataset
    feature_group_ids INTEGER[], -- Array of feature group IDs used
    dependency_type VARCHAR(50), -- 'training_data', 'feature_engineering', 'model_ensemble', 'transfer_learning'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_model_registry_model_id ON model_registry(model_id);
CREATE INDEX IF NOT EXISTS idx_model_registry_type ON model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_created_at ON model_registry(created_at);
CREATE INDEX IF NOT EXISTS idx_model_performance_model_id ON model_performance(model_id);
CREATE INDEX IF NOT EXISTS idx_model_performance_metric_name ON model_performance(metric_name);
CREATE INDEX IF NOT EXISTS idx_training_jobs_job_id ON training_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_model_deployments_model_id ON model_deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_model_deployments_environment ON model_deployments(environment);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_id ON model_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_time ON model_predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_feature_groups_name ON feature_groups(name);
CREATE INDEX IF NOT EXISTS idx_feature_values_entity_id ON feature_values(entity_id);
CREATE INDEX IF NOT EXISTS idx_model_alerts_model_id ON model_alerts(model_id);
CREATE INDEX IF NOT EXISTS idx_model_alerts_status ON model_alerts(status);

-- Functions for model management
CREATE OR REPLACE FUNCTION update_model_status(
    p_model_id VARCHAR(64),
    p_status VARCHAR(20)
)
RETURNS VOID AS $$
BEGIN
    UPDATE model_registry 
    SET status = p_status 
    WHERE model_id = p_model_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate model performance metrics
CREATE OR REPLACE FUNCTION calculate_model_metrics(
    p_model_id VARCHAR(64),
    p_start_date TIMESTAMP DEFAULT NULL,
    p_end_date TIMESTAMP DEFAULT NULL
)
RETURNS TABLE(
    metric_name VARCHAR(100),
    avg_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    count_values BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mp.metric_name,
        AVG(mp.metric_value) as avg_value,
        MIN(mp.metric_value) as min_value,
        MAX(mp.metric_value) as max_value,
        COUNT(*) as count_values
    FROM model_performance mp
    WHERE mp.model_id = p_model_id
        AND (p_start_date IS NULL OR mp.evaluation_date >= p_start_date)
        AND (p_end_date IS NULL OR mp.evaluation_date <= p_end_date)
    GROUP BY mp.metric_name;
END;
$$ LANGUAGE plpgsql;

-- Function to detect model drift
CREATE OR REPLACE FUNCTION detect_model_drift(
    p_model_id VARCHAR(64),
    p_threshold FLOAT DEFAULT 0.1
)
RETURNS BOOLEAN AS $$
DECLARE
    recent_performance FLOAT;
    baseline_performance FLOAT;
    drift_detected BOOLEAN := FALSE;
BEGIN
    -- Get recent performance (last 7 days)
    SELECT AVG(metric_value) INTO recent_performance
    FROM model_performance
    WHERE model_id = p_model_id 
        AND metric_name = 'accuracy'
        AND evaluation_date >= CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    -- Get baseline performance (first 30 days after deployment)
    SELECT AVG(metric_value) INTO baseline_performance
    FROM model_performance mp
    JOIN model_deployments md ON mp.model_id = md.model_id
    WHERE mp.model_id = p_model_id 
        AND mp.metric_name = 'accuracy'
        AND mp.evaluation_date BETWEEN md.deployed_at AND md.deployed_at + INTERVAL '30 days';
    
    -- Check if drift exceeds threshold
    IF recent_performance IS NOT NULL AND baseline_performance IS NOT NULL THEN
        IF ABS(recent_performance - baseline_performance) > p_threshold THEN
            drift_detected := TRUE;
            
            -- Create alert
            INSERT INTO model_alerts (
                alert_id, model_id, alert_type, severity, message, 
                threshold_value, actual_value
            ) VALUES (
                'drift_' || p_model_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP),
                p_model_id,
                'performance_degradation',
                'HIGH',
                'Model performance has degraded beyond acceptable threshold',
                p_threshold,
                ABS(recent_performance - baseline_performance)
            );
        END IF;
    END IF;
    
    RETURN drift_detected;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old prediction logs
CREATE OR REPLACE FUNCTION cleanup_old_predictions(
    p_retention_days INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM model_predictions 
    WHERE prediction_time < CURRENT_TIMESTAMP - (p_retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
