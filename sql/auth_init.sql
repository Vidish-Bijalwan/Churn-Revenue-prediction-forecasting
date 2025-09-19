-- Authentication Database Initialization
-- Create tables for user management, sessions, and audit logs

-- Users table with enhanced security features
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    roles TEXT, -- JSON array of roles
    permissions TEXT, -- JSON array of permissions
    profile_data JSONB,
    preferences JSONB,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(32)
);

-- Sessions table for tracking user sessions
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    device_fingerprint VARCHAR(255),
    location_country VARCHAR(2),
    location_city VARCHAR(100)
);

-- Audit logs for security and compliance
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100),
    resource_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    risk_score FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'SUCCESS'
);

-- Password reset tokens
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    is_used BOOLEAN DEFAULT FALSE
);

-- API keys for service-to-service authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    key_id VARCHAR(32) UNIQUE NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    permissions TEXT, -- JSON array of permissions
    rate_limit INTEGER DEFAULT 1000,
    usage_count INTEGER DEFAULT 0
);

-- OAuth applications for third-party integrations
CREATE TABLE IF NOT EXISTS oauth_applications (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(255) UNIQUE NOT NULL,
    client_secret_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    redirect_uris TEXT[], -- Array of allowed redirect URIs
    scopes TEXT[], -- Array of allowed scopes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE
);

-- OAuth authorization codes
CREATE TABLE IF NOT EXISTS oauth_authorization_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(255) UNIQUE NOT NULL,
    client_id VARCHAR(255) NOT NULL REFERENCES oauth_applications(client_id),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    redirect_uri TEXT NOT NULL,
    scopes TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_used BOOLEAN DEFAULT FALSE
);

-- OAuth access tokens
CREATE TABLE IF NOT EXISTS oauth_access_tokens (
    id SERIAL PRIMARY KEY,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    client_id VARCHAR(255) NOT NULL REFERENCES oauth_applications(client_id),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    scopes TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_revoked BOOLEAN DEFAULT FALSE
);

-- Roles and permissions system
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions TEXT[], -- Array of permission strings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_by INTEGER REFERENCES users(id),
    PRIMARY KEY (user_id, role_id)
);

-- Security events for monitoring
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    severity VARCHAR(20) DEFAULT 'INFO', -- INFO, WARNING, CRITICAL
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by INTEGER REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id);
CREATE INDEX IF NOT EXISTS idx_oauth_codes_code ON oauth_authorization_codes(code);
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_token_hash ON oauth_access_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);

-- Insert default roles
INSERT INTO roles (name, description, permissions) VALUES
('admin', 'System Administrator', ARRAY['*']),
('data_scientist', 'Data Scientist', ARRAY['read_data', 'write_data', 'train_models', 'view_analytics']),
('analyst', 'Business Analyst', ARRAY['read_data', 'view_analytics', 'export_reports']),
('viewer', 'Read-only User', ARRAY['read_data', 'view_analytics']),
('api_user', 'API Access User', ARRAY['api_access', 'read_data'])
ON CONFLICT (name) DO NOTHING;

-- Insert default admin user (password: admin123 - change in production!)
INSERT INTO users (email, username, hashed_password, full_name, is_superuser, roles, permissions) VALUES
('admin@foretel.ai', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VJWZrxnlO', 'System Administrator', TRUE, '["admin"]', '["*"]')
ON CONFLICT (username) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean expired sessions
CREATE OR REPLACE FUNCTION clean_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to log security events
CREATE OR REPLACE FUNCTION log_security_event(
    p_event_type VARCHAR(50),
    p_user_id INTEGER DEFAULT NULL,
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_details JSONB DEFAULT NULL,
    p_severity VARCHAR(20) DEFAULT 'INFO'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO security_events (event_type, user_id, ip_address, user_agent, details, severity)
    VALUES (p_event_type, p_user_id, p_ip_address, p_user_agent, p_details, p_severity);
END;
$$ LANGUAGE plpgsql;
