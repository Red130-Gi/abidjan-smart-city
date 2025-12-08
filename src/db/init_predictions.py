"""
Database Schema for Predictions and Route Optimization
Creates tables for storing ML predictions, route requests, and optimized routes.
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
sys.path.append('.')
from config.settings import postgres_config

PREDICTIONS_SCHEMA = """
-- ============================================
-- TRAFFIC PREDICTIONS TABLES
-- ============================================

-- Store ML predictions for each segment
CREATE TABLE IF NOT EXISTS traffic_predictions (
    id BIGSERIAL PRIMARY KEY,
    segment_id VARCHAR(20) NOT NULL,
    prediction_time TIMESTAMP NOT NULL,  -- When prediction was made
    target_time TIMESTAMP NOT NULL,       -- Time being predicted
    horizon_minutes INTEGER NOT NULL,     -- 5, 15, 30, 60 minutes ahead
    predicted_speed DECIMAL(6,2),
    predicted_congestion VARCHAR(20),     -- light, moderate, heavy, severe
    confidence_score DECIMAL(4,3),
    model_type VARCHAR(20),               -- xgboost, lstm, ensemble
    features_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predicted anomalies (future incidents/congestion)
CREATE TABLE IF NOT EXISTS predicted_anomalies (
    id BIGSERIAL PRIMARY KEY,
    segment_id VARCHAR(20) NOT NULL,
    prediction_time TIMESTAMP NOT NULL,
    expected_start TIMESTAMP NOT NULL,
    expected_end TIMESTAMP,
    anomaly_type VARCHAR(50),             -- congestion_spike, accident_risk, event_traffic
    severity INTEGER CHECK (severity BETWEEN 1 AND 5),
    probability DECIMAL(4,3),
    description TEXT,
    recommended_action TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- ROUTE OPTIMIZATION TABLES
-- ============================================

-- Route calculation requests
CREATE TABLE IF NOT EXISTS route_requests (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(50) UNIQUE NOT NULL,
    origin_segment VARCHAR(20) NOT NULL,
    destination_segment VARCHAR(20) NOT NULL,
    origin_name VARCHAR(100),
    destination_name VARCHAR(100),
    departure_time TIMESTAMP,
    preferences JSONB,                    -- avoid_tolls, prefer_highways, etc.
    status VARCHAR(20) DEFAULT 'pending', -- pending, completed, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Calculated routes (both normal and optimized)
CREATE TABLE IF NOT EXISTS optimized_routes (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(50) REFERENCES route_requests(request_id),
    route_type VARCHAR(20) NOT NULL,      -- normal, optimized, alternative
    total_distance_km DECIMAL(8,2),
    estimated_duration_minutes INTEGER,
    estimated_duration_optimized INTEGER,
    total_congestion_score DECIMAL(4,2),
    fuel_consumption_liters DECIMAL(5,2),
    co2_emissions_kg DECIMAL(6,2),
    time_saved_minutes INTEGER,
    route_geometry JSONB,                 -- GeoJSON LineString
    waypoints JSONB,                      -- Array of coordinates
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual segments in a route
CREATE TABLE IF NOT EXISTS route_segments (
    id BIGSERIAL PRIMARY KEY,
    route_id BIGINT REFERENCES optimized_routes(id),
    segment_order INTEGER NOT NULL,
    segment_id VARCHAR(20),
    segment_name VARCHAR(100),
    distance_km DECIMAL(6,2),
    duration_minutes INTEGER,
    current_speed DECIMAL(6,2),
    predicted_speed DECIMAL(6,2),
    congestion_level VARCHAR(20),
    geometry JSONB
);

-- ============================================
-- ML MODEL TRACKING
-- ============================================

-- Track model versions and performance
CREATE TABLE IF NOT EXISTS ml_model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(30) NOT NULL,      -- xgboost, lstm, random_forest
    version VARCHAR(20) NOT NULL,
    target_variable VARCHAR(50),          -- speed, congestion_level
    training_date TIMESTAMP,
    metrics JSONB,                        -- mae, rmse, accuracy
    parameters JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    model_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_predictions_segment_time ON traffic_predictions(segment_id, target_time);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON traffic_predictions(horizon_minutes);
CREATE INDEX IF NOT EXISTS idx_anomalies_segment ON predicted_anomalies(segment_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_expected ON predicted_anomalies(expected_start);
CREATE INDEX IF NOT EXISTS idx_routes_request ON optimized_routes(request_id);
CREATE INDEX IF NOT EXISTS idx_route_segments_route ON route_segments(route_id);
"""

def init_predictions_schema():
    """Initialize prediction and routing tables."""
    try:
        conn = psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            print("Creating predictions and routing schema...")
            cur.execute(PREDICTIONS_SCHEMA)
            print("Schema created successfully!")
            
        conn.close()
        
    except Exception as e:
        print(f"Error creating schema: {e}")
        raise

if __name__ == "__main__":
    init_predictions_schema()
