"""
Database Schema Initialization for PostgreSQL/PostGIS
Creates all tables, indexes and partitions for the Smart City platform.
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
sys.path.append('..')
from config.settings import postgres_config

SCHEMA_SQL = """
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ============================================
-- REFERENCE DATA TABLES
-- ============================================

-- Road segments table
CREATE TABLE IF NOT EXISTS road_segments (
    segment_id VARCHAR(20) PRIMARY KEY,
    segment_name VARCHAR(100) NOT NULL,
    start_point GEOMETRY(Point, 4326),
    end_point GEOMETRY(Point, 4326),
    segment_line GEOMETRY(LineString, 4326),
    max_speed_limit INTEGER,
    lanes INTEGER,
    road_type VARCHAR(50),
    is_bridge BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weather stations
CREATE TABLE IF NOT EXISTS weather_stations (
    station_id VARCHAR(20) PRIMARY KEY,
    station_name VARCHAR(100) NOT NULL,
    location GEOMETRY(Point, 4326),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- REAL-TIME DATA TABLES
-- ============================================

-- Traffic segment statistics (updated every minute)
CREATE TABLE IF NOT EXISTS traffic_segment_stats (
    id BIGSERIAL,
    segment_id VARCHAR(20) REFERENCES road_segments(segment_id),
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    avg_speed DECIMAL(6,2),
    max_speed DECIMAL(6,2),
    min_speed DECIMAL(6,2),
    speed_stddev DECIMAL(6,2),
    vehicle_count INTEGER,
    stopped_vehicles INTEGER,
    gbaka_count INTEGER,
    bus_count INTEGER,
    taxi_count INTEGER,
    congestion_level VARCHAR(20),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, window_start)
) PARTITION BY RANGE (window_start);

-- Create partitions for traffic stats (current month + 2 future months)
CREATE TABLE IF NOT EXISTS traffic_segment_stats_default PARTITION OF traffic_segment_stats DEFAULT;

-- Traffic anomalies
CREATE TABLE IF NOT EXISTS traffic_anomalies (
    id BIGSERIAL PRIMARY KEY,
    segment_id VARCHAR(20) REFERENCES road_segments(segment_id),
    segment_name VARCHAR(100),
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    avg_speed DECIMAL(6,2),
    vehicle_count INTEGER,
    stopped_count INTEGER,
    stopped_ratio DECIMAL(4,3),
    speed_variance DECIMAL(6,2),
    anomaly_type VARCHAR(50),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    is_resolved BOOLEAN DEFAULT FALSE
);

-- Incidents table
CREATE TABLE IF NOT EXISTS incidents (
    incident_id VARCHAR(30) PRIMARY KEY,
    incident_type VARCHAR(50) NOT NULL,
    location GEOMETRY(Point, 4326),
    location_name VARCHAR(100),
    severity INTEGER CHECK (severity BETWEEN 1 AND 5),
    description TEXT,
    estimated_duration_minutes INTEGER,
    lanes_affected INTEGER,
    is_resolved BOOLEAN DEFAULT FALSE,
    reported_by VARCHAR(50),
    vehicles_involved INTEGER,
    injuries INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Weather data
CREATE TABLE IF NOT EXISTS weather_data (
    id BIGSERIAL,
    station_id VARCHAR(20) REFERENCES weather_stations(station_id),
    recorded_at TIMESTAMP NOT NULL,
    temperature DECIMAL(4,1),
    humidity DECIMAL(4,1),
    precipitation DECIMAL(5,2),
    wind_speed DECIMAL(5,1),
    wind_direction DECIMAL(5,1),
    visibility DECIMAL(4,1),
    condition VARCHAR(30),
    pressure DECIMAL(6,1),
    uv_index DECIMAL(3,1),
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

CREATE TABLE IF NOT EXISTS weather_data_default PARTITION OF weather_data DEFAULT;

-- ============================================
-- PREDICTION TABLES
-- ============================================

-- Traffic predictions
CREATE TABLE IF NOT EXISTS traffic_predictions (
    id BIGSERIAL PRIMARY KEY,
    segment_id VARCHAR(20) REFERENCES road_segments(segment_id),
    prediction_time TIMESTAMP NOT NULL,
    target_time TIMESTAMP NOT NULL,
    predicted_speed DECIMAL(6,2),
    predicted_congestion_level VARCHAR(20),
    confidence_score DECIMAL(4,3),
    model_version VARCHAR(20),
    features_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- USER MANAGEMENT TABLES
-- ============================================

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    api_key VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit INTEGER DEFAULT 1000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_traffic_stats_segment ON traffic_segment_stats(segment_id);
CREATE INDEX IF NOT EXISTS idx_traffic_stats_window ON traffic_segment_stats(window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_traffic_stats_congestion ON traffic_segment_stats(congestion_level);

CREATE INDEX IF NOT EXISTS idx_anomalies_segment ON traffic_anomalies(segment_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_detected ON traffic_anomalies(detected_at);
CREATE INDEX IF NOT EXISTS idx_anomalies_unresolved ON traffic_anomalies(is_resolved) WHERE is_resolved = FALSE;

CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_incidents_unresolved ON incidents(is_resolved) WHERE is_resolved = FALSE;

CREATE INDEX IF NOT EXISTS idx_weather_station ON weather_data(station_id);
CREATE INDEX IF NOT EXISTS idx_weather_recorded ON weather_data(recorded_at);

CREATE INDEX IF NOT EXISTS idx_predictions_segment ON traffic_predictions(segment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON traffic_predictions(target_time);

-- Spatial index on road segments
CREATE INDEX IF NOT EXISTS idx_segments_line ON road_segments USING GIST(segment_line);
"""

SEED_DATA_SQL = """
-- Insert road segments
INSERT INTO road_segments (segment_id, segment_name, start_point, end_point, max_speed_limit, lanes, road_type, is_bridge)
VALUES 
    ('SEG001', 'Pont HKB', ST_SetSRID(ST_MakePoint(-4.0167, 5.3167), 4326), ST_SetSRID(ST_MakePoint(-4.0000, 5.2833), 4326), 60, 4, 'bridge', TRUE),
    ('SEG002', 'Boulevard VGE', ST_SetSRID(ST_MakePoint(-3.9833, 5.3500), 4326), ST_SetSRID(ST_MakePoint(-4.0167, 5.3167), 4326), 80, 6, 'boulevard', FALSE),
    ('SEG003', 'Autoroute du Nord', ST_SetSRID(ST_MakePoint(-4.0333, 5.3500), 4326), ST_SetSRID(ST_MakePoint(-4.0333, 5.4167), 4326), 100, 4, 'highway', FALSE),
    ('SEG004', 'Rue Lepic', ST_SetSRID(ST_MakePoint(-4.0167, 5.3167), 4326), ST_SetSRID(ST_MakePoint(-4.0333, 5.3500), 4326), 50, 2, 'urban', FALSE),
    ('SEG005', 'Boulevard Nangui Abrogoua', ST_SetSRID(ST_MakePoint(-4.0833, 5.3333), 4326), ST_SetSRID(ST_MakePoint(-4.0333, 5.3500), 4326), 70, 4, 'boulevard', FALSE),
    ('SEG006', 'Pont Général de Gaulle', ST_SetSRID(ST_MakePoint(-4.0000, 5.2833), 4326), ST_SetSRID(ST_MakePoint(-3.9667, 5.2833), 4326), 50, 4, 'bridge', TRUE),
    ('SEG007', 'Boulevard de Marseille', ST_SetSRID(ST_MakePoint(-3.9333, 5.2833), 4326), ST_SetSRID(ST_MakePoint(-3.9667, 5.2833), 4326), 60, 4, 'boulevard', FALSE),
    ('SEG008', 'Voie Express Yopougon', ST_SetSRID(ST_MakePoint(-4.0833, 5.3333), 4326), ST_SetSRID(ST_MakePoint(-4.0167, 5.3167), 4326), 80, 4, 'expressway', FALSE)
ON CONFLICT (segment_id) DO NOTHING;

-- Insert weather stations
INSERT INTO weather_stations (station_id, station_name, location)
VALUES 
    ('WS_001', 'Plateau', ST_SetSRID(ST_MakePoint(-4.0167, 5.3167), 4326)),
    ('WS_002', 'Cocody', ST_SetSRID(ST_MakePoint(-3.9833, 5.3500), 4326)),
    ('WS_003', 'Yopougon', ST_SetSRID(ST_MakePoint(-4.0833, 5.3333), 4326)),
    ('WS_004', 'Abobo', ST_SetSRID(ST_MakePoint(-4.0333, 5.4167), 4326)),
    ('WS_005', 'Port-Bouët', ST_SetSRID(ST_MakePoint(-3.9333, 5.2500), 4326))
ON CONFLICT (station_id) DO NOTHING;

-- Insert default admin user (password: admin123 - hashed with bcrypt)
INSERT INTO users (username, email, password_hash, role)
VALUES ('admin', 'admin@smartcity.ci', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKxcqgeZ4A8qmYC', 'admin')
ON CONFLICT (username) DO NOTHING;
"""

def init_database():
    """Initialize the database schema."""
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
            print("Creating schema...")
            cur.execute(SCHEMA_SQL)
            print("Inserting seed data...")
            cur.execute(SEED_DATA_SQL)
            print("Database initialization complete!")
            
        conn.close()
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_database()
