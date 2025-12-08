"""
Unit Tests for Abidjan Smart City Platform
Tests for API, producers, and ML components.
"""
import pytest
from datetime import datetime
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json

# FastAPI Testing
from fastapi.testclient import TestClient
from httpx import AsyncClient

# ============================================
# API TESTS
# ============================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)
    
    @pytest.fixture
    def auth_token(self, client):
        """Get authentication token."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        assert response.status_code == 200
        return response.json()["access_token"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_login_success(self, client):
        """Test successful login."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self, client):
        """Test failed login."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
    
    def test_get_segments_requires_auth(self, client):
        """Test that segments endpoint requires authentication."""
        response = client.get("/api/v1/traffic/segments")
        assert response.status_code == 403
    
    def test_get_segments_with_auth(self, client, auth_token):
        """Test getting segments with valid token."""
        response = client.get(
            "/api/v1/traffic/segments",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_single_segment(self, client, auth_token):
        """Test getting a single segment."""
        response = client.get(
            "/api/v1/traffic/segments/SEG001",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["segment_id"] == "SEG001"
        assert "avg_speed" in data
    
    def test_get_nonexistent_segment(self, client, auth_token):
        """Test getting a non-existent segment."""
        response = client.get(
            "/api/v1/traffic/segments/INVALID",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404
    
    def test_get_predictions(self, client, auth_token):
        """Test predictions endpoint."""
        response = client.get(
            "/api/v1/predictions/SEG001?horizon_minutes=30",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all("predicted_speed" in p for p in data)
    
    def test_get_anomalies(self, client, auth_token):
        """Test anomalies endpoint."""
        response = client.get(
            "/api/v1/anomalies",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_stats_summary(self, client, auth_token):
        """Test statistics summary endpoint."""
        response = client.get(
            "/api/v1/stats/summary",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_segments" in data
        assert "average_city_speed" in data

# ============================================
# PRODUCER TESTS
# ============================================

class TestTrafficProducer:
    """Tests for traffic data producer."""
    
    def test_traffic_data_point_structure(self):
        """Test TrafficDataPoint dataclass."""
        from src.producers.traffic_producer import TrafficDataPoint
        
        data_point = TrafficDataPoint(
            vehicle_id="VEH_00001",
            vehicle_type="gbaka",
            timestamp="2024-01-01T12:00:00Z",
            latitude=5.3167,
            longitude=-4.0167,
            speed=45.5,
            heading=90.0,
            segment_id="SEG001",
            segment_name="Pont HKB",
            acceleration=0.5,
            is_stopped=False,
            occupancy=15,
            fuel_level=75.0
        )
        
        assert data_point.vehicle_id == "VEH_00001"
        assert data_point.vehicle_type == "gbaka"
        assert data_point.speed == 45.5
    
    def test_vehicle_initialization(self):
        """Test vehicle fleet initialization."""
        from src.producers.traffic_producer import TrafficDataProducer
        
        producer = TrafficDataProducer(num_vehicles=10)
        assert len(producer.vehicles) == 10
        
        for vehicle in producer.vehicles:
            assert "id" in vehicle
            assert "type" in vehicle
            assert "latitude" in vehicle
            assert "longitude" in vehicle

# ============================================
# ML TESTS
# ============================================

class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample traffic data."""
        np.random.seed(42)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'segment_id': ['SEG001'] * 100,
            'avg_speed': np.random.uniform(20, 80, 100),
            'vehicle_count': np.random.randint(10, 100, 100),
            'stopped_vehicles': np.random.randint(0, 20, 100),
            'congestion_level': np.random.choice(['light', 'moderate', 'heavy'], 100),
            'gbaka_count': np.random.randint(0, 30, 100),
            'bus_count': np.random.randint(0, 10, 100),
            'taxi_count': np.random.randint(0, 20, 100),
        })
    
    def test_temporal_features(self, sample_data):
        """Test temporal feature extraction."""
        from src.ml.feature_engineering import TrafficFeatureEngineer
        
        engineer = TrafficFeatureEngineer()
        df = engineer.extract_temporal_features(sample_data)
        
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'is_weekend' in df.columns
        assert 'is_rush_hour' in df.columns
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
    
    def test_full_feature_set(self, sample_data):
        """Test full feature set creation."""
        from src.ml.feature_engineering import TrafficFeatureEngineer
        
        engineer = TrafficFeatureEngineer()
        featured_df, feature_cols = engineer.create_full_feature_set(sample_data)
        
        assert len(feature_cols) >= 40  # Should have 45+ features
        assert all(col in featured_df.columns for col in feature_cols)

class TestAnomalyDetection:
    """Tests for anomaly detection."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data."""
        np.random.seed(42)
        n_samples = 500
        return pd.DataFrame({
            'segment_id': np.random.choice(['SEG001', 'SEG002'], n_samples),
            'avg_speed': np.random.normal(40, 10, n_samples),
            'vehicle_count': np.random.poisson(50, n_samples),
            'stopped_vehicles': np.random.poisson(5, n_samples),
            'speed_stddev': np.random.uniform(2, 8, n_samples),
            'gbaka_count': np.random.poisson(10, n_samples),
        })
    
    def test_statistical_detector_baseline(self, sample_historical_data):
        """Test statistical detector baseline computation."""
        from src.ml.anomaly_detection import StatisticalAnomalyDetector
        
        detector = StatisticalAnomalyDetector()
        sample_historical_data['stopped_ratio'] = (
            sample_historical_data['stopped_vehicles'] / 
            sample_historical_data['vehicle_count']
        )
        detector.compute_baseline(sample_historical_data)
        
        assert 'SEG001' in detector.baseline_stats
        assert 'SEG002' in detector.baseline_stats
        assert 'mean' in detector.baseline_stats['SEG001']
    
    def test_zscore_anomaly_detection(self, sample_historical_data):
        """Test Z-score anomaly detection."""
        from src.ml.anomaly_detection import StatisticalAnomalyDetector
        
        detector = StatisticalAnomalyDetector(zscore_threshold=2.0)
        detector.compute_baseline(sample_historical_data)
        
        # Test normal value
        is_anomaly, zscore, direction = detector.detect_zscore_anomaly('SEG001', 40)
        assert not is_anomaly
        
        # Test extreme value
        is_anomaly, zscore, direction = detector.detect_zscore_anomaly('SEG001', 5)
        assert is_anomaly
        assert direction == "low"

# ============================================
# INTEGRATION TESTS
# ============================================

class TestIntegration:
    """Integration tests for end-to-end flows."""
    
    @pytest.mark.integration
    def test_producer_to_kafka_flow(self):
        """Test data flow from producer to Kafka."""
        # This would require running Kafka
        # Marked as integration test to skip in CI
        pass
    
    @pytest.mark.integration
    def test_spark_streaming_processing(self):
        """Test Spark streaming processing."""
        # This would require running Spark
        # Marked as integration test to skip in CI
        pass

# ============================================
# PERFORMANCE TESTS
# ============================================

class TestPerformance:
    """Performance and load tests."""
    
    def test_feature_engineering_performance(self):
        """Test feature engineering performance."""
        from src.ml.feature_engineering import TrafficFeatureEngineer
        import time
        
        np.random.seed(42)
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1min'),
            'segment_id': ['SEG001'] * 10000,
            'avg_speed': np.random.uniform(20, 80, 10000),
            'vehicle_count': np.random.randint(10, 100, 10000),
            'stopped_vehicles': np.random.randint(0, 20, 10000),
            'congestion_level': np.random.choice(['light', 'moderate', 'heavy'], 10000),
            'gbaka_count': np.random.randint(0, 30, 10000),
            'bus_count': np.random.randint(0, 10, 10000),
            'taxi_count': np.random.randint(0, 20, 10000),
        })
        
        engineer = TrafficFeatureEngineer()
        
        start = time.time()
        featured_df, _ = engineer.create_full_feature_set(large_data)
        elapsed = time.time() - start
        
        # Should process 10k records in less than 5 seconds
        assert elapsed < 5.0
        assert len(featured_df) == 10000

# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
