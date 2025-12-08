"""
Load Testing with Locust for Abidjan Smart City API
Simulates concurrent users accessing the API.
"""
from locust import HttpUser, task, between

class SmartCityUser(HttpUser):
    """Simulated API user for load testing."""
    
    wait_time = between(0.5, 2)
    host = "http://localhost:8000"
    
    def on_start(self):
        """Login on user start."""
        response = self.client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(10)
    def get_all_segments(self):
        """Get all traffic segments - most common operation."""
        self.client.get("/api/v1/traffic/segments", headers=self.headers)
    
    @task(5)
    def get_single_segment(self):
        """Get single segment status."""
        self.client.get("/api/v1/traffic/segments/SEG001", headers=self.headers)
    
    @task(3)
    def get_predictions(self):
        """Get traffic predictions."""
        self.client.get("/api/v1/predictions/SEG001?horizon_minutes=30", headers=self.headers)
    
    @task(2)
    def get_anomalies(self):
        """Get active anomalies."""
        self.client.get("/api/v1/anomalies", headers=self.headers)
    
    @task(2)
    def get_incidents(self):
        """Get active incidents."""
        self.client.get("/api/v1/incidents", headers=self.headers)
    
    @task(1)
    def get_stats(self):
        """Get statistics summary."""
        self.client.get("/api/v1/stats/summary", headers=self.headers)
    
    @task(1)
    def get_heatmap(self):
        """Get traffic heatmap data."""
        self.client.get("/api/v1/traffic/heatmap", headers=self.headers)

# Run with: locust -f tests/load_test.py --headless -u 100 -r 10 -t 60s
