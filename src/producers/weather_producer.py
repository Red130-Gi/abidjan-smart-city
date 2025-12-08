"""
Weather Data Producer for Abidjan Smart City Platform
Simulates weather data that affects traffic patterns.
"""
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
import sys
sys.path.append('..')
from config.settings import kafka_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Weather stations across Abidjan
WEATHER_STATIONS = [
    {"id": "WS_001", "name": "Plateau", "lat": 5.3167, "lon": -4.0167},
    {"id": "WS_002", "name": "Cocody", "lat": 5.3500, "lon": -3.9833},
    {"id": "WS_003", "name": "Yopougon", "lat": 5.3333, "lon": -4.0833},
    {"id": "WS_004", "name": "Abobo", "lat": 5.4167, "lon": -4.0333},
    {"id": "WS_005", "name": "Port-Bouët", "lat": 5.2500, "lon": -3.9333},
]

WEATHER_CONDITIONS = ["sunny", "cloudy", "light_rain", "heavy_rain", "thunderstorm", "foggy"]

@dataclass
class WeatherDataPoint:
    """Represents weather data from a station."""
    station_id: str
    station_name: str
    timestamp: str
    latitude: float
    longitude: float
    temperature: float  # Celsius
    humidity: float  # Percentage
    precipitation: float  # mm/h
    wind_speed: float  # km/h
    wind_direction: float  # degrees
    visibility: float  # km
    condition: str
    pressure: float  # hPa
    uv_index: float

class WeatherDataProducer:
    """Produces simulated weather data for Kafka ingestion."""
    
    def __init__(self):
        self.stations = WEATHER_STATIONS
        self.base_conditions = self._initialize_base_conditions()
        self.producer = None
    
    def _initialize_base_conditions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize base weather conditions for each station."""
        conditions = {}
        base_temp = random.uniform(25, 32)  # Tropical Abidjan temperature
        base_humidity = random.uniform(60, 85)
        
        for station in self.stations:
            conditions[station["id"]] = {
                "temperature": base_temp + random.uniform(-2, 2),
                "humidity": base_humidity + random.uniform(-5, 5),
                "precipitation": 0,
                "condition": random.choice(["sunny", "cloudy"]),
            }
        return conditions
    
    def connect_kafka(self) -> None:
        """Connect to Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            logger.info(f"Connected to Kafka at {kafka_config.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def _evolve_weather(self, station_id: str) -> None:
        """Evolve weather conditions gradually."""
        base = self.base_conditions[station_id]
        
        # Temperature varies slightly
        base["temperature"] += random.uniform(-0.5, 0.5)
        base["temperature"] = max(20, min(38, base["temperature"]))
        
        # Humidity changes
        base["humidity"] += random.uniform(-2, 2)
        base["humidity"] = max(40, min(100, base["humidity"]))
        
        # Weather condition transitions
        if random.random() < 0.05:  # 5% chance to change condition
            current_idx = WEATHER_CONDITIONS.index(base["condition"])
            # Transition to adjacent conditions
            if current_idx == 0:
                new_idx = random.choice([0, 1])
            elif current_idx == len(WEATHER_CONDITIONS) - 1:
                new_idx = random.choice([current_idx - 1, current_idx])
            else:
                new_idx = random.choice([current_idx - 1, current_idx, current_idx + 1])
            base["condition"] = WEATHER_CONDITIONS[new_idx]
        
        # Precipitation based on condition
        if base["condition"] in ["light_rain", "heavy_rain", "thunderstorm"]:
            if base["condition"] == "light_rain":
                base["precipitation"] = random.uniform(0.5, 5)
            elif base["condition"] == "heavy_rain":
                base["precipitation"] = random.uniform(5, 20)
            else:  # thunderstorm
                base["precipitation"] = random.uniform(15, 50)
        else:
            base["precipitation"] = 0
    
    def generate_data_point(self, station: Dict[str, Any]) -> WeatherDataPoint:
        """Generate a weather data point for a station."""
        self._evolve_weather(station["id"])
        base = self.base_conditions[station["id"]]
        
        # Calculate visibility based on condition
        if base["condition"] == "foggy":
            visibility = random.uniform(0.1, 1)
        elif base["condition"] in ["heavy_rain", "thunderstorm"]:
            visibility = random.uniform(1, 5)
        elif base["condition"] == "light_rain":
            visibility = random.uniform(3, 8)
        else:
            visibility = random.uniform(8, 15)
        
        return WeatherDataPoint(
            station_id=station["id"],
            station_name=station["name"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            latitude=station["lat"],
            longitude=station["lon"],
            temperature=round(base["temperature"], 1),
            humidity=round(base["humidity"], 1),
            precipitation=round(base["precipitation"], 2),
            wind_speed=round(random.uniform(0, 30), 1),
            wind_direction=round(random.uniform(0, 360), 1),
            visibility=round(visibility, 1),
            condition=base["condition"],
            pressure=round(random.uniform(1010, 1020), 1),
            uv_index=round(random.uniform(0, 11), 1) if base["condition"] == "sunny" else round(random.uniform(0, 3), 1),
        )
    
    def produce(self, interval_seconds: float = 60.0) -> None:
        """Start producing weather data to Kafka."""
        if not self.producer:
            self.connect_kafka()
        
        logger.info(f"Starting weather data production for {len(self.stations)} stations...")
        
        message_count = 0
        try:
            while True:
                for station in self.stations:
                    data_point = self.generate_data_point(station)
                    
                    self.producer.send(
                        kafka_config.topic_weather,
                        key=data_point.station_id,
                        value=asdict(data_point)
                    )
                    message_count += 1
                
                logger.info(f"Produced {message_count} weather messages")
                self.producer.flush()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info(f"Stopping producer. Total messages sent: {message_count}")
        finally:
            if self.producer:
                self.producer.close()

def main():
    """Main entry point."""
    producer = WeatherDataProducer()
    producer.produce(interval_seconds=60.0)

if __name__ == "__main__":
    main()
