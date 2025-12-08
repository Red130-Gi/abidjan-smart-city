"""
IoT Traffic Data Producer for Abidjan Smart City Platform
Simulates GPS data from vehicles (cars, Gbakas, buses) across Abidjan's road network.
"""
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
import sys
sys.path.append('..')
from config.settings import kafka_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Abidjan key locations (latitude, longitude)
ABIDJAN_LOCATIONS = {
    "plateau": (5.3167, -4.0167),
    "cocody": (5.3500, -3.9833),
    "yopougon": (5.3333, -4.0833),
    "adjame": (5.3500, -4.0333),
    "treichville": (5.2833, -4.0000),
    "marcory": (5.2833, -3.9667),
    "koumassi": (5.2833, -3.9333),
    "port_bouet": (5.2500, -3.9333),
    "abobo": (5.4167, -4.0333),
    "bingerville": (5.3500, -3.9000),
    "pont_hkb": (5.3100, -4.0050),
    "pont_general_de_gaulle": (5.2950, -4.0100),
    "boulevard_vge": (5.3200, -4.0200),
    "autoroute_nord": (5.3800, -4.0400),
}

# Road segments for simulation
ROAD_SEGMENTS = [
    {"id": "SEG001", "name": "Pont HKB", "start": "plateau", "end": "treichville", "max_speed": 60, "lanes": 4},
    {"id": "SEG002", "name": "Boulevard VGE", "start": "cocody", "end": "plateau", "max_speed": 80, "lanes": 6},
    {"id": "SEG003", "name": "Autoroute du Nord", "start": "adjame", "end": "abobo", "max_speed": 100, "lanes": 4},
    {"id": "SEG004", "name": "Rue Lepic", "start": "plateau", "end": "adjame", "max_speed": 50, "lanes": 2},
    {"id": "SEG005", "name": "Boulevard Nangui Abrogoua", "start": "yopougon", "end": "adjame", "max_speed": 70, "lanes": 4},
    {"id": "SEG006", "name": "Pont Général de Gaulle", "start": "treichville", "end": "marcory", "max_speed": 50, "lanes": 4},
    {"id": "SEG007", "name": "Boulevard de Marseille", "start": "koumassi", "end": "marcory", "max_speed": 60, "lanes": 4},
    {"id": "SEG008", "name": "Voie Express Yopougon", "start": "yopougon", "end": "plateau", "max_speed": 80, "lanes": 4},
]

VEHICLE_TYPES = ["car", "gbaka", "bus", "taxi", "moto"]

@dataclass
class TrafficDataPoint:
    """Represents a single traffic data point from an IoT device."""
    vehicle_id: str
    vehicle_type: str
    timestamp: str
    latitude: float
    longitude: float
    speed: float  # km/h
    heading: float  # degrees (0-360)
    segment_id: str
    segment_name: str
    acceleration: float  # m/s²
    is_stopped: bool
    occupancy: int  # for buses/gbakas
    fuel_level: float  # percentage

class TrafficDataProducer:
    """Produces simulated traffic data for Kafka ingestion."""
    
    def __init__(self, num_vehicles: int = 100):
        self.num_vehicles = num_vehicles
        self.vehicles = self._initialize_vehicles()
        self.producer = None
        
    def _initialize_vehicles(self) -> List[Dict[str, Any]]:
        """Initialize vehicle fleet with random attributes."""
        vehicles = []
        for i in range(self.num_vehicles):
            vehicle_type = random.choice(VEHICLE_TYPES)
            segment = random.choice(ROAD_SEGMENTS)
            start_loc = ABIDJAN_LOCATIONS[segment["start"]]
            
            vehicles.append({
                "id": f"VEH_{i:05d}",
                "type": vehicle_type,
                "current_segment": segment,
                "latitude": start_loc[0] + random.uniform(-0.01, 0.01),
                "longitude": start_loc[1] + random.uniform(-0.01, 0.01),
                "speed": random.uniform(10, segment["max_speed"]),
                "heading": random.uniform(0, 360),
                "fuel_level": random.uniform(20, 100),
                "occupancy": random.randint(1, 20) if vehicle_type in ["gbaka", "bus"] else random.randint(1, 4),
            })
        return vehicles
    
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
    
    def _simulate_movement(self, vehicle: Dict[str, Any]) -> None:
        """Simulate vehicle movement with realistic patterns."""
        segment = vehicle["current_segment"]
        
        # Traffic congestion factor (higher during rush hours)
        hour = datetime.now().hour
        congestion_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            congestion_factor = 0.4  # Rush hour - slower
        elif 12 <= hour <= 14:
            congestion_factor = 0.7  # Lunch time - moderate
        
        # Random speed variation
        target_speed = segment["max_speed"] * congestion_factor * random.uniform(0.6, 1.0)
        
        # Smooth acceleration/deceleration
        speed_diff = target_speed - vehicle["speed"]
        acceleration = min(max(speed_diff * 0.1, -3), 3)  # Limit acceleration
        vehicle["speed"] = max(0, vehicle["speed"] + acceleration)
        
        # Update position based on speed and heading
        speed_ms = vehicle["speed"] / 3.6  # Convert to m/s
        lat_delta = (speed_ms * 0.00001) * random.uniform(-0.5, 1.5)
        lon_delta = (speed_ms * 0.00001) * random.uniform(-0.5, 1.5)
        
        vehicle["latitude"] += lat_delta
        vehicle["longitude"] += lon_delta
        vehicle["heading"] = (vehicle["heading"] + random.uniform(-10, 10)) % 360
        
        # Consume fuel
        vehicle["fuel_level"] = max(0, vehicle["fuel_level"] - random.uniform(0.001, 0.01))
        
        # Occasionally change segment
        if random.random() < 0.02:
            vehicle["current_segment"] = random.choice(ROAD_SEGMENTS)
    
    def generate_data_point(self, vehicle: Dict[str, Any]) -> TrafficDataPoint:
        """Generate a traffic data point for a vehicle."""
        self._simulate_movement(vehicle)
        
        return TrafficDataPoint(
            vehicle_id=vehicle["id"],
            vehicle_type=vehicle["type"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            latitude=vehicle["latitude"],
            longitude=vehicle["longitude"],
            speed=round(vehicle["speed"], 2),
            heading=round(vehicle["heading"], 2),
            segment_id=vehicle["current_segment"]["id"],
            segment_name=vehicle["current_segment"]["name"],
            acceleration=round(random.uniform(-2, 2), 2),
            is_stopped=vehicle["speed"] < 5,
            occupancy=vehicle["occupancy"],
            fuel_level=round(vehicle["fuel_level"], 2),
        )
    
    def produce(self, interval_seconds: float = 1.0) -> None:
        """Start producing traffic data to Kafka."""
        if not self.producer:
            self.connect_kafka()
        
        logger.info(f"Starting traffic data production for {self.num_vehicles} vehicles...")
        
        message_count = 0
        try:
            while True:
                for vehicle in self.vehicles:
                    data_point = self.generate_data_point(vehicle)
                    
                    self.producer.send(
                        kafka_config.topic_traffic,
                        key=data_point.vehicle_id,
                        value=asdict(data_point)
                    )
                    message_count += 1
                
                if message_count % 1000 == 0:
                    logger.info(f"Produced {message_count} messages")
                
                self.producer.flush()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info(f"Stopping producer. Total messages sent: {message_count}")
        finally:
            if self.producer:
                self.producer.close()

def main():
    """Main entry point."""
    producer = TrafficDataProducer(num_vehicles=200)
    producer.produce(interval_seconds=1.0)

if __name__ == "__main__":
    main()
