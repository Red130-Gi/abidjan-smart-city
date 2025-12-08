"""
Incident Producer for Abidjan Smart City Platform
Simulates traffic incidents (accidents, breakdowns, road closures).
"""
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
import uuid
import sys
sys.path.append('..')
from config.settings import kafka_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INCIDENT_TYPES = [
    {"type": "accident", "severity_range": (2, 5), "duration_range": (15, 120)},
    {"type": "breakdown", "severity_range": (1, 3), "duration_range": (10, 60)},
    {"type": "road_closure", "severity_range": (4, 5), "duration_range": (60, 480)},
    {"type": "flooding", "severity_range": (3, 5), "duration_range": (30, 180)},
    {"type": "police_checkpoint", "severity_range": (1, 2), "duration_range": (30, 120)},
    {"type": "protest", "severity_range": (3, 5), "duration_range": (60, 240)},
]

HOTSPOT_LOCATIONS = [
    {"name": "Pont HKB", "lat": 5.3100, "lon": -4.0050, "incident_probability": 0.15},
    {"name": "Carrefour Indénié", "lat": 5.3300, "lon": -4.0100, "incident_probability": 0.12},
    {"name": "Adjamé Marché", "lat": 5.3550, "lon": -4.0350, "incident_probability": 0.10},
    {"name": "Yopougon Carrefour", "lat": 5.3400, "lon": -4.0800, "incident_probability": 0.10},
    {"name": "Treichville Gare", "lat": 5.2900, "lon": -4.0050, "incident_probability": 0.08},
    {"name": "Cocody Angré", "lat": 5.3600, "lon": -3.9700, "incident_probability": 0.07},
    {"name": "Abobo Terminus", "lat": 5.4200, "lon": -4.0400, "incident_probability": 0.10},
    {"name": "Pont Charles de Gaulle", "lat": 5.2950, "lon": -4.0100, "incident_probability": 0.12},
]

@dataclass
class IncidentDataPoint:
    """Represents a traffic incident."""
    incident_id: str
    incident_type: str
    timestamp: str
    latitude: float
    longitude: float
    location_name: str
    severity: int  # 1-5
    description: str
    estimated_duration_minutes: int
    lanes_affected: int
    is_resolved: bool
    reported_by: str
    vehicles_involved: int
    injuries: int

class IncidentProducer:
    """Produces simulated incident data for Kafka ingestion."""
    
    def __init__(self):
        self.active_incidents: Dict[str, Dict[str, Any]] = {}
        self.producer = None
    
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
    
    def _generate_new_incident(self, location: Dict[str, Any]) -> Optional[IncidentDataPoint]:
        """Generate a new incident at a location based on probability."""
        if random.random() > location["incident_probability"]:
            return None
        
        incident_config = random.choice(INCIDENT_TYPES)
        severity = random.randint(*incident_config["severity_range"])
        duration = random.randint(*incident_config["duration_range"])
        
        incident_id = f"INC_{uuid.uuid4().hex[:8].upper()}"
        
        # Store in active incidents
        self.active_incidents[incident_id] = {
            "created_at": datetime.utcnow(),
            "duration_minutes": duration,
        }
        
        vehicles = 0
        injuries = 0
        if incident_config["type"] == "accident":
            vehicles = random.randint(1, 4)
            injuries = random.randint(0, vehicles * 2)
        
        descriptions = {
            "accident": f"Accident impliquant {vehicles} véhicule(s) sur {location['name']}",
            "breakdown": f"Véhicule en panne sur {location['name']}",
            "road_closure": f"Route fermée à {location['name']}",
            "flooding": f"Inondation signalée à {location['name']}",
            "police_checkpoint": f"Contrôle de police à {location['name']}",
            "protest": f"Manifestation en cours à {location['name']}",
        }
        
        return IncidentDataPoint(
            incident_id=incident_id,
            incident_type=incident_config["type"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            latitude=location["lat"] + random.uniform(-0.005, 0.005),
            longitude=location["lon"] + random.uniform(-0.005, 0.005),
            location_name=location["name"],
            severity=severity,
            description=descriptions.get(incident_config["type"], "Incident signalé"),
            estimated_duration_minutes=duration,
            lanes_affected=random.randint(1, 3),
            is_resolved=False,
            reported_by=random.choice(["camera", "user_report", "patrol", "sensor"]),
            vehicles_involved=vehicles,
            injuries=injuries,
        )
    
    def _resolve_old_incidents(self) -> list:
        """Resolve incidents that have exceeded their duration."""
        resolved = []
        current_time = datetime.utcnow()
        
        for incident_id, data in list(self.active_incidents.items()):
            elapsed = (current_time - data["created_at"]).total_seconds() / 60
            if elapsed >= data["duration_minutes"]:
                resolved.append(incident_id)
                del self.active_incidents[incident_id]
        
        return resolved
    
    def produce(self, interval_seconds: float = 30.0) -> None:
        """Start producing incident data to Kafka."""
        if not self.producer:
            self.connect_kafka()
        
        logger.info("Starting incident data production...")
        
        message_count = 0
        try:
            while True:
                # Resolve old incidents
                resolved = self._resolve_old_incidents()
                for incident_id in resolved:
                    resolution_msg = {
                        "incident_id": incident_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "is_resolved": True,
                        "resolution_type": "cleared",
                    }
                    self.producer.send(
                        kafka_config.topic_incidents,
                        key=incident_id,
                        value=resolution_msg
                    )
                    logger.info(f"Resolved incident: {incident_id}")
                
                # Generate new incidents
                for location in HOTSPOT_LOCATIONS:
                    incident = self._generate_new_incident(location)
                    if incident:
                        self.producer.send(
                            kafka_config.topic_incidents,
                            key=incident.incident_id,
                            value=asdict(incident)
                        )
                        message_count += 1
                        logger.info(f"New incident: {incident.incident_type} at {incident.location_name}")
                
                self.producer.flush()
                logger.info(f"Active incidents: {len(self.active_incidents)}")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info(f"Stopping producer. Total incidents generated: {message_count}")
        finally:
            if self.producer:
                self.producer.close()

def main():
    """Main entry point."""
    producer = IncidentProducer()
    producer.produce(interval_seconds=30.0)

if __name__ == "__main__":
    main()
