"""
Configuration module for Abidjan Smart City Platform
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_traffic: str = os.getenv("KAFKA_TOPIC_TRAFFIC", "traffic_data")
    topic_weather: str = os.getenv("KAFKA_TOPIC_WEATHER", "weather_data")
    topic_incidents: str = os.getenv("KAFKA_TOPIC_INCIDENTS", "incident_alerts")
    topic_predictions: str = os.getenv("KAFKA_TOPIC_PREDICTIONS", "traffic_predictions")

@dataclass
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "admin")
    password: str = os.getenv("POSTGRES_PASSWORD", "password")
    database: str = os.getenv("POSTGRES_DB", "smart_city")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class MongoConfig:
    host: str = os.getenv("MONGO_HOST", "localhost")
    port: int = int(os.getenv("MONGO_PORT", "27017"))
    user: str = os.getenv("MONGO_USER", "admin")
    password: str = os.getenv("MONGO_PASSWORD", "password")
    database: str = os.getenv("MONGO_DB", "smart_city_logs")
    
    @property
    def connection_string(self) -> str:
        return f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}"

@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))

@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    jwt_secret: str = os.getenv("JWT_SECRET_KEY", "supersecretkey_change_in_production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Global config instances
kafka_config = KafkaConfig()
postgres_config = PostgresConfig()
mongo_config = MongoConfig()
redis_config = RedisConfig()
api_config = APIConfig()
