"""
MongoDB Initialization for Smart City Platform
Creates collections and indexes for unstructured data storage.
"""
from pymongo import MongoClient, ASCENDING, DESCENDING, GEOSPHERE
from datetime import datetime
import sys
sys.path.append('..')
from config.settings import mongo_config

def init_mongodb():
    """Initialize MongoDB collections and indexes."""
    try:
        client = MongoClient(mongo_config.connection_string)
        db = client[mongo_config.database]
        
        print("Initializing MongoDB collections...")
        
        # Raw traffic data collection
        if "raw_traffic_data" not in db.list_collection_names():
            db.create_collection("raw_traffic_data", 
                timeseries={
                    "timeField": "timestamp",
                    "metaField": "vehicle_id",
                    "granularity": "seconds"
                }
            )
        raw_traffic = db["raw_traffic_data"]
        raw_traffic.create_index([("timestamp", DESCENDING)])
        raw_traffic.create_index([("vehicle_id", ASCENDING)])
        raw_traffic.create_index([("segment_id", ASCENDING)])
        raw_traffic.create_index([("location", GEOSPHERE)])
        
        # Raw weather data collection
        if "raw_weather_data" not in db.list_collection_names():
            db.create_collection("raw_weather_data",
                timeseries={
                    "timeField": "timestamp",
                    "metaField": "station_id",
                    "granularity": "minutes"
                }
            )
        raw_weather = db["raw_weather_data"]
        raw_weather.create_index([("timestamp", DESCENDING)])
        raw_weather.create_index([("station_id", ASCENDING)])
        
        # System logs collection
        system_logs = db["system_logs"]
        system_logs.create_index([("timestamp", DESCENDING)])
        system_logs.create_index([("level", ASCENDING)])
        system_logs.create_index([("service", ASCENDING)])
        
        # ML model artifacts collection
        ml_models = db["ml_models"]
        ml_models.create_index([("model_name", ASCENDING), ("version", DESCENDING)])
        ml_models.create_index([("created_at", DESCENDING)])
        
        # Training data snapshots
        training_data = db["training_data_snapshots"]
        training_data.create_index([("snapshot_date", DESCENDING)])
        training_data.create_index([("model_type", ASCENDING)])
        
        # API request logs
        api_logs = db["api_request_logs"]
        api_logs.create_index([("timestamp", DESCENDING)])
        api_logs.create_index([("user_id", ASCENDING)])
        api_logs.create_index([("endpoint", ASCENDING)])
        api_logs.create_index([("response_time_ms", ASCENDING)])
        
        # Insert sample document to verify
        system_logs.insert_one({
            "timestamp": datetime.utcnow(),
            "level": "INFO",
            "service": "mongodb_init",
            "message": "MongoDB initialization completed successfully",
            "details": {
                "collections_created": [
                    "raw_traffic_data",
                    "raw_weather_data", 
                    "system_logs",
                    "ml_models",
                    "training_data_snapshots",
                    "api_request_logs"
                ]
            }
        })
        
        print("MongoDB initialization complete!")
        print(f"Collections: {db.list_collection_names()}")
        
        client.close()
        
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")
        raise

if __name__ == "__main__":
    init_mongodb()
