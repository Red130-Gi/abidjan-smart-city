import json
import logging
from kafka import KafkaConsumer
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from config.settings import kafka_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Traffic Consumer for Cassandra...")
    
    # Connect to Cassandra
    try:
        cluster = Cluster(['cassandra'], port=9042)
        session = cluster.connect('smart_city')
        logger.info("Connected to Cassandra")
        
        insert_stmt = session.prepare("""
            INSERT INTO traffic_data (segment_id, timestamp, vehicle_id, speed, latitude, longitude, vehicle_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """)
    except Exception as e:
        logger.error(f"Failed to connect to Cassandra: {e}")
        return

    # Connect to Kafka
    consumer = KafkaConsumer(
        kafka_config.topic_traffic,
        bootstrap_servers=kafka_config.bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='cassandra_writer_v2' # New group to force reset
    )
    logger.info(f"Connected to Kafka topic: {kafka_config.topic_traffic}")

    batch = BatchStatement()
    count = 0
    
    try:
        for message in consumer:
            data = message.value
            
            # Add to batch
            batch.add(insert_stmt, (
                data['segment_id'],
                data['timestamp'], # ISO string, Cassandra handles it if format is correct
                data['vehicle_id'],
                data['speed'],
                data['latitude'],
                data['longitude'],
                data['vehicle_type']
            ))
            count += 1
            
            # Execute batch every 100 records
            if count >= 100:
                session.execute(batch)
                batch = BatchStatement()
                count = 0
                logger.info("Flushed batch to Cassandra")
                
    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    except Exception as e:
        logger.error(f"Error processing message: {e}")
    finally:
        if count > 0:
            session.execute(batch)
        cluster.shutdown()
        consumer.close()

if __name__ == "__main__":
    main()
