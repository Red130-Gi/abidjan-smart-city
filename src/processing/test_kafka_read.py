from kafka import KafkaConsumer
import json
from config.settings import kafka_config

def main():
    print("Testing Kafka Read...")
    try:
        consumer = KafkaConsumer(
            kafka_config.topic_traffic,
            bootstrap_servers=kafka_config.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest', # Read from beginning to ensure we get something
            consumer_timeout_ms=5000 # Stop after 5 seconds if no message
        )
        print(f"Connected to Kafka: {kafka_config.bootstrap_servers}")
        
        for message in consumer:
            print(f"Received message: {message.value['vehicle_id']}")
            break # Just one
            
        print("Finished.")
        consumer.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
