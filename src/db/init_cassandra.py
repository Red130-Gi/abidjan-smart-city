from src.db.cassandra_db import CassandraConnector
import time

def main():
    print("Initializing Cassandra Schema...")
    # Retry logic because Cassandra takes time to start
    for i in range(10):
        try:
            db = CassandraConnector(hosts=['cassandra'])
            db.connect()
            print("Schema initialized successfully.")
            db.close()
            return
        except Exception as e:
            print(f"Attempt {i+1}/10 failed: {e}")
            time.sleep(5)
    
    print("Failed to initialize Cassandra after 10 attempts.")

if __name__ == "__main__":
    main()
