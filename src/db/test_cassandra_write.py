from cassandra.cluster import Cluster
import datetime

def main():
    print("Testing Cassandra Write...")
    try:
        cluster = Cluster(['cassandra'], port=9042)
        session = cluster.connect('smart_city')
        print("Connected.")
        
        session.execute("""
            INSERT INTO traffic_data (segment_id, timestamp, vehicle_id, speed, latitude, longitude, vehicle_type)
            VALUES ('TEST_SEG', toTimestamp(now()), 'TEST_VEH', 50.0, 5.3, -4.0, 'car')
        """)
        print("Inserted 1 row.")
        cluster.shutdown()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
