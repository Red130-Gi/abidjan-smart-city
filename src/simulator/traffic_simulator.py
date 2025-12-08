"""
Traffic Data Simulator - Inserts simulated traffic data directly into PostgreSQL.
This provides data for the Grafana dashboard without requiring Spark Streaming.
"""
import psycopg2
import random
import time
from datetime import datetime, timedelta
import sys
sys.path.append('.')
from config.settings import postgres_config

ROAD_SEGMENTS = [
    ('SEG001', 'Pont HKB'),
    ('SEG002', 'Boulevard VGE'),
    ('SEG003', 'Autoroute du Nord'),
    ('SEG004', 'Rue Lepic'),
    ('SEG005', 'Boulevard Nangui Abrogoua'),
    ('SEG006', 'Pont Général de Gaulle'),
    ('SEG007', 'Boulevard de Marseille'),
    ('SEG008', 'Voie Express Yopougon'),
]

def get_congestion_level(avg_speed):
    if avg_speed < 10:
        return 'severe'
    elif avg_speed < 25:
        return 'heavy'
    elif avg_speed < 40:
        return 'moderate'
    else:
        return 'light'

def generate_traffic_data():
    """Generate realistic traffic statistics for a segment."""
    # Vary speed based on time of day simulation
    base_speed = random.uniform(20, 60)
    
    # Add some randomness
    avg_speed = max(5, base_speed + random.uniform(-15, 15))
    max_speed = avg_speed + random.uniform(10, 25)
    min_speed = max(0, avg_speed - random.uniform(10, 20))
    speed_stddev = random.uniform(5, 15)
    
    vehicle_count = random.randint(20, 150)
    stopped_vehicles = random.randint(0, int(vehicle_count * 0.3))
    
    gbaka_count = random.randint(5, int(vehicle_count * 0.3))
    bus_count = random.randint(2, int(vehicle_count * 0.15))
    taxi_count = random.randint(3, int(vehicle_count * 0.25))
    
    return {
        'avg_speed': round(avg_speed, 2),
        'max_speed': round(max_speed, 2),
        'min_speed': round(min_speed, 2),
        'speed_stddev': round(speed_stddev, 2),
        'vehicle_count': vehicle_count,
        'stopped_vehicles': stopped_vehicles,
        'gbaka_count': gbaka_count,
        'bus_count': bus_count,
        'taxi_count': taxi_count,
        'congestion_level': get_congestion_level(avg_speed),
    }

def insert_traffic_stats(conn):
    """Insert traffic statistics for all segments."""
    now = datetime.now()
    window_start = now - timedelta(seconds=60)
    window_end = now
    
    with conn.cursor() as cur:
        for segment_id, segment_name in ROAD_SEGMENTS:
            data = generate_traffic_data()
            
            cur.execute("""
                INSERT INTO traffic_segment_stats 
                (segment_id, window_start, window_end, avg_speed, max_speed, min_speed, 
                 speed_stddev, vehicle_count, stopped_vehicles, gbaka_count, bus_count, 
                 taxi_count, congestion_level, processed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                segment_id, window_start, window_end,
                data['avg_speed'], data['max_speed'], data['min_speed'],
                data['speed_stddev'], data['vehicle_count'], data['stopped_vehicles'],
                data['gbaka_count'], data['bus_count'], data['taxi_count'],
                data['congestion_level'], now
            ))
        
        conn.commit()
        print(f"[{now.strftime('%H:%M:%S')}] Inserted traffic stats for {len(ROAD_SEGMENTS)} segments")

def insert_weather_data(conn):
    """Insert weather data."""
    now = datetime.now()
    
    with conn.cursor() as cur:
        stations = ['WS_001', 'WS_002', 'WS_003', 'WS_004', 'WS_005']
        for station_id in stations:
            cur.execute("""
                INSERT INTO weather_data 
                (station_id, recorded_at, temperature, humidity, precipitation, 
                 wind_speed, wind_direction, visibility, condition, pressure, uv_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                station_id, now,
                round(random.uniform(26, 32), 1),  # temperature
                round(random.uniform(60, 85), 1),  # humidity
                round(random.uniform(0, 5), 2),    # precipitation
                round(random.uniform(5, 15), 1),   # wind_speed
                round(random.uniform(0, 360), 1),  # wind_direction
                round(random.uniform(8, 10), 1),   # visibility
                random.choice(['sunny', 'partly_cloudy', 'cloudy', 'light_rain']),
                round(random.uniform(1010, 1020), 1),  # pressure
                round(random.uniform(6, 11), 1)    # uv_index
            ))
        
        conn.commit()
        print(f"[{now.strftime('%H:%M:%S')}] Inserted weather data for {len(stations)} stations")

def main():
    print("Starting Traffic Data Simulator...")
    print(f"Connecting to PostgreSQL at {postgres_config.host}:{postgres_config.port}")
    
    try:
        conn = psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
        print("Connected to PostgreSQL!")
        
        iteration = 0
        while True:
            iteration += 1
            insert_traffic_stats(conn)
            
            # Insert weather data every 5 iterations (less frequently)
            if iteration % 5 == 0:
                insert_weather_data(conn)
            
            # Wait 10 seconds before next batch
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nStopping simulator...")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
