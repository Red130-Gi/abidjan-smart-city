import sys
sys.path.append('.')
import psycopg2
from config.settings import postgres_config

def test_access():
    print("Testing DB Access...")
    try:
        conn = psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
        cur = conn.cursor()
        
        # Check tables
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        print("Tables found:", [t[0] for t in tables])
        
        # Check weather_data
        cur.execute("SELECT count(*) FROM weather_data;")
        count = cur.fetchone()[0]
        print(f"Weather data count: {count}")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_access()
