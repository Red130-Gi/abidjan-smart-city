import sys
sys.path.append('.')
from src.ml.prediction_service import prediction_service
import traceback

print("🔍 Starting Debug Script")
try:
    print("1. Testing DB Connection...")
    conn = prediction_service._get_db_connection()
    print("✅ DB Connection Successful")
    # conn.close() - Keep open for next steps

    print("2. Loading Data (Raw Cursor)...")
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM traffic_segment_stats WHERE window_start > NOW() - INTERVAL '2 days'")
    count = cur.fetchone()[0]
    print(f"✅ Found {count} rows")
    cur.close()

    print("3. Loading Data (Pandas with Limit)...")
    query = """
        SELECT * FROM traffic_segment_stats 
        WHERE window_start > NOW() - INTERVAL '2 days'
        LIMIT 100
    """
    df = pd.read_sql(query, conn)
    print(f"✅ Pandas Loaded 100 rows. Shape: {df.shape}")
    
    print("4. Loading Full Data (2 days)...")
    df_full = prediction_service.load_training_data(days=2)
    print(f"✅ Full Data Loaded. Shape: {df_full.shape}")

    print("3. Testing Preprocessing...")
    clean_df = prediction_service.engineer.clean_data(df)
    print(f"✅ Data Cleaned. Shape: {clean_df.shape}")

except Exception as e:
    print("❌ ERROR OCCURRED:")
    traceback.print_exc()
