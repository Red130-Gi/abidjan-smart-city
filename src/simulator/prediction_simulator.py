"""
Prediction Simulator - Uses Real ML Models for Predictions
Trains on actual traffic data and generates real predictions.
Also simulates route calculations for the routes dashboard.
"""
import time
import random
import uuid
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
import json
import sys
sys.path.append('.')
sys.path.append('/app')
from config.settings import postgres_config

# Try to import Ensemble ML predictor
try:
    from src.ml.ensemble_model import ensemble_predictor
    USE_REAL_ML = True
    print("✅ Ensemble ML Predictor (XGBoost + LSTM) imported successfully")
except ImportError as e:
    USE_REAL_ML = False
    print(f"⚠️ Could not import Ensemble Predictor: {e}")
    print("   Falling back to simulated predictions")

# Road segments
SEGMENTS = {
    'SEG001': 'Pont HKB',
    'SEG002': 'Boulevard VGE',
    'SEG003': 'Autoroute du Nord',
    'SEG004': 'Rue Lepic',
    'SEG005': 'Bd Nangui Abrogoua',
    'SEG006': 'Pont Général de Gaulle',
    'SEG007': 'Boulevard de Marseille',
    'SEG008': 'Voie Express Yopougon'
}

HORIZONS = [5, 15, 30, 60]  # minutes

def get_db_connection():
    return psycopg2.connect(
        host=postgres_config.host,
        port=postgres_config.port,
        user=postgres_config.user,
        password=postgres_config.password,
        database=postgres_config.database
    )

def get_congestion_level(speed):
    if speed < 10:
        return 'severe'
    elif speed < 25:
        return 'heavy'
    elif speed < 40:
        return 'moderate'
    else:
        return 'light'

def generate_real_predictions(predictor):
    """Generate predictions using Ensemble ML models."""
    now = datetime.now()
    
    # Train if needed (Ensemble handles its own training logic)
    # predictor is now the ensemble_predictor instance
    try:
        predictor.train_all()
    except Exception as e:
        print(f"Training warning: {e}")
    
    predictions = []
    for segment_id in SEGMENTS:
        for horizon in HORIZONS:
            pred = predictor.predict(segment_id, horizon)
            predictions.append(pred)
            
    # Save to database (using helper from real_predictor inside ensemble if needed, or manual)
    # Since ensemble returns dicts, we can reuse the save logic if we access the inner predictor
    # or just implement saving here. Let's implement saving here to be safe.
    
    conn = get_db_connection()
    data = [
        (
            p['segment_id'],
            p['prediction_time'],
            p['target_time'],
            p['horizon_minutes'],
            p['predicted_speed'],
            p['predicted_congestion'],
            p['confidence_score'],
            p['model_type'],
            json.dumps(p.get('details', {}))
        )
        for p in predictions
    ]
    
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO traffic_predictions 
                (segment_id, prediction_time, target_time, horizon_minutes,
                 predicted_speed, predicted_congestion, confidence_score, 
                 model_type, features_used)
                VALUES %s
            """, data)
        conn.commit()
    except Exception as e:
        print(f"Error saving predictions: {e}")
    finally:
        conn.close()
    
    print(f"[{now.strftime('%H:%M:%S')}] 🔮 Generated {len(predictions)} Ensemble predictions (XGB+LSTM)")
    
    return predictions

def generate_simulated_predictions(conn):
    """Fallback: Generate simulated predictions (no ML)."""
    now = datetime.now()
    hour = now.hour
    
    predictions = []
    
    for segment_id in SEGMENTS:
        # Base speed varies by segment and time
        base_speed = 35 + random.uniform(-5, 10)
        
        # Rush hour effect
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_speed -= 20
        elif 22 <= hour or hour <= 5:
            base_speed += 15
        
        for horizon in HORIZONS:
            target_time = now + timedelta(minutes=horizon)
            target_hour = target_time.hour
            
            # Predict speed at target time
            predicted_speed = base_speed
            
            # Adjust for rush hour at target time
            if 7 <= target_hour <= 9 or 17 <= target_hour <= 19:
                predicted_speed -= 15
            elif 22 <= target_hour or target_hour <= 5:
                predicted_speed += 10
            
            # Add noise
            predicted_speed += random.uniform(-8, 8)
            predicted_speed = max(5, min(75, predicted_speed))
            
            congestion = get_congestion_level(predicted_speed)
            
            # Confidence decreases with horizon
            confidence = 0.95 - (horizon / 100) + random.uniform(-0.05, 0.05)
            confidence = max(0.5, min(0.98, confidence))
            
            predictions.append((
                segment_id,
                now,
                target_time,
                horizon,
                round(predicted_speed, 2),
                congestion,
                round(confidence, 3),
                'simulated_heuristic',
                '{}'
            ))
    
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO traffic_predictions 
            (segment_id, prediction_time, target_time, horizon_minutes,
             predicted_speed, predicted_congestion, confidence_score, model_type, features_used)
            VALUES %s
        """, predictions)
    
    conn.commit()
    print(f"[{now.strftime('%H:%M:%S')}] ⚠️ Generated {len(predictions)} SIMULATED predictions (no ML)")

def generate_anomalies(conn):
    """Generate predicted anomalies based on severe congestion predictions."""
    now = datetime.now()
    
    # Randomly generate anomalies
    if random.random() < 0.3:  # 30% chance of anomaly
        segment_id = random.choice(list(SEGMENTS.keys()))
        expected_start = now + timedelta(minutes=random.randint(10, 45))
        
        anomaly_types = [
            ('congestion_spike', 'Pic de congestion prévu'),
            ('accident_risk', 'Risque d\'accident élevé'),
            ('event_traffic', 'Trafic lié à un événement')
        ]
        
        anomaly_type, description = random.choice(anomaly_types)
        severity = random.randint(2, 5)
        probability = round(0.6 + random.uniform(0, 0.35), 3)
        
        actions = [
            'Activer itinéraires alternatifs',
            'Renforcer surveillance',
            'Préparer équipes d\'intervention',
            'Informer usagers via app mobile'
        ]
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predicted_anomalies 
                (segment_id, prediction_time, expected_start, anomaly_type, 
                 severity, probability, description, recommended_action)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                segment_id, now, expected_start, anomaly_type,
                severity, probability,
                f"{description} sur {SEGMENTS[segment_id]}",
                random.choice(actions)
            ))
        
        conn.commit()
        print(f"[{now.strftime('%H:%M:%S')}] ⚠️ Generated anomaly: {anomaly_type} on {segment_id}")

def simulate_route_request(conn):
    """Simulate a route calculation request."""
    now = datetime.now()
    
    # Random origin and destination
    segments = list(SEGMENTS.keys())
    origin = random.choice(segments)
    destination = random.choice([s for s in segments if s != origin])
    
    request_id = f"RT{uuid.uuid4().hex[:8].upper()}"
    
    # Generate route data
    distance = 3 + random.uniform(2, 10)
    normal_duration = distance * 2 + random.uniform(5, 15)  # ~30 km/h + variance
    optimized_duration = normal_duration * (0.65 + random.uniform(0, 0.2))
    
    fuel_normal = distance * 0.08
    fuel_optimized = distance * 1.1 * 0.08
    
    with conn.cursor() as cur:
        # Insert request
        cur.execute("""
            INSERT INTO route_requests 
            (request_id, origin_segment, destination_segment, origin_name, destination_name, status)
            VALUES (%s, %s, %s, %s, %s, 'completed')
        """, (request_id, origin, destination, SEGMENTS[origin], SEGMENTS[destination]))
        
        # Insert normal route
        cur.execute("""
            INSERT INTO optimized_routes 
            (request_id, route_type, total_distance_km, estimated_duration_minutes,
             total_congestion_score, fuel_consumption_liters, co2_emissions_kg, time_saved_minutes,
             route_geometry, waypoints)
            VALUES (%s, 'normal', %s, %s, %s, %s, %s, 0, '{}', '[]')
            RETURNING id
        """, (
            request_id, round(distance, 2), round(normal_duration, 1),
            round(2.5 + random.uniform(-0.5, 1), 2),
            round(fuel_normal, 2), round(fuel_normal * 2.31, 2)
        ))
        normal_route_id = cur.fetchone()[0]
        
        # Insert optimized route
        time_saved = round(normal_duration - optimized_duration, 1)
        cur.execute("""
            INSERT INTO optimized_routes 
            (request_id, route_type, total_distance_km, estimated_duration_minutes,
             total_congestion_score, fuel_consumption_liters, co2_emissions_kg, time_saved_minutes,
             route_geometry, waypoints)
            VALUES (%s, 'optimized', %s, %s, %s, %s, %s, %s, '{}', '[]')
            RETURNING id
        """, (
            request_id, round(distance * 1.1, 2), round(optimized_duration, 1),
            round(1.5 + random.uniform(-0.3, 0.5), 2),
            round(fuel_optimized, 2), round(fuel_optimized * 2.31, 2),
            time_saved
        ))
        optimized_route_id = cur.fetchone()[0]
        
        # Add segments for both routes
        route_segments = [origin, random.choice([s for s in segments if s not in [origin, destination]]), destination]
        
        for i, seg_id in enumerate(route_segments):
            for route_id in [normal_route_id, optimized_route_id]:
                speed = 25 + random.uniform(-10, 25)
                cur.execute("""
                    INSERT INTO route_segments 
                    (route_id, segment_order, segment_id, segment_name, 
                     distance_km, duration_minutes, current_speed, predicted_speed, congestion_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    route_id, i, seg_id, SEGMENTS[seg_id],
                    round(distance / len(route_segments), 2),
                    round(normal_duration / len(route_segments), 1),
                    round(speed, 1), round(speed + random.uniform(-5, 5), 1),
                    get_congestion_level(speed)
                ))
    
    conn.commit()
    print(f"[{now.strftime('%H:%M:%S')}] 🗺️ Route: {SEGMENTS[origin]} → {SEGMENTS[destination]} (saved {time_saved:.0f} min)")

def cleanup_old_data(conn):
    """Remove old predictions and route data."""
    with conn.cursor() as cur:
        # Keep last 2 hours of predictions
        cur.execute("DELETE FROM traffic_predictions WHERE prediction_time < NOW() - INTERVAL '2 hours'")
        
        # Keep last 24 hours of anomalies
        cur.execute("DELETE FROM predicted_anomalies WHERE created_at < NOW() - INTERVAL '24 hours'")
        
        # Keep last 24 hours of routes
        cur.execute("""
            DELETE FROM route_segments WHERE route_id IN (
                SELECT id FROM optimized_routes WHERE created_at < NOW() - INTERVAL '24 hours'
            )
        """)
        cur.execute("DELETE FROM optimized_routes WHERE created_at < NOW() - INTERVAL '24 hours'")
        cur.execute("DELETE FROM route_requests WHERE created_at < NOW() - INTERVAL '24 hours'")
    
    conn.commit()

def main():
    print("=" * 60)
    print("🔮 Real ML Prediction Simulator")
    print("=" * 60)
    print(f"Connecting to PostgreSQL at {postgres_config.host}:{postgres_config.port}")
    print(f"Using Real ML: {USE_REAL_ML}")
    
    # Initialize predictor if available
    predictor = None
    if USE_REAL_ML:
        try:
            predictor = ensemble_predictor
            print("✅ Ensemble Predictor initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize predictor: {e}")
            predictor = None
    
    try:
        conn = get_db_connection()
        print("✅ Connected to PostgreSQL!")
        
        iteration = 0
        retrain_interval = 24  # Retrain every 24 iterations (~6 minutes)
        
        while True:
            iteration += 1
            
            # Generate predictions
            if predictor and USE_REAL_ML:
                # Force retrain periodically
                if iteration % retrain_interval == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Retraining models...")
                    predictor.train(force=True)
                
                generate_real_predictions(predictor)
            else:
                generate_simulated_predictions(conn)
            
            # Generate anomalies occasionally
            generate_anomalies(conn)
            
            # Simulate route request every 2nd iteration
            if iteration % 2 == 0:
                simulate_route_request(conn)
            
            # Cleanup every 10 iterations
            if iteration % 10 == 0:
                cleanup_old_data(conn)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧹 Cleaned up old data")
            
            # Wait 15 seconds
            time.sleep(15)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulator...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()
