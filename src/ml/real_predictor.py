"""
Real ML Prediction Service for Traffic Predictions
Trains on actual traffic_segment_stats data and generates real predictions.
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, using fallback predictions")

import sys
sys.path.append('.')
from config.settings import postgres_config

# Model storage path
MODEL_DIR = '/app/models' if os.path.exists('/app/models') else './models'
SPEED_MODEL_PATH = os.path.join(MODEL_DIR, 'speed_model.pkl')
CONGESTION_MODEL_PATH = os.path.join(MODEL_DIR, 'congestion_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Segments
SEGMENTS = ['SEG001', 'SEG002', 'SEG003', 'SEG004', 'SEG005', 'SEG006', 'SEG007', 'SEG008']
HORIZONS = [5, 15, 30, 60]  # minutes

class RealMLPredictor:
    """
    Real ML predictor that trains on actual traffic data.
    Uses GradientBoostingRegressor for speed and RandomForestClassifier for congestion.
    """
    
    def __init__(self):
        self.speed_model = None
        self.congestion_model = None
        self.label_encoder = None
        self.is_trained = False
        self.last_training_time = None
        self.min_training_samples = 100
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Try to load existing models
        self._load_models()
    
    def _get_db_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
    
    def _load_models(self) -> bool:
        """Load pre-trained models from disk."""
        try:
            if os.path.exists(SPEED_MODEL_PATH) and os.path.exists(CONGESTION_MODEL_PATH):
                with open(SPEED_MODEL_PATH, 'rb') as f:
                    self.speed_model = pickle.load(f)
                with open(CONGESTION_MODEL_PATH, 'rb') as f:
                    self.congestion_model = pickle.load(f)
                if os.path.exists(ENCODER_PATH):
                    with open(ENCODER_PATH, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                self.is_trained = True
                print("✅ Loaded pre-trained ML models from disk")
                return True
        except Exception as e:
            print(f"Could not load models: {e}")
        return False
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            with open(SPEED_MODEL_PATH, 'wb') as f:
                pickle.dump(self.speed_model, f)
            with open(CONGESTION_MODEL_PATH, 'wb') as f:
                pickle.dump(self.congestion_model, f)
            with open(ENCODER_PATH, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"💾 Models saved to {MODEL_DIR}")
        except Exception as e:
            print(f"Could not save models: {e}")
    
    def _fetch_training_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical traffic data for training."""
        conn = self._get_db_connection()
        
        query = """
            SELECT 
                segment_id,
                avg_speed,
                vehicle_count,
                congestion_level,
                EXTRACT(HOUR FROM processed_at) as hour,
                EXTRACT(DOW FROM processed_at) as day_of_week,
                EXTRACT(MINUTE FROM processed_at) as minute,
                processed_at
            FROM traffic_segment_stats
            WHERE processed_at > NOW() - INTERVAL '24 hours'
            ORDER BY processed_at
        """
        
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) < self.min_training_samples:
                print(f"⚠️ Not enough training data ({len(df)} samples, need {self.min_training_samples})")
                return None
            
            print(f"📊 Fetched {len(df)} training samples")
            return df
        except Exception as e:
            print(f"Error fetching training data: {e}")
            conn.close()
            return None
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for training."""
        # Encode segment_id
        self.label_encoder = LabelEncoder()
        df['segment_encoded'] = self.label_encoder.fit_transform(df['segment_id'])
        
        # Create features
        features = df[['segment_encoded', 'hour', 'day_of_week', 'minute', 'vehicle_count']].values
        
        # Targets
        speed_target = df['avg_speed'].values
        
        # Encode congestion levels
        congestion_mapping = {'light': 0, 'moderate': 1, 'heavy': 2, 'severe': 3}
        congestion_target = df['congestion_level'].map(congestion_mapping).fillna(1).values
        
        return features, speed_target, congestion_target
    
    def train(self, force: bool = False) -> bool:
        """
        Train ML models on real traffic data.
        
        Args:
            force: Force retraining even if models exist
            
        Returns:
            True if training successful
        """
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn not available")
            return False
        
        # Check if we need to retrain
        if self.is_trained and not force:
            # Only retrain every 6 hours
            if self.last_training_time:
                hours_since = (datetime.now() - self.last_training_time).total_seconds() / 3600
                if hours_since < 6:
                    return True
        
        print("🎓 Training ML models on real traffic data...")
        
        # Fetch data
        df = self._fetch_training_data()
        if df is None:
            return False
        
        # Prepare features
        X, y_speed, y_congestion = self._prepare_features(df)
        
        # Split data
        X_train, X_test, y_speed_train, y_speed_test = train_test_split(
            X, y_speed, test_size=0.2, random_state=42
        )
        _, _, y_cong_train, y_cong_test = train_test_split(
            X, y_congestion, test_size=0.2, random_state=42
        )
        
        # Train speed model (GradientBoosting)
        print("  📈 Training speed prediction model (GradientBoosting)...")
        self.speed_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.speed_model.fit(X_train, y_speed_train)
        speed_score = self.speed_model.score(X_test, y_speed_test)
        print(f"     Speed model R² score: {speed_score:.3f}")
        
        # Train congestion model (RandomForest)
        print("  🚦 Training congestion classification model (RandomForest)...")
        self.congestion_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.congestion_model.fit(X_train, y_cong_train)
        cong_score = self.congestion_model.score(X_test, y_cong_test)
        print(f"     Congestion model accuracy: {cong_score:.3f}")
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        print(f"✅ Training complete! Speed R²={speed_score:.3f}, Congestion Acc={cong_score:.3f}")
        return True
    
    def predict(self, segment_id: str, horizon_minutes: int = 15) -> Dict:
        """
        Make a real prediction for a segment.
        
        Args:
            segment_id: Segment ID to predict for
            horizon_minutes: How far ahead to predict (minutes)
            
        Returns:
            Prediction dictionary with speed, congestion, confidence
        """
        now = datetime.now()
        target_time = now + timedelta(minutes=horizon_minutes)
        
        # If models not trained, use fallback
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return self._fallback_prediction(segment_id, horizon_minutes, target_time)
        
        # Get current vehicle count (latest from DB)
        vehicle_count = self._get_current_vehicle_count(segment_id)
        
        # Encode segment
        try:
            segment_encoded = self.label_encoder.transform([segment_id])[0]
        except:
            segment_encoded = SEGMENTS.index(segment_id) if segment_id in SEGMENTS else 0
        
        # Prepare features for target time
        features = np.array([[
            segment_encoded,
            target_time.hour,
            target_time.weekday(),
            target_time.minute,
            vehicle_count
        ]])
        
        # Predict speed
        predicted_speed = self.speed_model.predict(features)[0]
        predicted_speed = max(5, min(80, predicted_speed))  # Clamp to reasonable range
        
        # Predict congestion
        congestion_proba = self.congestion_model.predict_proba(features)[0]
        congestion_class = self.congestion_model.predict(features)[0]
        congestion_mapping = {0: 'light', 1: 'moderate', 2: 'heavy', 3: 'severe'}
        predicted_congestion = congestion_mapping.get(int(congestion_class), 'moderate')
        
        # Calculate confidence from prediction probabilities
        confidence = float(max(congestion_proba))
        # Adjust confidence based on horizon (further = less confident)
        confidence *= (1 - horizon_minutes / 200)
        confidence = max(0.5, min(0.98, confidence))
        
        return {
            'segment_id': segment_id,
            'prediction_time': now,
            'target_time': target_time,
            'horizon_minutes': horizon_minutes,
            'predicted_speed': round(predicted_speed, 2),
            'predicted_congestion': predicted_congestion,
            'confidence_score': round(confidence, 3),
            'model_type': 'gradient_boosting_rf',
            'is_real_prediction': True
        }
    
    def _get_current_vehicle_count(self, segment_id: str) -> int:
        """Get latest vehicle count for a segment."""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT vehicle_count FROM traffic_segment_stats 
                WHERE segment_id = %s 
                ORDER BY processed_at DESC LIMIT 1
            """, (segment_id,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result[0] if result else 50
        except:
            return 50
    
    def _fallback_prediction(self, segment_id: str, horizon: int, target_time: datetime) -> Dict:
        """Fallback prediction when models aren't trained."""
        import random
        
        hour = target_time.hour
        base_speed = 40
        
        # Time-based adjustments
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_speed -= 15
        elif 22 <= hour or hour <= 5:
            base_speed += 10
        
        # Add some randomness
        speed = base_speed + random.uniform(-10, 10)
        speed = max(10, min(70, speed))
        
        # Determine congestion
        if speed < 15:
            congestion = 'severe'
        elif speed < 25:
            congestion = 'heavy'
        elif speed < 40:
            congestion = 'moderate'
        else:
            congestion = 'light'
        
        return {
            'segment_id': segment_id,
            'prediction_time': datetime.now(),
            'target_time': target_time,
            'horizon_minutes': horizon,
            'predicted_speed': round(speed, 2),
            'predicted_congestion': congestion,
            'confidence_score': round(0.7 - horizon/200, 3),
            'model_type': 'fallback_heuristic',
            'is_real_prediction': False
        }
    
    def generate_all_predictions(self) -> List[Dict]:
        """Generate predictions for all segments and horizons."""
        predictions = []
        
        for segment_id in SEGMENTS:
            for horizon in HORIZONS:
                pred = self.predict(segment_id, horizon)
                predictions.append(pred)
        
        return predictions
    
    def save_predictions_to_db(self, predictions: List[Dict]):
        """Save predictions to PostgreSQL."""
        conn = self._get_db_connection()
        
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
                '{}'
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
            conn.rollback()
        finally:
            conn.close()


# Singleton instance
ml_predictor = RealMLPredictor()


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Real ML Prediction Service Test")
    print("=" * 60)
    
    predictor = RealMLPredictor()
    
    # Train on real data
    success = predictor.train(force=True)
    
    if success:
        # Make predictions
        print("\n📊 Sample Predictions:")
        for seg in ['SEG001', 'SEG003', 'SEG008']:
            pred = predictor.predict(seg, horizon_minutes=15)
            print(f"  {seg}: {pred['predicted_speed']:.1f} km/h, "
                  f"{pred['predicted_congestion']}, "
                  f"conf={pred['confidence_score']:.2f}, "
                  f"real={pred['is_real_prediction']}")
