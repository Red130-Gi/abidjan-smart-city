"""
ML Prediction Service for Traffic Speed and Congestion
Uses XGBoost for short-term predictions and simulated LSTM for trends.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import os
import json

# ML Libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Database
import psycopg2
from psycopg2.extras import execute_values
import sys
sys.path.append('.')
from config.settings import postgres_config

class TrafficPredictionService:
    """
    ML-based traffic prediction service.
    - XGBoost for speed prediction (regression)
    - Random Forest for congestion classification
    - Simulated LSTM-like trend prediction
    """
    
    CONGESTION_LEVELS = ['light', 'moderate', 'heavy', 'severe']
    HORIZONS = [5, 15, 30, 60]  # minutes
    
    def __init__(self):
        self.speed_model = None
        self.congestion_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.CONGESTION_LEVELS)
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _get_db_connection(self):
        return psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based and historical features."""
        features = pd.DataFrame()
        
        # Time features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            features['is_rush_hour'] = ((features['hour'] >= 7) & (features['hour'] <= 9) | 
                                        (features['hour'] >= 17) & (features['hour'] <= 19)).astype(int)
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Segment encoding
        if 'segment_id' in df.columns:
            segment_map = {f'SEG00{i}': i for i in range(1, 9)}
            features['segment_encoded'] = df['segment_id'].map(segment_map).fillna(0)
        
        # Historical features
        if 'avg_speed' in df.columns:
            features['current_speed'] = df['avg_speed']
            features['speed_lag_1'] = df['avg_speed'].shift(1).fillna(df['avg_speed'].mean())
            features['speed_rolling_mean'] = df['avg_speed'].rolling(3, min_periods=1).mean()
        
        if 'vehicle_count' in df.columns:
            features['vehicle_count'] = df['vehicle_count']
            features['density'] = df['vehicle_count'] / 150  # Normalized
            
        if 'stopped_vehicles' in df.columns:
            features['stopped_ratio'] = df['stopped_vehicles'] / (df['vehicle_count'] + 1)
        
        return features.fillna(0)
    
    def load_training_data(self, hours: int = 24) -> pd.DataFrame:
        """Load historical traffic data for training."""
        conn = self._get_db_connection()
        query = f"""
            SELECT 
                segment_id,
                window_start as timestamp,
                avg_speed,
                vehicle_count,
                stopped_vehicles,
                congestion_level
            FROM traffic_segment_stats
            WHERE window_start > NOW() - INTERVAL '{hours} hours'
            ORDER BY window_start
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def train_models(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Train speed prediction and congestion classification models."""
        print("🚀 Starting model training...")
        
        if df is None:
            df = self.load_training_data(hours=48)
        
        if len(df) < 50:
            print("⚠️ Not enough data for training. Generating synthetic data...")
            df = self._generate_synthetic_data(1000)
        
        print(f"📊 Training on {len(df)} samples")
        
        # Extract features
        X = self.extract_features(df)
        
        # Speed prediction (regression)
        if 'avg_speed' in df.columns:
            y_speed = df['avg_speed'].values
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_speed, test_size=0.2, random_state=42
            )
            
            self.speed_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.speed_model.fit(X_train, y_train)
            
            y_pred = self.speed_model.predict(X_test)
            speed_mae = mean_absolute_error(y_test, y_pred)
            print(f"✅ Speed Model MAE: {speed_mae:.2f} km/h")
        
        # Congestion classification
        if 'congestion_level' in df.columns:
            y_cong = self.label_encoder.transform(df['congestion_level'].values)
            X_scaled = self.scaler.transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_cong, test_size=0.2, random_state=42
            )
            
            self.congestion_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.congestion_model.fit(X_train, y_train)
            
            y_pred = self.congestion_model.predict(X_test)
            cong_acc = accuracy_score(y_test, y_pred)
            print(f"✅ Congestion Model Accuracy: {cong_acc:.2%}")
        
        # Save models
        self._save_models()
        
        return {
            'speed_mae': float(speed_mae) if 'speed_mae' in locals() else None,
            'congestion_accuracy': float(cong_acc) if 'cong_acc' in locals() else None,
            'samples_used': len(df)
        }
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic training data."""
        np.random.seed(42)
        
        segments = [f'SEG00{i}' for i in range(1, 9)]
        data = []
        
        base_time = datetime.now() - timedelta(hours=48)
        
        for i in range(n_samples):
            segment = np.random.choice(segments)
            timestamp = base_time + timedelta(minutes=i * 5)
            hour = timestamp.hour
            
            # Simulate realistic traffic patterns
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_speed = np.random.uniform(15, 35)
            elif 10 <= hour <= 16:
                base_speed = np.random.uniform(35, 55)
            else:
                base_speed = np.random.uniform(50, 70)
            
            speed = max(5, base_speed + np.random.normal(0, 10))
            vehicle_count = int(np.random.uniform(20, 150))
            stopped = int(vehicle_count * np.random.uniform(0, 0.3))
            
            if speed < 10:
                congestion = 'severe'
            elif speed < 25:
                congestion = 'heavy'
            elif speed < 40:
                congestion = 'moderate'
            else:
                congestion = 'light'
            
            data.append({
                'segment_id': segment,
                'timestamp': timestamp,
                'avg_speed': speed,
                'vehicle_count': vehicle_count,
                'stopped_vehicles': stopped,
                'congestion_level': congestion
            })
        
        return pd.DataFrame(data)
    
    def predict(self, segment_id: str, current_data: Dict, 
                horizon_minutes: int = 15) -> Dict:
        """Make prediction for a specific segment and horizon."""
        
        # Try to use Ensemble Predictor first
        try:
            from .ensemble_model import ensemble_predictor
            return ensemble_predictor.predict(segment_id, horizon_minutes)
        except ImportError:
            pass

        if self.speed_model is None:
            self._load_models()
        
        # Prepare features
        now = datetime.now()
        target_time = now + timedelta(minutes=horizon_minutes)
        
        features = pd.DataFrame([{
            'hour': target_time.hour,
            'day_of_week': target_time.weekday(),
            'is_weekend': 1 if target_time.weekday() >= 5 else 0,
            'is_rush_hour': 1 if (7 <= target_time.hour <= 9 or 17 <= target_time.hour <= 19) else 0,
            'hour_sin': np.sin(2 * np.pi * target_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * target_time.hour / 24),
            'segment_encoded': int(segment_id[-1]) if segment_id.startswith('SEG') else 0,
            'current_speed': current_data.get('avg_speed', 40),
            'speed_lag_1': current_data.get('speed_lag', 40),
            'speed_rolling_mean': current_data.get('speed_rolling', 40),
            'vehicle_count': current_data.get('vehicle_count', 50),
            'density': current_data.get('vehicle_count', 50) / 150,
            'stopped_ratio': current_data.get('stopped_ratio', 0.1)
        }])
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Predict
        predicted_speed = float(self.speed_model.predict(X_scaled)[0])
        
        # Add horizon adjustment (further = more uncertainty)
        uncertainty = horizon_minutes / 60 * 5
        predicted_speed += np.random.uniform(-uncertainty, uncertainty)
        predicted_speed = max(5, min(80, predicted_speed))
        
        # Predict congestion
        congestion_proba = self.congestion_model.predict_proba(X_scaled)[0]
        predicted_congestion_idx = np.argmax(congestion_proba)
        predicted_congestion = self.label_encoder.inverse_transform([predicted_congestion_idx])[0]
        confidence = float(congestion_proba[predicted_congestion_idx])
        
        return {
            'segment_id': segment_id,
            'prediction_time': now.isoformat(),
            'target_time': target_time.isoformat(),
            'horizon_minutes': horizon_minutes,
            'predicted_speed': round(predicted_speed, 2),
            'predicted_congestion': predicted_congestion,
            'confidence': round(confidence, 3),
            'model_type': 'xgboost_ensemble'
        }
    
    def predict_all_segments(self, horizons: List[int] = None) -> List[Dict]:
        """Predict for all segments and multiple horizons."""
        if horizons is None:
            horizons = self.HORIZONS
        
        segments = [f'SEG00{i}' for i in range(1, 9)]
        predictions = []
        
        # Get current data
        current_data = self._get_current_segment_data()
        
        for segment in segments:
            seg_data = current_data.get(segment, {
                'avg_speed': 40,
                'vehicle_count': 50,
                'stopped_ratio': 0.1
            })
            
            for horizon in horizons:
                pred = self.predict(segment, seg_data, horizon)
                predictions.append(pred)
        
        return predictions
    
    def _get_current_segment_data(self) -> Dict:
        """Get current traffic data for all segments."""
        conn = self._get_db_connection()
        query = """
            SELECT DISTINCT ON (segment_id)
                segment_id,
                avg_speed,
                vehicle_count,
                stopped_vehicles::float / NULLIF(vehicle_count, 0) as stopped_ratio
            FROM traffic_segment_stats
            ORDER BY segment_id, window_start DESC
        """
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            return {row['segment_id']: row.to_dict() for _, row in df.iterrows()}
        except:
            conn.close()
            return {}
    
    def save_predictions_to_db(self, predictions: List[Dict]):
        """Save predictions to PostgreSQL."""
        conn = self._get_db_connection()
        cur = conn.cursor()
        
        values = [
            (
                p['segment_id'],
                p['prediction_time'],
                p['target_time'],
                p['horizon_minutes'],
                p['predicted_speed'],
                p['predicted_congestion'],
                p['confidence'],
                p['model_type'],
                json.dumps({})
            )
            for p in predictions
        ]
        
        execute_values(cur, """
            INSERT INTO traffic_predictions 
            (segment_id, prediction_time, target_time, horizon_minutes, 
             predicted_speed, predicted_congestion, confidence_score, model_type, features_used)
            VALUES %s
        """, values)
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"💾 Saved {len(predictions)} predictions to database")
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect predicted anomalies based on speed/congestion predictions."""
        predictions = self.predict_all_segments([15, 30, 60])
        anomalies = []
        
        for pred in predictions:
            # Severe congestion anomaly
            if pred['predicted_congestion'] == 'severe' and pred['confidence'] > 0.7:
                anomalies.append({
                    'segment_id': pred['segment_id'],
                    'prediction_time': pred['prediction_time'],
                    'expected_start': pred['target_time'],
                    'anomaly_type': 'congestion_spike',
                    'severity': 4,
                    'probability': pred['confidence'],
                    'description': f"Severe congestion predicted ({pred['predicted_speed']:.0f} km/h)",
                    'recommended_action': 'Recommend alternative routes'
                })
            
            # Speed drop anomaly
            elif pred['predicted_speed'] < 15 and pred['confidence'] > 0.6:
                anomalies.append({
                    'segment_id': pred['segment_id'],
                    'prediction_time': pred['prediction_time'],
                    'expected_start': pred['target_time'],
                    'anomaly_type': 'speed_drop',
                    'severity': 3,
                    'probability': pred['confidence'],
                    'description': f"Significant speed reduction expected ({pred['predicted_speed']:.0f} km/h)",
                    'recommended_action': 'Monitor closely, prepare diversions'
                })
        
        return anomalies
    
    def _save_models(self):
        """Save trained models to disk."""
        with open(f"{self.model_dir}/speed_model.pkl", 'wb') as f:
            pickle.dump(self.speed_model, f)
        with open(f"{self.model_dir}/congestion_model.pkl", 'wb') as f:
            pickle.dump(self.congestion_model, f)
        with open(f"{self.model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Models saved to {self.model_dir}/")
    
    def _load_models(self):
        """Load models from disk."""
        try:
            with open(f"{self.model_dir}/speed_model.pkl", 'rb') as f:
                self.speed_model = pickle.load(f)
            with open(f"{self.model_dir}/congestion_model.pkl", 'rb') as f:
                self.congestion_model = pickle.load(f)
            with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Models loaded successfully")
        except FileNotFoundError:
            print("⚠️ No saved models found. Training new models...")
            self.train_models()


# Singleton instance
prediction_service = TrafficPredictionService()


if __name__ == "__main__":
    service = TrafficPredictionService()
    
    # Train models
    metrics = service.train_models()
    print(f"\n📈 Training Metrics: {metrics}")
    
    # Make predictions
    predictions = service.predict_all_segments([5, 15, 30])
    print(f"\n🔮 Sample Predictions:")
    for p in predictions[:3]:
        print(f"  {p['segment_id']} @ +{p['horizon_minutes']}min: "
              f"{p['predicted_speed']:.1f} km/h ({p['predicted_congestion']})")
    
    # Detect anomalies
    anomalies = service.detect_anomalies()
    print(f"\n⚠️ Detected {len(anomalies)} potential anomalies")
