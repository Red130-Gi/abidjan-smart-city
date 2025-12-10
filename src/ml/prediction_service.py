"""
ML Prediction Service (Orchestrator)
- Orchestrates the full ML pipeline: Data -> Features -> Training -> Prediction
- Manages Classical, LSTM, and Ensemble models.
- Provides unified prediction API.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import psycopg2
from psycopg2.extras import execute_values
import sys
sys.path.append('.')

from config.settings import postgres_config
from src.ml.feature_engineering import TrafficFeatureEngineer
from src.ml.ensemble_model import ensemble_predictor

class TrafficPredictionService:
    def __init__(self):
        self.engineer = TrafficFeatureEngineer()
        self.ensemble = ensemble_predictor
        self.segments = [f'SEG00{i}' for i in range(1, 9)]
        
    def _get_db_connection(self):
        return psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )

    def load_training_data(self, days: int = 7) -> pd.DataFrame:
        """Load historical data from DB with Weather."""
        print(f"📥 Loading {days} days of training data with Weather...")
        conn = self._get_db_connection()
        # Join traffic stats with weather data
        # We use date_trunc('hour', ...) to align them roughly, or a lateral join for nearest
        # For simplicity, let's assume weather is available hourly or we take the latest weather before the traffic bucket
        query = f"""
            SELECT 
                t.segment_id,
                t.window_start as timestamp,
                t.avg_speed,
                t.vehicle_count,
                t.stopped_vehicles,
                t.congestion_level,
                w.precipitation,
                w.temperature
            FROM traffic_segment_stats t
            LEFT JOIN LATERAL (
                SELECT precipitation, temperature
                FROM weather_data w
                WHERE w.recorded_at <= t.window_start
                ORDER BY w.recorded_at DESC
                LIMIT 1
            ) w ON TRUE
            WHERE t.window_start > NOW() - INTERVAL '{days} days'
            ORDER BY t.segment_id, t.window_start
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def train_full_pipeline(self):
        """
        Execute the complete training pipeline.
        1. Load & Clean Data
        2. Feature Engineering
        3. Train Classical Models
        4. Train LSTM Models
        5. Train Ensemble Meta-Model
        """
        print("\n🚀 STARTING FULL ML PIPELINE TRAINING 🚀")
        
        # 1. Load Data
        raw_df = self.load_training_data(days=2)
        if len(raw_df) < 100:
            print("⚠️ Not enough data. Skipping training.")
            return

        # 2. Preprocessing
        print("🛠️ Preprocessing data...")
        clean_df = self.engineer.clean_data(raw_df)
        featured_df = self.engineer.create_features(clean_df)
        
        # Split for TimeSeries validation (last 20% for validation)
        split_idx = int(len(featured_df) * 0.8)
        train_df = featured_df.iloc[:split_idx]
        val_df = featured_df.iloc[split_idx:]
        
        # Fit Scaler on Train only
        print("⚖️ Fitting Scaler...")
        self.engineer.normalize_data(train_df, fit=True)
        
        # Prepare X, y
        feature_cols = [c for c in train_df.columns if c not in ['avg_speed', 'congestion_level', 'timestamp', 'segment_id', 'congestion_label']]
        X_train = self.engineer.normalize_data(train_df)[feature_cols]
        y_train_speed = train_df['avg_speed']
        y_train_cong = train_df['congestion_label']
        
        X_val = self.engineer.normalize_data(val_df)[feature_cols]
        y_val_speed = val_df['avg_speed']
        
        # 3. Train Classical Models
        print("\n🤖 Training Classical Models...")
        self.ensemble.classical.train_speed_model(X_train, y_train_speed)
        self.ensemble.classical.train_congestion_model(X_train, y_train_cong)
        
        # 4. Train LSTM Models (Per Segment)
        print("\n🧠 Training LSTM Models...")
        for seg in self.segments:
            seg_train = train_df[train_df['segment_id'] == seg]
            seg_val = val_df[val_df['segment_id'] == seg]
            
            if len(seg_train) > 50:
                # Prepare sequences
                # Note: We use 'avg_speed' as the target for LSTM
                # In a real scenario, we might use multivariate input. 
                # For simplicity/robustness here, we use univariate speed sequences.
                X_seq_train, y_seq_train = self.ensemble.lstm.prepare_sequences(seg_train['avg_speed'].values)
                X_seq_val, y_seq_val = self.ensemble.lstm.prepare_sequences(seg_val['avg_speed'].values)
                
                if len(X_seq_train) > 0 and len(X_seq_val) > 0:
                    # Reshape for LSTM [samples, time steps, features]
                    X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], X_seq_train.shape[1], 1))
                    X_seq_val = X_seq_val.reshape((X_seq_val.shape[0], X_seq_val.shape[1], 1))
                    
                    self.ensemble.lstm.train_segment(seg, X_seq_train, y_seq_train, X_seq_val, y_seq_val)
        
        # 5. Train Ensemble Meta-Model
        print("\n🔗 Training Ensemble Meta-Model...")
        # We use the validation set to train the meta-model to avoid overfitting
        self.ensemble.train_meta_model(X_val, y_val_speed)
        
        print("\n✅ PIPELINE TRAINING COMPLETE")

    def predict_all(self, horizon_minutes: int = 15):
        """Generate predictions for all segments."""
        # Load current data
        current_df = self.load_training_data(days=1) # Need recent history for LSTM
        if len(current_df) == 0:
            return []
            
        current_df = self.engineer.clean_data(current_df)
        current_df = self.engineer.create_features(current_df)
        
        predictions = []
        now = datetime.now()
        target_time = now + timedelta(minutes=horizon_minutes)
        
        for seg in self.segments:
            seg_data = current_df[current_df['segment_id'] == seg].tail(1)
            if len(seg_data) == 0:
                continue
                
            # Prepare features for classical model
            # We need to shift 'hour' etc to target time
            future_features = seg_data.copy()
            future_features['hour'] = target_time.hour
            future_features['minute'] = target_time.minute
            # Re-create temporal features
            future_features = self.engineer.create_features(future_features) 
            
            feature_cols = [c for c in future_features.columns if c not in ['avg_speed', 'congestion_level', 'timestamp', 'segment_id', 'congestion_label']]
            X_input = self.engineer.normalize_data(future_features)[feature_cols]
            
            # Prepare sequence for LSTM
            seg_history = current_df[current_df['segment_id'] == seg]['avg_speed'].values
            if len(seg_history) >= 20:
                recent_seq = seg_history[-20:]
                recent_seq = recent_seq.reshape((1, 20, 1))
            else:
                recent_seq = np.zeros((1, 20, 1)) # Dummy
            
            # Predict
            pred = self.ensemble.predict(seg, X_input, recent_seq)
            
            predictions.append({
                'segment_id': seg,
                'prediction_time': now.isoformat(),
                'target_time': target_time.isoformat(),
                'horizon_minutes': horizon_minutes,
                'predicted_speed': pred['predicted_speed'],
                'predicted_congestion': pred['predicted_congestion'],
                'confidence': pred['confidence'],
                'model_type': 'ensemble_v2',
                'details': pred['details']
            })
            
        # Save to DB
        self.save_predictions(predictions)
        return predictions

    def save_predictions(self, predictions: List[Dict]):
        """Save to DB."""
        if not predictions:
            return
            
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
                json.dumps(p['details'])
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
        conn.close()
        print(f"💾 Saved {len(predictions)} predictions")

# Singleton
prediction_service = TrafficPredictionService()

if __name__ == "__main__":
    # Test run
    prediction_service.train_full_pipeline()
