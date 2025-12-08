"""
LSTM Model for Long-Term Traffic Predictions
Uses historical sequence data to predict future traffic conditions.
"""
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from typing import List, Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM model will run in dummy mode.")

MODEL_DIR = '/app/models' if os.path.exists('/app/models') else './models'
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_lstm.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')

class TrafficLSTM:
    def __init__(self, sequence_length: int = 12):
        """
        Initialize LSTM model.
        Args:
            sequence_length: Number of past time steps to use (e.g., 12 steps of 5 mins = 1 hour)
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
        self._load_model()

    def _load_model(self):
        """Load trained model and scaler if they exist."""
        if not TF_AVAILABLE:
            return

        try:
            if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = load_model(LSTM_MODEL_PATH)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("✅ LSTM model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load LSTM model: {e}")

    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM architecture."""
        if not TF_AVAILABLE:
            return

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)  # Predict speed
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def prepare_data(self, data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM."""
        dataset = np.array(data).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)

    def train(self, historical_speeds: List[float], epochs: int = 10, batch_size: int = 32):
        """Train the LSTM model."""
        if not TF_AVAILABLE:
            print("❌ TensorFlow missing, skipping training")
            return False

        print(f"🧠 Training LSTM on {len(historical_speeds)} data points...")
        
        # Prepare data
        X, y = self.prepare_data(historical_speeds)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build model if not exists
        if self.model is None:
            self.build_model((X.shape[1], 1))
            
        # Train
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
        
        # Save
        self.model.save(LSTM_MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        self.is_trained = True
        print("✅ LSTM training complete")
        return True

    def predict(self, recent_speeds: List[float], steps_ahead: int = 1) -> float:
        """
        Predict future speed.
        Args:
            recent_speeds: List of recent speed values (must be >= sequence_length)
            steps_ahead: How many steps ahead to predict
        """
        if not self.is_trained or not TF_AVAILABLE:
            # Fallback: simple moving average
            return sum(recent_speeds[-5:]) / 5 if recent_speeds else 30.0

        if len(recent_speeds) < self.sequence_length:
            # Not enough data
            return recent_speeds[-1]

        # Prepare input
        input_seq = np.array(recent_speeds[-self.sequence_length:]).reshape(-1, 1)
        scaled_seq = self.scaler.transform(input_seq)
        X_input = np.reshape(scaled_seq, (1, self.sequence_length, 1))
        
        # Predict
        predicted_scaled = self.model.predict(X_input, verbose=0)
        predicted_speed = self.scaler.inverse_transform(predicted_scaled)[0][0]
        
        return float(predicted_speed)
