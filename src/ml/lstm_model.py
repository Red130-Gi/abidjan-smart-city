"""
LSTM Model for Long-Term Traffic Predictions (Per Segment)
- Architecture: LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(1)
- Training: Per-segment models with EarlyStopping & ReduceLROnPlateau
"""
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM model will run in dummy mode.")

class TrafficLSTM:
    def __init__(self, model_dir: str = "models/lstm", window_size: int = 20):
        self.model_dir = model_dir
        self.window_size = window_size
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Dictionary to hold models per segment
        self.models: Dict[str, Sequential] = {}
        self.is_trained: Dict[str, bool] = {}
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM architecture:
        LSTM(64, return_sequences=True) -> LSTM(32) -> Dense(16, relu) -> Dense(1)
        """
        if not TF_AVAILABLE:
            return None

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Predict speed
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train_segment(self, segment_id: str, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50):
        """Train LSTM for a specific segment."""
        if not TF_AVAILABLE:
            return

        print(f"🧠 Training LSTM for {segment_id}...")
        
        # Build model if needed
        if segment_id not in self.models:
            self.models[segment_id] = self.build_model((X_train.shape[1], X_train.shape[2]))
            
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        ]
        
        # Train
        history = self.models[segment_id].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        val_loss = min(history.history['val_loss'])
        print(f"✅ {segment_id} Best Val Loss (MSE): {val_loss:.4f}")
        
        self.is_trained[segment_id] = True
        self._save_model(segment_id)
        
        return val_loss

    def predict(self, segment_id: str, recent_sequence: np.ndarray) -> float:
        """Predict next step for a segment."""
        if not TF_AVAILABLE or segment_id not in self.models:
            return 0.0
            
        # Reshape [1, window, features]
        if recent_sequence.ndim == 2:
            recent_sequence = np.expand_dims(recent_sequence, axis=0)
            
        pred = self.models[segment_id].predict(recent_sequence, verbose=0)
        return float(pred[0][0])

    def _save_model(self, segment_id: str):
        """Save segment model."""
        path = os.path.join(self.model_dir, f"{segment_id}_lstm.h5")
        self.models[segment_id].save(path)

    def load_models(self, segments: List[str]):
        """Load models for specified segments."""
        if not TF_AVAILABLE:
            return

        for seg in segments:
            path = os.path.join(self.model_dir, f"{seg}_lstm.h5")
            if os.path.exists(path):
                try:
                    self.models[seg] = load_model(path)
                    self.is_trained[seg] = True
                    print(f"✅ Loaded LSTM for {seg}")
                except Exception as e:
                    print(f"⚠️ Failed to load {seg}: {e}")
