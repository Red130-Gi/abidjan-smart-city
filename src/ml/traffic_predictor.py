"""
Traffic Prediction Models - XGBoost + LSTM Ensemble
Implements ensemble learning for traffic speed prediction.
Target: MAE < 5 km/h
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
import json
from datetime import datetime
import logging

# XGBoost
import xgboost as xgb

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Scikit-learn utilities
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for prediction models."""
    # XGBoost params
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # LSTM params
    lstm_units: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_sequence_length: int = 12
    
    # Ensemble params
    xgb_weight: float = 0.4
    lstm_weight: float = 0.6

class XGBoostTrafficModel:
    """XGBoost model for traffic speed prediction."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_scaled, label=y)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.config.xgb_max_depth,
            'learning_rate': self.config.xgb_learning_rate,
            'subsample': self.config.xgb_subsample,
            'colsample_bytree': self.config.xgb_colsample_bytree,
            'eval_metric': 'mae',
            'seed': 42,
        }
        
        evals = [(dtrain, 'train')]
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            evals.append((dval, 'validation'))
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.xgb_n_estimators,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        # Store feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Compute metrics
        train_pred = self.model.predict(dtrain)
        metrics = {
            'train_mae': mean_absolute_error(y, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y, train_pred)),
            'train_r2': r2_score(y, train_pred),
        }
        
        if X_val is not None:
            val_pred = self.model.predict(dval)
            metrics.update({
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred),
            })
        
        logger.info(f"XGBoost training complete. Val MAE: {metrics.get('val_mae', metrics['train_mae']):.2f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        dmatrix = xgb.DMatrix(X_scaled)
        return self.model.predict(dmatrix)
    
    def save(self, path: str):
        """Save model and scaler."""
        self.model.save_model(f"{path}/xgboost_model.json")
        with open(f"{path}/xgboost_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}/xgboost_feature_importance.json", 'w') as f:
            json.dump(self.feature_importance, f)
    
    def load(self, path: str):
        """Load model and scaler."""
        self.model = xgb.Booster()
        self.model.load_model(f"{path}/xgboost_model.json")
        with open(f"{path}/xgboost_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

class LSTMTrafficModel:
    """LSTM model for sequential traffic prediction."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM architecture."""
        model = Sequential([
            LSTM(self.config.lstm_units, 
                 return_sequences=True if self.config.lstm_layers > 1 else False,
                 input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.config.lstm_dropout),
        ])
        
        # Additional LSTM layers
        for i in range(1, self.config.lstm_layers):
            return_seq = i < self.config.lstm_layers - 1
            model.add(LSTM(self.config.lstm_units // (2 ** i), return_sequences=return_seq))
            model.add(BatchNormalization())
            model.add(Dropout(self.config.lstm_dropout))
        
        # Output layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train LSTM model."""
        logger.info("Training LSTM model...")
        
        # Scale features
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        
        if X_val is not None:
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_reshaped).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Build model
        self.model = self.build_model((seq_len, n_features))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]
        
        # Train
        self.history = self.model.fit(
            X_scaled, y,
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Compute metrics
        train_pred = self.model.predict(X_scaled, verbose=0).flatten()
        metrics = {
            'train_mae': mean_absolute_error(y, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y, train_pred)),
            'train_r2': r2_score(y, train_pred),
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val_scaled, verbose=0).flatten()
            metrics.update({
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred),
            })
        
        logger.info(f"LSTM training complete. Val MAE: {metrics.get('val_mae', metrics['train_mae']):.2f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def save(self, path: str):
        """Save model and scaler."""
        self.model.save(f"{path}/lstm_model.keras")
        with open(f"{path}/lstm_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, path: str):
        """Load model and scaler."""
        self.model = load_model(f"{path}/lstm_model.keras")
        with open(f"{path}/lstm_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

class EnsembleTrafficPredictor:
    """
    Ensemble model combining XGBoost and LSTM.
    Uses weighted averaging for final predictions.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.xgb_model = XGBoostTrafficModel(config)
        self.lstm_model = LSTMTrafficModel(config)
        self.metrics = {}
        self.version = None
        
    def train(self, 
              X_tabular: np.ndarray, 
              X_sequential: np.ndarray, 
              y: np.ndarray,
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train both models on the data.
        X_tabular: 2D array for XGBoost (samples x features)
        X_sequential: 3D array for LSTM (samples x sequence_length x features)
        y: Target values (speed)
        """
        logger.info("Training Ensemble Traffic Predictor...")
        
        # Split data
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        X_tab_train, X_tab_val = X_tabular[train_idx], X_tabular[val_idx]
        X_seq_train, X_seq_val = X_sequential[train_idx], X_sequential[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train XGBoost
        xgb_metrics = self.xgb_model.train(X_tab_train, y_train, X_tab_val, y_val)
        
        # Train LSTM
        lstm_metrics = self.lstm_model.train(X_seq_train, y_train, X_seq_val, y_val)
        
        # Ensemble predictions on validation set
        xgb_pred = self.xgb_model.predict(X_tab_val)
        lstm_pred = self.lstm_model.predict(X_seq_val)
        
        ensemble_pred = (
            self.config.xgb_weight * xgb_pred + 
            self.config.lstm_weight * lstm_pred
        )
        
        ensemble_metrics = {
            'ensemble_val_mae': mean_absolute_error(y_val, ensemble_pred),
            'ensemble_val_rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred)),
            'ensemble_val_r2': r2_score(y_val, ensemble_pred),
        }
        
        self.metrics = {
            'xgboost': xgb_metrics,
            'lstm': lstm_metrics,
            'ensemble': ensemble_metrics,
            'trained_at': datetime.utcnow().isoformat(),
            'n_samples': len(y),
        }
        
        self.version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Ensemble training complete. Ensemble MAE: {ensemble_metrics['ensemble_val_mae']:.2f} km/h")
        
        return self.metrics
    
    def predict(self, X_tabular: np.ndarray, X_sequential: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        xgb_pred = self.xgb_model.predict(X_tabular)
        lstm_pred = self.lstm_model.predict(X_sequential)
        
        return (
            self.config.xgb_weight * xgb_pred + 
            self.config.lstm_weight * lstm_pred
        )
    
    def save(self, path: str):
        """Save all models and metadata."""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.xgb_model.save(path)
        self.lstm_model.save(path)
        
        with open(f"{path}/ensemble_metadata.json", 'w') as f:
            json.dump({
                'version': self.version,
                'metrics': self.metrics,
                'config': {
                    'xgb_weight': self.config.xgb_weight,
                    'lstm_weight': self.config.lstm_weight,
                }
            }, f, indent=2)
    
    def load(self, path: str):
        """Load all models and metadata."""
        self.xgb_model.load(path)
        self.lstm_model.load(path)
        
        with open(f"{path}/ensemble_metadata.json", 'r') as f:
            metadata = json.load(f)
            self.version = metadata['version']
            self.metrics = metadata['metrics']


def train_production_model(data_path: str, output_path: str):
    """
    Full training pipeline for production.
    """
    from feature_engineering import TrafficFeatureEngineer, FeatureConfig
    
    # Load data
    logger.info("Loading training data...")
    # In production, load from PostgreSQL or MongoDB
    # df = pd.read_parquet(data_path)
    
    # For demo, create synthetic data
    np.random.seed(42)
    n_samples = 10000
    n_features = 45
    seq_length = 12
    
    X_tabular = np.random.randn(n_samples, n_features)
    X_sequential = np.random.randn(n_samples, seq_length, n_features)
    y = 30 + 20 * np.random.randn(n_samples)  # Speed around 30 km/h
    y = np.clip(y, 0, 100)
    
    # Train ensemble
    config = ModelConfig()
    ensemble = EnsembleTrafficPredictor(config)
    metrics = ensemble.train(X_tabular, X_sequential, y)
    
    # Save model
    ensemble.save(output_path)
    
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Final metrics: {metrics}")
    
    return metrics


if __name__ == "__main__":
    train_production_model("data/training_data.parquet", "models/ensemble_v1")
