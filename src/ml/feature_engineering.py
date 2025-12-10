"""
Feature Engineering Pipeline for Traffic Prediction Models
Refactored to meet strict requirements:
- Cleaning: Remove negatives, interpolate
- Features: Temporal (hour, dayofweek, peak), Traffic (speed, flow)
- Labeling: Binary congestion (speed < 15)
- Scaling: StandardScaler with persistence
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class TrafficScaler:
    """Handles scaling of features and target variables."""
    
    def __init__(self, model_dir: str = "models/scalers"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, columns: List[str]):
        """Fit scaler on specified columns."""
        self.scaler.fit(df[columns])
        self.is_fitted = True
        self.save()
        
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transform specified columns."""
        if not self.is_fitted:
            self.load()
            
        df_scaled = df.copy()
        df_scaled[columns] = self.scaler.transform(df[columns])
        return df_scaled
    
    def save(self):
        """Save scaler to disk."""
        with open(os.path.join(self.model_dir, "traffic_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
            
    def load(self):
        """Load scaler from disk."""
        path = os.path.join(self.model_dir, "traffic_scaler.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
        else:
            print("⚠️ Scaler not found, must be fitted first.")

class TrafficFeatureEngineer:
    """
    Pipeline for cleaning, feature extraction, and preparation.
    """
    
    def __init__(self):
        self.scaler = TrafficScaler()
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data:
        - Remove duplicates
        - Sort by time
        - Remove negative/impossible speeds
        - Interpolate missing values
        """
        df = df.copy()
        
        # Sort
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['segment_id', 'timestamp'])
            
        # Remove impossible values
        if 'avg_speed' in df.columns:
            df.loc[df['avg_speed'] < 0, 'avg_speed'] = np.nan
            df.loc[df['avg_speed'] > 150, 'avg_speed'] = 150 # Cap at 150 km/h
            
        # Interpolate missing values per segment
        df['avg_speed'] = df.groupby('segment_id')['avg_speed'].transform(
            lambda x: x.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        )
        
        return df.dropna()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create essential features for ML models.
        """
        df = df.copy()
        
        # 1. Temporal Features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
            # Peak hours: 6-9 and 16-19
            df['peak'] = df['hour'].apply(
                lambda h: 1 if (6 <= h <= 9) or (16 <= h <= 19) else 0
            )
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 2. Traffic Features
        if 'vehicle_count' in df.columns:
            df['flow'] = df['vehicle_count'] # Alias for flow
            df['occupancy'] = df['vehicle_count'] / 200.0 # Normalized occupancy approximation
        
        # 3. Lag Features (for classical models)
        if 'avg_speed' in df.columns:
            for lag in [1, 3, 6, 12]: # 5m, 15m, 30m, 1h
                df[f'speed_lag_{lag}'] = df.groupby('segment_id')['avg_speed'].shift(lag)
            
            # Rolling stats
            df['speed_rolling_mean_3'] = df.groupby('segment_id')['avg_speed'].transform(
                lambda x: x.rolling(3).mean()
            )
            df['speed_rolling_std_3'] = df.groupby('segment_id')['avg_speed'].transform(
                lambda x: x.rolling(3).std()
            )

        # 4. Weather Features
        if 'precipitation' in df.columns:
            df['precipitation'] = df['precipitation'].fillna(0.0)
            df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
            # Interaction feature: Rain * Traffic
            if 'vehicle_count' in df.columns:
                df['rain_traffic_interaction'] = df['precipitation'] * df['vehicle_count']

        # 4. Target Labels
        # Congestion: 1 if speed < 15, else 0
        if 'avg_speed' in df.columns:
            df['congestion_label'] = (df['avg_speed'] < 15).astype(int)

        return df.fillna(0)

    def normalize_data(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Normalize continuous columns using TrafficScaler.
        """
        continuous_cols = [
            'avg_speed', 'occupancy', 'flow', 
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'precipitation', 'temperature', 'rain_traffic_interaction'
        ]
        
        # Add lag columns if present
        lag_cols = [c for c in df.columns if 'speed_lag' in c or 'rolling' in c]
        cols_to_scale = continuous_cols + lag_cols
        
        # Filter only existing columns
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]
        
        if fit:
            self.scaler.fit(df, cols_to_scale)
            
        return self.scaler.transform(df, cols_to_scale)
