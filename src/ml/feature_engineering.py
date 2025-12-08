"""
Feature Engineering Pipeline for Traffic Prediction Models
Generates 45+ features from raw traffic, weather, and temporal data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    sequence_length: int = 12  # 12 time steps for LSTM
    prediction_horizon: int = 6  # Predict 6 steps ahead
    aggregation_window: str = "5min"

class TrafficFeatureEngineer:
    """
    Generates features for traffic prediction models.
    Target: 45+ features as specified in the requirements.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp.
        Features: hour, day_of_week, is_weekend, is_rush_hour, etc.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Derived temporal features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
        
        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def extract_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract traffic-specific features.
        Features: speed stats, congestion metrics, vehicle counts.
        """
        df = df.copy()
        
        # Speed-based features
        df['speed_normalized'] = df['avg_speed'] / 100  # Normalize by max expected speed
        df['speed_log'] = np.log1p(df['avg_speed'])
        
        # Congestion encoding
        congestion_map = {'light': 0, 'moderate': 1, 'heavy': 2, 'severe': 3}
        df['congestion_encoded'] = df['congestion_level'].map(congestion_map).fillna(0)
        
        # Vehicle density
        df['vehicle_density'] = df['vehicle_count'] / (df.get('lanes', 4) * 1000)  # per km per lane
        
        # Stopped ratio
        df['stopped_ratio'] = df['stopped_vehicles'] / df['vehicle_count'].replace(0, 1)
        
        # Transport mode distribution
        total_vehicles = df['vehicle_count'].replace(0, 1)
        df['gbaka_ratio'] = df.get('gbaka_count', 0) / total_vehicles
        df['bus_ratio'] = df.get('bus_count', 0) / total_vehicles
        df['taxi_ratio'] = df.get('taxi_count', 0) / total_vehicles
        
        return df
    
    def extract_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract weather impact features.
        Features: temperature, precipitation, visibility impact.
        """
        df = df.copy()
        
        if weather_df is not None and len(weather_df) > 0:
            # Merge weather data (assuming nearest station match)
            weather_cols = ['temperature', 'humidity', 'precipitation', 'visibility', 'condition']
            for col in weather_cols:
                if col in weather_df.columns:
                    df[f'weather_{col}'] = weather_df[col].values[-1] if len(weather_df) > 0 else 0
        else:
            # Default values if no weather data
            df['weather_temperature'] = 28.0
            df['weather_humidity'] = 70.0
            df['weather_precipitation'] = 0.0
            df['weather_visibility'] = 10.0
            df['weather_condition'] = 'sunny'
        
        # Weather impact features
        df['is_raining'] = (df['weather_precipitation'] > 0).astype(int)
        df['heavy_rain'] = (df['weather_precipitation'] > 10).astype(int)
        df['poor_visibility'] = (df['weather_visibility'] < 3).astype(int)
        
        # Weather condition encoding
        condition_map = {
            'sunny': 0, 'cloudy': 1, 'light_rain': 2, 
            'heavy_rain': 3, 'thunderstorm': 4, 'foggy': 5
        }
        df['weather_condition_encoded'] = df['weather_condition'].map(condition_map).fillna(0)
        
        return df
    
    def extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spatial/segment features.
        Features: segment type, bridge flag, lane count.
        """
        df = df.copy()
        
        # Segment type encoding
        segment_type_map = {
            'bridge': 0, 'boulevard': 1, 'highway': 2, 
            'urban': 3, 'expressway': 4
        }
        df['segment_type_encoded'] = df.get('road_type', 'urban').map(segment_type_map).fillna(3)
        
        # Bridge flag
        df['is_bridge'] = df.get('is_bridge', False).astype(int)
        
        # Normalize lane count
        df['lanes_normalized'] = df.get('lanes', 4) / 6
        
        # Speed limit ratio
        df['speed_to_limit_ratio'] = df['avg_speed'] / df.get('max_speed_limit', 60).replace(0, 60)
        
        return df
    
    def extract_lag_features(self, df: pd.DataFrame, target_col: str = 'avg_speed') -> pd.DataFrame:
        """
        Extract lag features for time series prediction.
        Features: lag_1, lag_2, ..., lag_n, rolling averages.
        """
        df = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'{target_col}_lag_{lag}'] = df.groupby('segment_id')[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby('segment_id')[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby('segment_id')[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
        
        # Rate of change
        df[f'{target_col}_diff_1'] = df.groupby('segment_id')[target_col].diff(1)
        df[f'{target_col}_diff_3'] = df.groupby('segment_id')[target_col].diff(3)
        
        # Exponential moving average
        df[f'{target_col}_ema_6'] = (
            df.groupby('segment_id')[target_col]
            .transform(lambda x: x.ewm(span=6, adjust=False).mean())
        )
        
        return df
    
    def create_full_feature_set(self, 
                                 traffic_df: pd.DataFrame,
                                 weather_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create the complete feature set for model training.
        Returns DataFrame with features and list of feature column names.
        """
        df = traffic_df.copy()
        
        # Apply all feature extractors
        df = self.extract_temporal_features(df)
        df = self.extract_traffic_features(df)
        df = self.extract_weather_features(df, weather_df)
        df = self.extract_spatial_features(df)
        df = self.extract_lag_features(df)
        
        # Define feature columns (45+ features)
        feature_columns = [
            # Temporal (16 features)
            'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
            'is_night', 'is_lunch_hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            
            # Traffic (10 features)
            'speed_normalized', 'speed_log', 'congestion_encoded', 'vehicle_density',
            'stopped_ratio', 'gbaka_ratio', 'bus_ratio', 'taxi_ratio',
            'vehicle_count', 'stopped_vehicles',
            
            # Weather (8 features)
            'weather_temperature', 'weather_humidity', 'weather_precipitation',
            'weather_visibility', 'weather_condition_encoded', 'is_raining',
            'heavy_rain', 'poor_visibility',
            
            # Spatial (4 features)
            'segment_type_encoded', 'is_bridge', 'lanes_normalized', 'speed_to_limit_ratio',
            
            # Lag features (13 features)
            'avg_speed_lag_1', 'avg_speed_lag_2', 'avg_speed_lag_3',
            'avg_speed_lag_6', 'avg_speed_lag_12',
            'avg_speed_rolling_mean_3', 'avg_speed_rolling_mean_6', 'avg_speed_rolling_mean_12',
            'avg_speed_rolling_std_3', 'avg_speed_rolling_std_6', 'avg_speed_rolling_std_12',
            'avg_speed_diff_1', 'avg_speed_ema_6',
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Fill NaN values
        df[available_features] = df[available_features].fillna(0)
        
        return df, available_features
    
    def prepare_lstm_sequences(self, 
                               df: pd.DataFrame, 
                               feature_columns: List[str],
                               target_column: str = 'avg_speed') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        Returns X (sequences) and y (targets).
        """
        sequences = []
        targets = []
        
        for segment_id in df['segment_id'].unique():
            segment_data = df[df['segment_id'] == segment_id].sort_values('timestamp')
            
            X = segment_data[feature_columns].values
            y = segment_data[target_column].values
            
            for i in range(len(X) - self.config.sequence_length - self.config.prediction_horizon + 1):
                sequences.append(X[i:i + self.config.sequence_length])
                targets.append(y[i + self.config.sequence_length + self.config.prediction_horizon - 1])
        
        return np.array(sequences), np.array(targets)


def main():
    """Demo feature engineering."""
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'segment_id': ['SEG001'] * 100,
        'avg_speed': np.random.uniform(20, 80, 100),
        'vehicle_count': np.random.randint(10, 100, 100),
        'stopped_vehicles': np.random.randint(0, 20, 100),
        'congestion_level': np.random.choice(['light', 'moderate', 'heavy', 'severe'], 100),
        'gbaka_count': np.random.randint(0, 30, 100),
        'bus_count': np.random.randint(0, 10, 100),
        'taxi_count': np.random.randint(0, 20, 100),
    })
    
    engineer = TrafficFeatureEngineer()
    featured_df, feature_cols = engineer.create_full_feature_set(sample_data)
    
    print(f"Total features created: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"DataFrame shape: {featured_df.shape}")

if __name__ == "__main__":
    main()
