"""
Anomaly Detection for Traffic Data
Implements statistical and ML-based methods for detecting traffic anomalies.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of traffic anomalies."""
    SEVERE_CONGESTION = "severe_congestion"
    SUDDEN_SPEED_DROP = "sudden_speed_drop"
    MASS_STOP = "mass_stop"
    UNUSUAL_PATTERN = "unusual_pattern"
    POTENTIAL_ACCIDENT = "potential_accident"
    ABNORMAL_DENSITY = "abnormal_density"

@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    segment_id: str
    segment_name: str
    timestamp: datetime
    severity: int  # 1-5
    confidence: float  # 0-1
    details: Dict[str, Any]
    recommended_action: str

class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection.
    Uses Z-score and IQR methods for baseline anomaly detection.
    """
    
    def __init__(self, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def compute_baseline(self, historical_data: pd.DataFrame, 
                         segment_col: str = 'segment_id',
                         value_col: str = 'avg_speed') -> None:
        """Compute baseline statistics per segment."""
        for segment in historical_data[segment_col].unique():
            segment_data = historical_data[historical_data[segment_col] == segment][value_col]
            
            self.baseline_stats[segment] = {
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'q1': segment_data.quantile(0.25),
                'q3': segment_data.quantile(0.75),
                'median': segment_data.median(),
                'min': segment_data.min(),
                'max': segment_data.max(),
            }
            
            # Compute IQR bounds
            iqr = self.baseline_stats[segment]['q3'] - self.baseline_stats[segment]['q1']
            self.baseline_stats[segment]['lower_bound'] = (
                self.baseline_stats[segment]['q1'] - self.iqr_multiplier * iqr
            )
            self.baseline_stats[segment]['upper_bound'] = (
                self.baseline_stats[segment]['q3'] + self.iqr_multiplier * iqr
            )
    
    def detect_zscore_anomaly(self, segment_id: str, value: float) -> Tuple[bool, float, str]:
        """Detect anomaly using Z-score method."""
        if segment_id not in self.baseline_stats:
            return False, 0.0, "unknown_segment"
        
        stats = self.baseline_stats[segment_id]
        if stats['std'] == 0:
            return False, 0.0, "zero_variance"
        
        zscore = abs((value - stats['mean']) / stats['std'])
        is_anomaly = zscore > self.zscore_threshold
        
        direction = "low" if value < stats['mean'] else "high"
        
        return is_anomaly, zscore, direction
    
    def detect_iqr_anomaly(self, segment_id: str, value: float) -> Tuple[bool, str]:
        """Detect anomaly using IQR method."""
        if segment_id not in self.baseline_stats:
            return False, "unknown_segment"
        
        stats = self.baseline_stats[segment_id]
        
        if value < stats['lower_bound']:
            return True, "below_lower_bound"
        elif value > stats['upper_bound']:
            return True, "above_upper_bound"
        
        return False, "normal"

class MLAnomalyDetector:
    """
    ML-based anomaly detection using Isolation Forest.
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def train(self, data: pd.DataFrame, 
              segment_col: str = 'segment_id',
              feature_cols: List[str] = None) -> None:
        """Train Isolation Forest models per segment."""
        if feature_cols is None:
            feature_cols = ['avg_speed', 'vehicle_count', 'stopped_ratio']
        
        for segment in data[segment_col].unique():
            segment_data = data[data[segment_col] == segment][feature_cols].dropna()
            
            if len(segment_data) < 100:
                logger.warning(f"Insufficient data for segment {segment}, skipping...")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(segment_data)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            self.models[segment] = model
            self.scalers[segment] = scaler
            
        logger.info(f"Trained ML anomaly detectors for {len(self.models)} segments")
    
    def detect(self, segment_id: str, features: np.ndarray) -> Tuple[bool, float]:
        """Detect anomaly using Isolation Forest."""
        if segment_id not in self.models:
            return False, 0.0
        
        X_scaled = self.scalers[segment_id].transform(features.reshape(1, -1))
        prediction = self.models[segment_id].predict(X_scaled)
        score = self.models[segment_id].decision_function(X_scaled)[0]
        
        is_anomaly = prediction[0] == -1
        confidence = max(0, min(1, -score))  # Convert score to confidence
        
        return is_anomaly, confidence

class TrafficAnomalyEngine:
    """
    Main anomaly detection engine combining statistical and ML methods.
    """
    
    def __init__(self):
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()
        self.alert_rules = self._define_alert_rules()
        
    def _define_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define rules for anomaly classification and alerting."""
        return {
            'severe_congestion': {
                'condition': lambda d: d['avg_speed'] < 5 and d['vehicle_count'] > 20,
                'severity': 5,
                'action': "Activer itinéraires alternatifs et notifier les autorités"
            },
            'sudden_speed_drop': {
                'condition': lambda d: d.get('speed_change', 0) < -30,
                'severity': 4,
                'action': "Vérifier incident potentiel, alerter patrouille"
            },
            'mass_stop': {
                'condition': lambda d: d['stopped_ratio'] > 0.8,
                'severity': 4,
                'action': "Enquêter sur blocage, déployer agents de circulation"
            },
            'unusual_pattern': {
                'condition': lambda d: d.get('ml_anomaly', False),
                'severity': 3,
                'action': "Analyser pattern inhabituel, surveiller évolution"
            },
            'potential_accident': {
                'condition': lambda d: d['avg_speed'] < 3 and d['speed_variance'] > 20,
                'severity': 5,
                'action': "Alerte urgence, déployer secours, activer contournement"
            },
        }
    
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train all detection components."""
        logger.info("Training anomaly detection engine...")
        
        # Train statistical baseline
        self.statistical_detector.compute_baseline(historical_data)
        
        # Train ML models
        feature_cols = ['avg_speed', 'vehicle_count', 'stopped_vehicles', 
                       'speed_stddev', 'gbaka_count']
        available_cols = [c for c in feature_cols if c in historical_data.columns]
        self.ml_detector.train(historical_data, feature_cols=available_cols)
        
        logger.info("Anomaly detection engine trained successfully")
    
    def detect_anomalies(self, data_point: Dict[str, Any]) -> List[Anomaly]:
        """
        Detect all anomalies in a data point.
        Returns list of detected anomalies.
        """
        anomalies = []
        segment_id = data_point.get('segment_id', 'unknown')
        segment_name = data_point.get('segment_name', 'Unknown Segment')
        
        # Statistical detection
        zscore_anomaly, zscore, direction = self.statistical_detector.detect_zscore_anomaly(
            segment_id, data_point.get('avg_speed', 0)
        )
        
        iqr_anomaly, iqr_status = self.statistical_detector.detect_iqr_anomaly(
            segment_id, data_point.get('avg_speed', 0)
        )
        
        # ML detection
        if segment_id in self.ml_detector.models:
            features = np.array([
                data_point.get('avg_speed', 0),
                data_point.get('vehicle_count', 0),
                data_point.get('stopped_ratio', 0)
            ])
            ml_anomaly, ml_confidence = self.ml_detector.detect(segment_id, features)
            data_point['ml_anomaly'] = ml_anomaly
            data_point['ml_confidence'] = ml_confidence
        else:
            data_point['ml_anomaly'] = False
            data_point['ml_confidence'] = 0
        
        # Apply alert rules
        for rule_name, rule_config in self.alert_rules.items():
            try:
                if rule_config['condition'](data_point):
                    anomaly = Anomaly(
                        anomaly_id=f"ANO_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{rule_name[:4].upper()}",
                        anomaly_type=AnomalyType[rule_name.upper()],
                        segment_id=segment_id,
                        segment_name=segment_name,
                        timestamp=datetime.utcnow(),
                        severity=rule_config['severity'],
                        confidence=data_point.get('ml_confidence', 0.8) if 'ml' in rule_name else 0.9,
                        details={
                            'avg_speed': data_point.get('avg_speed'),
                            'vehicle_count': data_point.get('vehicle_count'),
                            'stopped_ratio': data_point.get('stopped_ratio'),
                            'zscore': zscore if zscore_anomaly else None,
                            'iqr_status': iqr_status if iqr_anomaly else None,
                        },
                        recommended_action=rule_config['action']
                    )
                    anomalies.append(anomaly)
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule_name}: {e}")
        
        return anomalies
    
    def process_stream(self, data_points: List[Dict[str, Any]]) -> List[Anomaly]:
        """Process a batch of data points for anomalies."""
        all_anomalies = []
        for point in data_points:
            anomalies = self.detect_anomalies(point)
            all_anomalies.extend(anomalies)
        return all_anomalies


def main():
    """Demo anomaly detection."""
    # Create sample historical data
    np.random.seed(42)
    n_samples = 1000
    
    historical_data = pd.DataFrame({
        'segment_id': np.random.choice(['SEG001', 'SEG002', 'SEG003'], n_samples),
        'avg_speed': np.random.normal(40, 10, n_samples),
        'vehicle_count': np.random.poisson(50, n_samples),
        'stopped_vehicles': np.random.poisson(5, n_samples),
        'speed_stddev': np.random.uniform(2, 8, n_samples),
        'gbaka_count': np.random.poisson(10, n_samples),
    })
    historical_data['stopped_ratio'] = historical_data['stopped_vehicles'] / historical_data['vehicle_count']
    
    # Train engine
    engine = TrafficAnomalyEngine()
    engine.train(historical_data)
    
    # Test with anomalous data point
    test_point = {
        'segment_id': 'SEG001',
        'segment_name': 'Pont HKB',
        'avg_speed': 3.0,  # Very low - anomaly
        'vehicle_count': 80,
        'stopped_vehicles': 70,
        'stopped_ratio': 0.875,
        'speed_stddev': 25,
        'speed_variance': 25,
    }
    
    anomalies = engine.detect_anomalies(test_point)
    
    print(f"\nDetected {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"  - {anomaly.anomaly_type.value} (Severity: {anomaly.severity})")
        print(f"    Action: {anomaly.recommended_action}")

if __name__ == "__main__":
    main()
