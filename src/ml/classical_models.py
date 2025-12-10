"""
Classical ML Models for Traffic Prediction
- Speed Prediction: XGBoost Regressor
- Congestion Classification: Random Forest Classifier
- Optimization: RandomizedSearchCV with TimeSeriesSplit
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

class TrafficClassicalModels:
    def __init__(self, model_dir: str = "models/classical"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.speed_model = None
        self.congestion_model = None
        
    def train_speed_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train XGBoost Regressor for speed prediction.
        Target: R² > 0.50
        """
        print("🚀 Training XGBoost Speed Model...")
        
        # TimeSeriesSplit (No shuffle!)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Hyperparameters to optimize
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=20,
            scoring='r2',
            cv=tscv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.speed_model = search.best_estimator_
        best_score = search.best_score_
        print(f"✅ Best XGBoost R²: {best_score:.4f}")
        print(f"   Params: {search.best_params_}")
        
        # Final evaluation on last split
        train_index, test_index = list(tscv.split(X))[-1]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        y_pred = self.speed_model.predict(X_test)
        
        final_r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"📊 Final Validation R²: {final_r2:.4f}")
        
        self._save_model(self.speed_model, "xgboost_speed.pkl")
        
        return {'r2': final_r2, 'mse': mse, 'best_params': search.best_params_}

    def train_congestion_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train Random Forest for congestion classification.
        Target: Accuracy > 0.80
        """
        print("🚀 Training Random Forest Congestion Model...")
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            scoring='accuracy',
            cv=tscv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.congestion_model = search.best_estimator_
        best_score = search.best_score_
        print(f"✅ Best RF Accuracy: {best_score:.4f}")
        
        # Final evaluation
        train_index, test_index = list(tscv.split(X))[-1]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        y_pred = self.congestion_model.predict(X_test)
        
        final_acc = accuracy_score(y_test, y_pred)
        print(f"📊 Final Validation Accuracy: {final_acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        self._save_model(self.congestion_model, "rf_congestion.pkl")
        
        return {'accuracy': final_acc, 'best_params': search.best_params_}

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using both models."""
        if not self.speed_model or not self.congestion_model:
            self.load_models()
            
        speed_pred = self.speed_model.predict(X)
        cong_pred = self.congestion_model.predict(X)
        cong_proba = self.congestion_model.predict_proba(X)
        
        return {
            'speed': speed_pred,
            'congestion': cong_pred,
            'confidence': np.max(cong_proba, axis=1)
        }

    def _save_model(self, model, filename: str):
        with open(os.path.join(self.model_dir, filename), "wb") as f:
            pickle.dump(model, f)
            
    def load_models(self):
        try:
            with open(os.path.join(self.model_dir, "xgboost_speed.pkl"), "rb") as f:
                self.speed_model = pickle.load(f)
            with open(os.path.join(self.model_dir, "rf_congestion.pkl"), "rb") as f:
                self.congestion_model = pickle.load(f)
            print("✅ Classical models loaded.")
        except FileNotFoundError:
            print("⚠️ Models not found.")
