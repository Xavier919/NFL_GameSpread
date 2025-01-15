import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from typing import Tuple, List, Dict
import logging

class SpreadPredictor:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.model = LinearRegression()
        
    def prepare_features(self, feature_vectors: List[Dict], targets: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert feature dictionaries to numpy arrays"""
        all_keys = set().union(*[d.keys() for d in feature_vectors])
        all_keys = sorted(all_keys)
        
        X = np.array([[fv.get(key, 0) for key in all_keys] for fv in feature_vectors])
        y = np.array(targets)
        
        return X, y, all_keys
        
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform cross-validation and return metrics"""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        metrics = {
            'mse_scores': [],
            'mae_scores': [],
            'r2_scores': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            metrics['mse_scores'].append(mean_squared_error(y_val, y_pred))
            metrics['mae_scores'].append(mean_absolute_error(y_val, y_pred))
            metrics['r2_scores'].append(r2_score(y_val, y_pred))
            
        return metrics 