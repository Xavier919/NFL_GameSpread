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
        
    def get_feature_columns(self) -> List[str]:
        """Return list of column names to use as features"""
        return [
            # Team performance metrics
            'home_ypp', 'away_ypp',
            'home_completion_pct', 'away_completion_pct',
            'home_critical_down_rate', 'away_critical_down_rate',
            'home_turnover_differential', 'away_turnover_differential',
            
            # Historical performance
            'home_historical_win_pct', 'away_historical_win_pct',
            'home_historical_points_for', 'away_historical_points_for',
            'home_historical_points_against', 'away_historical_points_against',
            

        ]
        
    def prepare_features(self, feature_vectors: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert feature DataFrame to numpy arrays and handle missing values"""
        # Create a DataFrame with both features and target for easier handling
        data = pd.concat([feature_vectors, targets.rename('spread')], axis=1)
        
        # Remove rows with NaN values in either features or target
        data_clean = data.dropna()
        
        # Print statistics about removed rows
        removed_count = len(data) - len(data_clean)
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} rows containing NaN values")
        
        # Print feature statistics
        #self.logger.info("\nFeature Statistics:")
        #for col in feature_vectors.columns:
        #    stats = data_clean[col].describe()
        #    self.logger.info(f"\n{col}:")
        #    self.logger.info(f"Min: {stats['min']:.2f}")
        #    self.logger.info(f"Max: {stats['max']:.2f}")
        #    self.logger.info(f"Mean: {stats['mean']:.2f}")
        #    
        #    # Check for infinity
        #    inf_count = np.isinf(data_clean[col]).sum()
        #    if inf_count > 0:
        #        self.logger.warning(f"Found {inf_count} infinity values in {col}")
        
        # Split back into features and target
        X = data_clean[feature_vectors.columns].to_numpy()
        y = data_clean['spread'].to_numpy()
        feature_names = feature_vectors.columns.tolist()
        
        return X, y, feature_names
        
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

    def analyze_coefficients(self, feature_names: List[str]) -> Dict[str, float]:
        """Analyze and return the coefficients of the linear regression model"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model has not been fitted yet")
        
        coefficients = {}
        for name, coef in zip(feature_names, self.model.coef_):
            coefficients[name] = coef
        
        # Add intercept if it exists
        if hasattr(self.model, 'intercept_'):
            coefficients['intercept'] = self.model.intercept_
        
        return coefficients 