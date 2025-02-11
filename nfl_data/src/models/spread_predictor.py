import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from typing import Tuple, List, Dict
import logging
from nfl_data.src.processors.feature_engineering import FeatureEngineer
import statsmodels.api as sm

class SpreadPredictor:
    def __init__(self, n_splits: int = 5, random_state: int = 42, n_games: int = 10, alpha: float = 1.0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_games = n_games
        self.logger = logging.getLogger(__name__)
        self.model = Ridge(alpha=alpha)
        
    def get_feature_columns(self) -> List[str]:
        """Return list of column names to use as features"""
        return [
            # Team performance metrics
            'home_ypp', 'away_ypp',
            'home_completion_pct', 'away_completion_pct',
            'home_critical_down_rate', 'away_critical_down_rate',
            'home_turnover_differential', 'away_turnover_differential',
            
            # Historical performance
            #'home_historical_win_pct', 'away_historical_win_pct',
            'home_historical_points_for', 'away_historical_points_for',
            'home_historical_points_against', 'away_historical_points_against',
            
            'home_sack_differential', 'away_sack_differential',
            #'home_strength_of_schedule', 'away_strength_of_schedule',
            'home_interaction_term', 'away_interaction_term',
            'home_defense_ypp', 'away_defense_ypp'
        ]
        
    def prepare_features(self, feature_vectors: pd.DataFrame, targets: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert feature DataFrame to numpy arrays and handle missing values"""
        if targets is not None:
            # Create a DataFrame with both features and target for easier handling
            data = pd.concat([feature_vectors, targets.rename('spread')], axis=1)
        else:
            data = feature_vectors.copy()
        
        # Remove rows with NaN values in either features or target
        data_clean = data.dropna()
        
        # Print statistics about removed rows
        removed_count = len(data) - len(data_clean)
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} rows containing NaN values")
        
        # Split back into features and target if targets were provided
        feature_names = self.get_feature_columns()
        X = data_clean[feature_names].to_numpy()
        y = data_clean['spread'].to_numpy() if targets is not None else None
        
        if targets is not None:
            return X, y, feature_names
        else:
            return X, feature_names
        
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
            
            feature_names = self.get_feature_columns()
            coefficients = self.analyze_coefficients(feature_names, X_train, y_train)
        
        return metrics

    def analyze_coefficients(self, feature_names: List[str], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Analyze and return the coefficients of the linear regression model along with p-values"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model has not been fitted yet")
        
        # Add a constant to the feature matrix for OLS
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        coefficients = {}
        for name, coef, pval in zip(feature_names, model.params[1:], model.pvalues[1:]):
            coefficients[name] = {'coef': coef, 'pval': pval}
        
        # Add intercept if it exists
        coefficients['intercept'] = {'coef': model.params[0], 'pval': model.pvalues[0]}
        
        return coefficients
