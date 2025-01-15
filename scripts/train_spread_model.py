import pandas as pd
from pathlib import Path
import logging
from typing import List
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from nfl_data.src.processors.feature_engineering import FeatureEngineer
from nfl_data.src.models.spread_predictor import SpreadPredictor

def load_game_data(years: List[int]) -> pd.DataFrame:
    """Load and combine game data from multiple seasons"""
    dfs = []
    for year in years:
        try:
            file_path = f'data/processed/game_data/nfl_boxscore_{year}.csv'
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error loading data for {year}: {str(e)}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load all game data from 2018-2023
    years = range(2018, 2024)
    game_data = load_game_data(years)
    
    if game_data is None:
        logger.error("No game data loaded")
        return
        
    # Initialize feature engineer and compute features
    engineer = FeatureEngineer(n_games=10)
    processed_data = engineer.compute_all_features(game_data)
    
    # Initialize and train model
    predictor = SpreadPredictor()
    feature_vectors, targets = processed_data[predictor.get_feature_columns()], processed_data['spread']
    
    # Train and evaluate model
    X, y, feature_names = predictor.prepare_features(feature_vectors, targets)
    metrics = predictor.train_and_evaluate(X, y)
    
    # Log results
    logger.info(f"Average MSE: {np.mean(metrics['mse_scores']):.2f}")
    logger.info(f"Average MAE: {np.mean(metrics['mae_scores']):.2f}")
    logger.info(f"Average R2: {np.mean(metrics['r2_scores']):.3f}")

if __name__ == "__main__":
    main() 