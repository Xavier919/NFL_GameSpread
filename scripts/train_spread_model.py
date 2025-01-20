import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
import sys
import numpy as np
import pickle
import argparse

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

def display_feature_importance(coefficients: Dict[str, float], logger: logging.Logger):
    """Display the importance of each feature based on coefficients and p-values"""
    # Sort coefficients by absolute value to show most influential features
    sorted_coefs = sorted(
        [(k, v['coef'], v['pval']) for k, v in coefficients.items() if k != 'intercept'],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    logger.info("\nFeature Coefficients and P-values (sorted by importance):")
    logger.info("-" * 70)
    for feature, coef, pval in sorted_coefs:
        logger.info(f"{feature:30} {coef:>10.4f} {pval:>10.4f}")
        
    if 'intercept' in coefficients:
        logger.info("-" * 70)
        logger.info(f"{'Intercept':30} {coefficients['intercept']['coef']:>10.4f} {coefficients['intercept']['pval']:>10.4f}")

def find_available_years(start_year: int = 2010, end_year: int = 2024) -> List[int]:
    """Find all years that have available data files"""
    available_years = []
    for year in range(start_year, end_year + 1):
        file_path = Path(f'data/processed/game_data/nfl_boxscore_{year}.csv')
        if file_path.exists():
            available_years.append(year)
    return available_years

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train NFL spread prediction model')
    parser.add_argument('--n-games', type=int, default=10,
                      help='Number of historical games to use for feature engineering (default: 10)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Find and load all available game data
    available_years = find_available_years()
    logger.info(f"Found data for years: {available_years}")
    logger.info(f"Using {args.n_games} historical games for feature engineering")
    
    game_data = load_game_data(available_years)
    
    if game_data is None:
        logger.error("No game data loaded")
        return
        
    # Calculate actual point differential and other features all at once
    new_columns = {
        'point_differential': -(game_data['home_score'] - game_data['away_score'])
    }

    # Create new DataFrame with all columns at once
    game_data = pd.concat([
        game_data,
        pd.DataFrame(new_columns, index=game_data.index)
    ], axis=1)
    
    # Initialize feature engineer with specified n_games
    engineer = FeatureEngineer(n_games=args.n_games)
    processed_data = engineer.compute_all_features(game_data)
    
    # Initialize and train model
    predictor = SpreadPredictor()
    # Use point_differential instead of spread as target
    feature_vectors, targets = processed_data[predictor.get_feature_columns()], processed_data['point_differential']
    
    # Train and evaluate model
    X, y, feature_names = predictor.prepare_features(feature_vectors, targets)
    metrics = predictor.train_and_evaluate(X, y)
    
    # Log results
    logger.info(f"Average MSE: {np.mean(metrics['mse_scores']):.2f}")
    logger.info(f"Average MAE: {np.mean(metrics['mae_scores']):.2f}")
    logger.info(f"Average R2: {np.mean(metrics['r2_scores']):.3f}")
    
    # Train final model on all data
    predictor.model.fit(X, y)
    
    # Analyze coefficients
    coefficients = predictor.analyze_coefficients(feature_names, X, y)
    display_feature_importance(coefficients, logger)
    
    # Save model and feature names
    with open("models/spread_model.pkl", "wb") as f:
        pickle.dump({
            "model": predictor.model,
            "feature_names": feature_names,
            "n_games": args.n_games  # Save n_games with the model
        }, f)
    
    logger.info("Model saved to models/spread_model.pkl")

if __name__ == "__main__":
    main() 