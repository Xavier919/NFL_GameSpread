import sys
from pathlib import Path
import pickle
import pandas as pd
import logging
import argparse
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from nfl_data.src.processors.feature_engineering import FeatureEngineer

# Setup logging at module level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    with open("models/spread_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_names"]

def load_recent_game_data():
    """Load the most recent season's data available"""
    dfs = []
    for year in range(2024, 2009, -1):
        try:
            file_path = f'data/processed/game_data/nfl_boxscore_{year}.csv'
            df = pd.read_csv(file_path)
            dfs.append(df)
        except FileNotFoundError:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else None

def get_team_stats(df: pd.DataFrame, team: str, n_games: int = 10) -> dict:
    """Calculate team statistics from their recent games"""
    # Get games where team was either home or away
    team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    team_games = team_games.sort_values('date').tail(n_games)
    
    if len(team_games) == 0:
        logger.error(f"No games found for {team}")
        return None
        
    # Calculate win percentage and points
    wins = len(team_games[
        ((team_games['home_team'] == team) & (team_games['home_score'] > team_games['away_score'])) |
        ((team_games['away_team'] == team) & (team_games['away_score'] > team_games['home_score']))
    ])
    win_pct = wins / len(team_games)
    
    # Calculate all required stats
    stats = {
        'win_pct': win_pct,
        'points_for': 0,
        'points_against': 0,
        'ypp': 0,
        'completion_pct': 0,
        'critical_down_rate': 0,
        'turnover_differential': 0
    }
    
    # Calculate stats when team was home vs away
    for _, game in team_games.iterrows():
        is_home = game['home_team'] == team
        
        # Points
        if is_home:
            stats['points_for'] += game['home_score']
            stats['points_against'] += game['away_score']
            my_turnovers = game['home_turnovers']
            opp_turnovers = game['away_turnovers']
            prefix = 'home_'
        else:
            stats['points_for'] += game['away_score']
            stats['points_against'] += game['home_score']
            my_turnovers = game['away_turnovers']
            opp_turnovers = game['home_turnovers']
            prefix = 'away_'
            
        # Calculate total plays (pass attempts + rush attempts + sacks)
        pass_attempts = game[f'{prefix}pass_attempts']
        rush_attempts = game[f'{prefix}rush_attempts']
        total_plays = pass_attempts + rush_attempts
        
        # Add up stats
        yards = game[f'{prefix}total_yds']
        stats['ypp'] += yards / total_plays if total_plays > 0 else 0
        
        completions = game[f'{prefix}pass_completions']
        stats['completion_pct'] += completions / pass_attempts if pass_attempts > 0 else 0
        
        critical_attempts = game[f'{prefix}third_down_attempts'] + game[f'{prefix}fourth_down_attempts']
        critical_conversions = game[f'{prefix}third_down_converted'] + game[f'{prefix}fourth_down_converted']
        stats['critical_down_rate'] += critical_conversions / critical_attempts if critical_attempts > 0 else 0
        
        stats['turnover_differential'] += opp_turnovers - my_turnovers
    
    # Convert sums to averages
    n = len(team_games)
    for stat in ['points_for', 'points_against', 'ypp', 'completion_pct', 'critical_down_rate']:
        stats[stat] /= n
    stats['turnover_differential'] /= n
    
    return stats

def predict_spread(home_team: str, away_team: str):
    # Load model and feature names
    try:
        model, feature_names = load_model()
    except FileNotFoundError:
        logger.error("Model file not found. Please train the model first.")
        return
        
    # Load recent game data
    recent_data = load_recent_game_data()
    if recent_data is None:
        logger.error("No game data found. Please ensure data is collected.")
        return
    
    # Get stats for both teams
    home_stats = get_team_stats(recent_data, home_team)
    away_stats = get_team_stats(recent_data, away_team)
    
    if home_stats is None or away_stats is None:
        return
    
    # Create feature vector with all required features
    features = pd.DataFrame([{
        'home_ypp': home_stats['ypp'],
        'away_ypp': away_stats['ypp'],
        'home_completion_pct': home_stats['completion_pct'],
        'away_completion_pct': away_stats['completion_pct'],
        'home_critical_down_rate': home_stats['critical_down_rate'],
        'away_critical_down_rate': away_stats['critical_down_rate'],
        'home_turnover_differential': home_stats['turnover_differential'],
        'away_turnover_differential': away_stats['turnover_differential'],
        'home_historical_win_pct': home_stats['win_pct'],
        'away_historical_win_pct': away_stats['win_pct'],
        'home_historical_points_for': home_stats['points_for'],
        'away_historical_points_for': away_stats['points_for'],
        'home_historical_points_against': home_stats['points_against'],
        'away_historical_points_against': away_stats['points_against']
    }])
    
    # Make prediction
    predicted_spread = model.predict(features)[0]
    
    # Print result
    display_prediction(home_team, away_team, predicted_spread)
    
    return predicted_spread

def display_prediction(home_team: str, away_team: str, predicted_spread: float):
    """Display the prediction in a clear format"""
    logger.info(f"\nPredicted spread for {home_team} (Home) vs {away_team} (Away): {predicted_spread:.1f}")
    
    if predicted_spread < 0:
        logger.info(f"{home_team} favored by {abs(predicted_spread):.1f} points")
    else:
        logger.info(f"{away_team} favored by {predicted_spread:.1f} points")

def main():
    parser = argparse.ArgumentParser(description='Predict NFL game spread')
    parser.add_argument('home_team', help='Home team name')
    parser.add_argument('away_team', help='Away team name')
    
    args = parser.parse_args()
    predict_spread(args.home_team, args.away_team)

if __name__ == "__main__":
    main() 