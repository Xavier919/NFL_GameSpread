import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

class FeatureEngineer:
    def __init__(self, n_games: int = 10):
        self.n_games = n_games
        self.logger = logging.getLogger(__name__)
            
    def add_margin_of_victory(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['home_margin'] = df['home_score'] - df['away_score']
        df['away_margin'] = df['away_score'] - df['home_score']
        return df
        
    def add_yards_per_play(self, df):
        df = df.copy()
        
        # Home team YPP
        total_home_plays = df['home_rush_attempts'] + df['home_pass_attempts']
        total_home_yards = df['home_rush_yards'] + df['home_pass_yards']
        df['home_ypp'] = total_home_yards / total_home_plays.replace(0, 1)
        
        # Away team YPP
        total_away_plays = df['away_rush_attempts'] + df['away_pass_attempts']
        total_away_yards = df['away_rush_yards'] + df['away_pass_yards']
        df['away_ypp'] = total_away_yards / total_away_plays.replace(0, 1)
        
        return df
    
    def add_completion_pct(self, df):
        df = df.copy()
        
        df['home_completion_pct'] = (df['home_pass_completions'] / 
                                   df['home_pass_attempts'].replace(0, 1)) * 100
        df['away_completion_pct'] = (df['away_pass_completions'] / 
                                   df['away_pass_attempts'].replace(0, 1)) * 100
        
        return df
    
    def add_critical_down_rate(self, df):
        df = df.copy()
        
        # Home team critical down rate
        home_conversions = (df['home_third_down_converted'] + 
                          df['home_fourth_down_converted'])
        home_attempts = (df['home_third_down_attempts'] + 
                        df['home_fourth_down_attempts'])
        df['home_critical_down_rate'] = (home_conversions / home_attempts.replace(0, 1)) * 100
        
        # Away team critical down rate
        away_conversions = (df['away_third_down_converted'] + 
                          df['away_fourth_down_converted'])
        away_attempts = (df['away_third_down_attempts'] + 
                        df['away_fourth_down_attempts'])
        df['away_critical_down_rate'] = (away_conversions / away_attempts.replace(0, 1)) * 100
        
        return df
    
    def add_turnover_differential(self, df):
        df = df.copy()
        
        # Calculate total turnovers for each team
        home_turnovers = df['home_pass_ints'] + df['home_fumbles_lost']
        away_turnovers = df['away_pass_ints'] + df['away_fumbles_lost']
        
        # Calculate differential from each team's perspective
        df['home_turnover_differential'] = away_turnovers - home_turnovers
        df['away_turnover_differential'] = home_turnovers - away_turnovers
        
        return df
    
    def add_binary_roof(self, df):
        df = df.copy()
        df['is_outdoors'] = (df['roof'].str.lower() == 'outdoors').astype(int)
        return df
    
    def add_binary_surface(self, df):
        df = df.copy()
        df['is_grass'] = (df['surface'].str.lower() == 'grass').astype(int)
        return df

    def add_historical_win_pct(self, df, n_games=None):
        n_games = self.n_games
        df = df.copy()
        df = df.sort_values('date')
        
        def get_win_pct(team, current_date):
            # Get all previous games for this team
            mask = ((df['date'] < current_date) & 
                   ((df['home_team'] == team) | (df['away_team'] == team)))
            prev_games = df[mask]
            
            if prev_games.empty:
                return None
                
            # Take up to n_games most recent games
            prev_games = prev_games.tail(n_games)
            wins = 0
            for _, game in prev_games.iterrows():
                if game['home_team'] == team:
                    wins += 1 if game['home_score'] > game['away_score'] else 0
                else:
                    wins += 1 if game['away_score'] > game['home_score'] else 0
                    
            return wins / len(prev_games)
        
        df['home_historical_win_pct'] = df.apply(lambda row: 
            get_win_pct(row['home_team'], row['date']), axis=1)
        df['away_historical_win_pct'] = df.apply(lambda row: 
            get_win_pct(row['away_team'], row['date']), axis=1)
        
        # Remove rows where either team has no historical data
        df = df.dropna(subset=['home_historical_win_pct', 'away_historical_win_pct'])
        
        return df

    def add_historical_scoring(self, df, n_games=None):
        n_games = self.n_games
        df = df.copy()
        df = df.sort_values('date')
        
        def get_scoring_averages(team, current_date):
            # Get all previous games for this team
            mask = ((df['date'] < current_date) & 
                   ((df['home_team'] == team) | (df['away_team'] == team)))
            prev_games = df[mask]
            
            if prev_games.empty:
                return None, None
                
            # Take up to n_games most recent games
            prev_games = prev_games.tail(n_games)
            points_for = 0
            points_against = 0
            
            for _, game in prev_games.iterrows():
                if game['home_team'] == team:
                    points_for += game['home_score']
                    points_against += game['away_score']
                else:
                    points_for += game['away_score']
                    points_against += game['home_score']
            
            # Ensure we don't divide by zero
            n = len(prev_games)
            if n == 0:
                return None, None
            
            avg_points_for = points_for / n
            avg_points_against = points_against / n
            
            return avg_points_for, avg_points_against
        
        # Calculate for home team
        home_stats = df.apply(lambda row: 
            get_scoring_averages(row['home_team'], row['date']), axis=1)
        df['home_historical_points_for'] = [stats[0] for stats in home_stats]
        df['home_historical_points_against'] = [stats[1] for stats in home_stats]
        
        # Calculate for away team
        away_stats = df.apply(lambda row: 
            get_scoring_averages(row['away_team'], row['date']), axis=1)
        df['away_historical_points_for'] = [stats[0] for stats in away_stats]
        df['away_historical_points_against'] = [stats[1] for stats in away_stats]
        
        # Remove rows where either team has no historical data
        df = df.dropna(subset=[
            'home_historical_points_for', 'home_historical_points_against',
            'away_historical_points_for', 'away_historical_points_against'
        ])
        
        return df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all game features in sequence"""
        try:
            #df = self.add_margin_of_victory(df)
            df = self.add_yards_per_play(df)
            df = self.add_completion_pct(df)
            df = self.add_critical_down_rate(df)
            df = self.add_turnover_differential(df)
            #df = self.add_binary_roof(df)
            #df = self.add_binary_surface(df)
            #df = self.handle_weather_nulls(df)
            df = self.add_historical_win_pct(df)
            df = self.add_historical_scoring(df)
            return df
        except Exception as e:
            self.logger.error(f"Error computing features: {str(e)}")
            raise 