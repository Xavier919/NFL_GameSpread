import pandas as pd
from collections import Counter
import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

schedule_df = pd.read_csv('data/raw/schedules/schedule_all.csv')
nfl_game_data = pd.read_csv('data/processed/game_data/nfl_boxscore_2018.csv')

n_games=10

class GameFeatures:
    def __init__(self, n_games=10):
        self.n_games = n_games
        pass
        
    def _convert_time_to_seconds(self, time_str):
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except:
            return None
    
    def add_margin_of_victory(self, df):
        df = df.copy()
        df['home_margin'] = df['Home_score'] - df['Away_score']
        df['away_margin'] = df['Away_score'] - df['Home_score']
        return df
    
    def add_yards_per_play(self, df):
        df = df.copy()
        
        # Home team YPP
        total_home_plays = df['Home_Rush_attempts'] + df['Home_Pass_attempts']
        total_home_yards = df['Home_Rush_yds'] + df['Home_Pass_yds']
        df['home_ypp'] = total_home_yards / total_home_plays.replace(0, 1)
        
        # Away team YPP
        total_away_plays = df['Away_Rush_attempts'] + df['Away_Pass_attempts']
        total_away_yards = df['Away_Rush_yds'] + df['Away_Pass_yds']
        df['away_ypp'] = total_away_yards / total_away_plays.replace(0, 1)
        
        return df
    
    def add_completion_pct(self, df):
        df = df.copy()
        
        df['home_completion_pct'] = (df['Home_Pass_completions'] / 
                                   df['Home_Pass_attempts'].replace(0, 1)) * 100
        df['away_completion_pct'] = (df['Away_Pass_completions'] / 
                                   df['Away_Pass_attempts'].replace(0, 1)) * 100
        
        return df
    
    def add_critical_down_rate(self, df):
        ### TODO : Converted and attempts are inverted ... modify scraper
        df = df.copy()
        
        # Home team critical down rate
        home_conversions = (df['Home_Third_down_converted'] + 
                          df['Home_Fourth_down_converted'])
        home_attempts = (df['Home_Third_down_attempts'] + 
                        df['Home_Fourth_down_attempts'])
        df['home_critical_down_rate'] = (home_attempts / home_conversions) * 100
        
        # Away team critical down rate
        away_conversions = (df['Away_Third_down_converted'] + 
                          df['Away_Fourth_down_converted'])
        away_attempts = (df['Away_Third_down_attempts'] + 
                        df['Away_Fourth_down_attempts'])
        df['away_critical_down_rate'] = (away_attempts / away_conversions) * 100
        
        return df
    
    def add_turnover_differential(self, df):
        df = df.copy()
        
        # Calculate total turnovers for each team
        home_turnovers = df['Home_Ints'] + df['Home_Fumbles_converted']
        away_turnovers = df['Away_Ints'] + df['Away_Fumbles_converted']
        
        # Calculate differential from each team's perspective
        df['home_turnover_differential'] = away_turnovers - home_turnovers
        df['away_turnover_differential'] = home_turnovers - away_turnovers
        
        return df
    
    def add_binary_roof(self, df):
        df = df.copy()
        df['is_outdoors'] = (df['Roof'] == 'outdoors').astype(int)
        return df
    
    def add_binary_surface(self, df):
        df = df.copy()
        df['is_grass'] = (df['Surface'] == 'grass').astype(int)
        return df

    def add_possession_pct(self, df):
        df = df.copy()
        
        def calculate_possession_pct(row, team_prefix):
            possession_time = self._convert_time_to_seconds(row[f'{team_prefix}_Time_of_possession'])
            if possession_time is not None:
                return (possession_time / (60 * 60)) 
            return None
        
        df['home_possession_pct'] = df.apply(lambda row: calculate_possession_pct(row, 'Home'), axis=1) * 100
        df['away_possession_pct'] = df.apply(lambda row: calculate_possession_pct(row, 'Away'), axis=1) * 100
        
        return df

    def handle_weather_nulls(self, df):
        df = df.copy()
        
        # Calculate means for outdoor games only
        outdoor_mask = df['Roof'] == 'outdoors'
        temp_mean = df.loc[outdoor_mask, 'Degrees'].mean()
        humidity_mean = df.loc[outdoor_mask, 'Humidity'].mean()
        wind_mean = df.loc[outdoor_mask, 'Wind'].mean()
        
        # Fill NaN values
        df['Degrees'] = df['Degrees'].fillna(temp_mean)
        df['Humidity'] = df['Humidity'].fillna(humidity_mean)
        df['Wind'] = df['Wind'].fillna(wind_mean)
        
        return df

    def add_historical_win_pct(self, df, n_games=None):
        n_games = self.n_games

        df = df.copy()
        df = df.sort_values('date')
        
        def get_win_pct(team, current_date):
            # Get all previous games for this team
            mask = ((df['date'] < current_date) & 
                   ((df['home_team_name'] == team) | (df['away_team_name'] == team)))
            prev_games = df[mask].tail(n_games)
            
            if prev_games.empty:
                return None
                
            wins = 0
            for _, game in prev_games.iterrows():
                if game['home_team_name'] == team:
                    wins += 1 if game['Home_score'] > game['Away_score'] else 0
                else:
                    wins += 1 if game['Away_score'] > game['Home_score'] else 0
                    
            return wins / len(prev_games)
        
        df['home_historical_win_pct'] = df.apply(lambda row: 
            get_win_pct(row['home_team_name'], row['date']), axis=1)
        df['away_historical_win_pct'] = df.apply(lambda row: 
            get_win_pct(row['away_team_name'], row['date']), axis=1)
        
        return df

    def add_historical_scoring(self, df, n_games=None):
        n_games = self.n_games
        
        df = df.copy()
        df = df.sort_values('date')
        
        def get_scoring_averages(team, current_date):
            # Get all previous games for this team
            mask = ((df['date'] < current_date) & 
                   ((df['home_team_name'] == team) | (df['away_team_name'] == team)))
            prev_games = df[mask].tail(n_games)
            
            if prev_games.empty:
                return None, None
                
            points_for = 0
            points_against = 0
            
            for _, game in prev_games.iterrows():
                if game['home_team_name'] == team:
                    points_for += game['Home_score']
                    points_against += game['Away_score']
                else:
                    points_for += game['Away_score']
                    points_against += game['Home_score']
                    
            avg_points_for = points_for / len(prev_games) * 100
            avg_points_against = points_against / len(prev_games) * 100
            
            return avg_points_for, avg_points_against
        
        # Calculate for home team
        home_stats = df.apply(lambda row: 
            get_scoring_averages(row['home_team_name'], row['date']), axis=1)
        df['home_historical_points_for'] = [stats[0] for stats in home_stats]
        df['home_historical_points_against'] = [stats[1] for stats in home_stats]
        
        # Calculate for away team
        away_stats = df.apply(lambda row: 
            get_scoring_averages(row['away_team_name'], row['date']), axis=1)
        df['away_historical_points_for'] = [stats[0] for stats in away_stats]
        df['away_historical_points_against'] = [stats[1] for stats in away_stats]
        
        return df

    def compute_all_features(self, df):
        df = self.add_margin_of_victory(df)
        df = self.add_yards_per_play(df)
        df = self.add_completion_pct(df)
        df = self.add_critical_down_rate(df)
        df = self.add_possession_pct(df)
        df = self.add_turnover_differential(df)
        df = self.add_binary_roof(df)
        df = self.add_binary_surface(df)
        df = self.handle_weather_nulls(df)
        df = self.add_historical_win_pct(df)
        df = self.add_historical_scoring(df)
        return df
    

# Initialize the calculator
calculator = GameFeatures(n_games=n_games)

nfl_game_data = calculator.compute_all_features(nfl_game_data)


def calculate_team_averages(df, team, current_game, n_games=10, is_home=True):
    """
    Calculate team metrics using N previous games using pre-computed features
    """
    prefix = 'home' if is_home else 'away'
    
    # Team filter for all games of this team (both home and away)
    team_filter = (df['home_team_name'] == team) | (df['away_team_name'] == team)
    
    # Time filter: All games before current game's date
    time_filter = df['date'] < current_game['date']
    
    # Get team's past games
    team_games = df[team_filter & time_filter].sort_values('date', ascending=True)
    
    # Skip if this is week 1 of the first season
    first_season = df['year'].min()
    if current_game['year'] == first_season and current_game['week'] == '1':
        return None
    
    # If no historical games available, return None
    if team_games.empty:
        return None
        
    # Take last N games (or all if fewer available)
    team_data = team_games.tail(n_games)
    
    averages = {}
    
    # List of features to average
    features_to_average = [
        'completion_pct',
        'ypp', 
        'critical_down_rate',
        'possession_pct',
        'turnover_differential'
    ]
    
    # For each game, get the team's stats whether they were home or away
    stats = []
    for _, game in team_data.iterrows():
        game_prefix = 'home' if game['home_team_name'] == team else 'away'
            
        game_stats = {
            feature: game[f'{game_prefix}_{feature}'] 
            for feature in features_to_average
        }
        stats.append(game_stats)
    
    # Calculate averages for the metrics
    for metric in features_to_average:
        values = [s[metric] for s in stats]
        averages[f'{prefix}_{metric}'] = sum(values) / len(values)
            
    return averages

def create_feature_vectors(df, n_games=10):
    """Create feature vectors using N previous games"""
    df = df.sort_values('date')
    
    feature_vectors = []
    targets = []
    game_info = []
    
    for idx, game in df.iterrows():
        current_game = {
            'date': game['date'],
            'week': game['week'],
            'year': game['year']
        }
        
        # Get team averages
        home_stats = calculate_team_averages(df, game['home_team_name'], current_game, is_home=True)
        away_stats = calculate_team_averages(df, game['away_team_name'], current_game, is_home=False)
        
        # Skip if either team has no history (only happens in week 1 of first season)
        if home_stats is None or away_stats is None:
            continue
            
        # Combine stats with current game environmental features and historical metrics
        features = {
            **home_stats,
            **away_stats,
            'home_historical_win_pct': game['home_historical_win_pct'],
            'away_historical_win_pct': game['away_historical_win_pct'],
            'home_historical_points_for': game['home_historical_points_for'],
            'away_historical_points_for': game['away_historical_points_for'],
            'home_historical_points_against': game['home_historical_points_against'],
            'away_historical_points_against': game['away_historical_points_against']
        }
        
        spread = game['Away_score'] - game['Home_score']
        
        feature_vectors.append(features)
        targets.append(spread)
        game_info.append(current_game)
    
    return feature_vectors, targets, game_info

def prepare_nfl_data(df, n_games=10):
    """Main function to prepare NFL data with zero vector filtering"""
    # Make sure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # First compute the game-level features using our GameFeatures class
    calculator = GameFeatures()
    df = calculator.compute_all_features(df)
    
    # Create feature vectors and targets
    feature_vectors, targets, game_info = create_feature_vectors(df, n_games=n_games)
    
    # Convert feature vectors to consistent format
    all_keys = set().union(*[d.keys() for d in feature_vectors])
    all_keys = sorted(all_keys)
    
    # Convert to numpy arrays while preserving all values
    X = np.array([[fv.get(key, 0) for key in all_keys] for fv in feature_vectors])
    y = np.array(targets)
    
    # Filter out rows where all features are zero
    non_zero_mask = ~np.all(X == 0, axis=1)
    X_filtered = X[non_zero_mask]
    y_filtered = y[non_zero_mask]
    
    # Filter game_info accordingly
    game_info_filtered = [info for i, info in enumerate(game_info) if non_zero_mask[i]]
    
    # Print filtering statistics
    total_samples = len(X)
    filtered_samples = len(X_filtered)
    removed_samples = total_samples - filtered_samples

    return X_filtered, y_filtered, all_keys, game_info_filtered


feature_vectors, targets, feature_names, game_info = prepare_nfl_data(nfl_game_data, n_games=n_games)



# Convert to numpy arrays
X = np.array(feature_vectors)
y = np.array(targets)

def display_predictions(y_true, y_pred, n_samples=5):
    """Display comparison between true and predicted values"""
    indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    
    print("\nSample True vs Predicted Values:")
    print("-" * 45)
    print(f"{'True Spread':>12} | {'Predicted Spread':>15} | {'Diff':>8}")
    print("-" * 45)
    
    for idx in indices:
        true_val = y_true[idx]
        pred_val = y_pred[idx]
        diff = abs(true_val - pred_val)
        print(f"{true_val:12.2f} | {pred_val:15.2f} | {diff:8.2f}")
    print("-" * 45)

# Initialize 5-fold cross validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Initialize lists to store metrics
mse_scores = []
mae_scores = []
r2_scores = []

# Initialize Linear Regression model
model = LinearRegression()

# Perform 5-fold cross validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Store metrics
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    
    print(f"\nFold {fold} Results:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")
    
    # Display sample predictions for this fold
    display_predictions(y_val, y_pred)

# Calculate and print average metrics
print("\nAverage Results Across All Folds:")
print(f"Average MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
print(f"Average RMSE: {np.sqrt(np.mean(mse_scores)):.2f}")
print(f"Average MAE: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
print(f"Average R2: {np.mean(r2_scores):.3f} (+/- {np.std(r2_scores):.3f})")

# Train final model on all data
final_model = LinearRegression()
final_model.fit(X, y)

# Get feature coefficients
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': final_model.coef_
})
coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(coefficients.head(20).to_string(index=False))