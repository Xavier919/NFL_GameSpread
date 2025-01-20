# NFL Game Spread Modeling
## Overview
This repository contains code for modeling NFL game spreads using historical data from the past 15 NFL seasons. The project aims to analyze and predict point spreads in NFL games using various statistical and machine learning approaches.
## What is a Spread?
The point spread is the predicted scoring differential between two teams in a game. For example, if Team A is favored by 7 points over Team B, the spread would be written as:
- Team A -7.0
- Team B +7.0
- If Team A wins by more than 7 points, they "cover" the spread
- If Team B loses by less than 7 points (or wins), they "cover" the spread
- If Team A wins by exactly 7 points, it's a "push"
## Installation and Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/NFL_GameSpread.git
# Install required packages
pip install -r requirements.txt
# Run the scraper to collect data
python main --start_season 2010 --end_season 2024 
# Train the spread model
python scripts/train_spread_model.py --n-games 15 # Use 15 historical games
# Use the trained model to predict future games by specifying the home and away team, respectively
python scripts/predict_spread.py "San Francisco 49ers" "Dallas Cowboys"
```
## Variable Descriptions
- **home_interaction_term**: Interaction term for home team performance (Win Pct * Strength of Schedule).
- **away_interaction_term**: Interaction term for away team performance (Win Pct * Strength of Schedule).
- **away_historical_win_pct**: Historical winning percentage of the away team.
- **home_historical_win_pct**: Historical winning percentage of the home team.
- **home_strength_of_schedule**: Strength of schedule metric for the home team.
- **away_strength_of_schedule**: Strength of schedule metric for the away team.
- **home_turnover_differential**: Turnover differential for the home team.
- **away_turnover_differential**: Turnover differential for the away team.
- **home_ypp**: Yards per play for the home team.
- **away_defense_ypp**: Yards per play allowed by the away team's defense.
- **away_ypp**: Yards per play for the away team.
- **home_defense_ypp**: Yards per play allowed by the home team's defense.
- **away_critical_down_rate**: Critical down conversion rate for the away team.
- **home_critical_down_rate**: Critical down conversion rate for the home team.
- **home_historical_points_for**: Historical points scored by the home team.
- **away_historical_points_for**: Historical points scored by the away team.
- **away_completion_pct**: Completion percentage for the away team.
- **home_completion_pct**: Completion percentage for the home team.
- **home_historical_points_against**: Historical points allowed by the home team.
- **away_historical_points_against**: Historical points allowed by the away team.
- **away_sack_differential**: Sack differential for the away team.
- **home_sack_differential**: Sack differential for the home team.
- **Intercept**: The intercept term of the model, representing the home field advantage.

## Disclaimer
This project is for educational and research purposes only and does not claim ownership of any NFL data.

