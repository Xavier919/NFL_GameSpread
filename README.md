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
python scripts/train_spread_model.py
# Use the trained model to predict future games by specifying the home and away team, respectively
python scripts/predict_spread.py "San Francisco 49ers" "Dallas Cowboys"
```
## Disclaimer
This project is for educational and research purposes only. This project does not claim ownership of any NFL data or statistics.

