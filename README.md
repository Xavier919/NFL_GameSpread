# NFL_GameSpread
# NFL Game Spread Prediction
## Overview
This repository contains code for modeling NFL game spreads using historical data from the past 10 NFL seasons. The project aims to analyze and predict point spreads in NFL games using various statistical and machine learning approaches.
## What is a Spread?
In NFL betting and analysis, the spread (or point spread) is the predicted scoring differential between two teams in a game. For example, if Team A is favored by 7 points over Team B, the spread would be written as:
- Team A -7.0
- Team B +7.0
This means:
- If Team A wins by more than 7 points, they "cover" the spread
- If Team B loses by less than 7 points (or wins), they "cover" the spread
- If Team A wins by exactly 7 points, it's a "push"
The spread serves several purposes:
1. It creates a theoretical level playing field between two teams of different skill levels
2. It provides a quantitative measure of team strength differences
3. It represents the market's aggregate prediction of game outcomes
## Project Structure
### 1. Scraper.py
- Scrapes game data from Pro Football Reference
- Collects comprehensive box scores and player statistics
- Handles both historical and current season data
- Features robust error handling and rate limiting
- Saves raw data in structured format
### 2. GameFeatureVector.py
- Processes raw game data into structured feature vectors
- Extracts relevant game statistics and metrics
- Creates normalized and standardized features
- Handles missing data and outliers
- Generates consistent feature representation for each game
### 3. PredictSpread.py
- Implements various prediction models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - Neural Networks
- Performs model evaluation and comparison
- Includes cross-validation and performance metrics
- Provides visualization of results
## Installation and Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/NFL_GameSpread.git
# Install required packages
pip install -r requirements.txt
# Run the scraper to collect data
python Scraper.py
# Generate feature vectors
python GameFeatureVector.py
# Train and evaluate models
python PredictSpread.py
```
## Data Sources
- Game statistics from Pro Football Reference
- Historical spread data from verified sources
- Team performance metrics and player statistics
## Model Features
The project uses various features including:
- Team performance metrics
- Historical head-to-head results
- Player availability and statistics
- Weather conditions
- Home/away performance
- Recent form and momentum
- Strength of schedule
## Disclaimer
This project is for educational and research purposes only. It is not intended for and should not be used for gambling or betting purposes. The NFL data used in this project is owned by the National Football League and its partners. This project does not claim ownership of any NFL data or statistics.
## License
[Insert your chosen license here]
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## Contact
[Your contact information or preferred method of contact]
