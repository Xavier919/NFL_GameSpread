from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from nfl_data.src.scrapers.game_scraper import GameScraper
from nfl_data.src.utils.logger import setup_logger

class NFLDataProcessor:
    def __init__(self, schedule_df: pd.DataFrame, scraper: GameScraper):
        self.schedule_df = schedule_df
        self.scraper = scraper
        self.logger = setup_logger("NFLDataProcessor")
        
    def process_game(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single game row and combine with scraped game vector"""
        try:
            tables_dict, error = self.scraper.scrape_tables(row['game_url'])
            if error:
                self.logger.error(f"Error scraping {row['game_url']}: {error}")
                return None
                
            game_vector = self.scraper.get_game_vector(tables_dict)
            if not game_vector:
                return None
                
            schedule_info = {
                'season': row['season'],
                'week': row['week'],
                'away_team': row['away_team'],
                'home_team': row['home_team'],
                'game_url': row['game_url'],
                'date': row['date']
            }
            
            game_vector.update(schedule_info)
            return game_vector
            
        except Exception as e:
            self.logger.error(f"Error processing game {row['game_url']}: {str(e)}")
            return None
            
    def process_all_games(self) -> List[Dict[str, Any]]:
        """Process all games in the schedule dataframe"""
        all_game_vectors = []
        for _, row in tqdm(self.schedule_df.iterrows(), total=len(self.schedule_df)):
            game_vector = self.process_game(row)
            if game_vector:
                all_game_vectors.append(game_vector)
        return all_game_vectors
        
    def save_to_csv(self, game_vectors: List[Dict[str, Any]], output_file: str):
        """Save all game vectors to a CSV file"""
        df = pd.DataFrame(game_vectors)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {len(game_vectors)} games to {output_file}")