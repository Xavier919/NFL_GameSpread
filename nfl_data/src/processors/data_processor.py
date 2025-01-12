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
        """
        Process a single game row and combine with scraped game vector
        """
        try:
            # Scrape data from URL using scrape_tables instead of scrape_game_data
            tables_dict, error = self.scraper.scrape_tables(row['game_url'])
            if error:
                self.logger.error(f"Error scraping {row['game_url']}: {error}")
                return None
                
            # Get game vector
            game_vector = self.scraper.get_game_vector(tables_dict)
            if not game_vector:
                self.logger.warning(f"No game vector created for {row['game_url']}")
                return None
            
            # Add schedule information to game vector
            schedule_info = {
                'season': row['season'],
                'week': row['week'],
                'away_team': row['away_team'],
                'home_team': row['home_team'],
                'game_url': row['game_url'],
                'date': row['date']
            }
            
            # Combine dictionaries
            game_vector.update(schedule_info)
            
            return game_vector
            
        except Exception as e:
            self.logger.error(f"Error processing game {row['game_url']}: {str(e)}")
            return None

    def _save_game_data(self, game_vectors: List[Dict[str, Any]], year: int) -> None:
        """Save processed game data"""
        try:
            df = pd.DataFrame(game_vectors)
            output_path = Path(f"data/processed/game_data/nfl_games_{year}.csv")
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(game_vectors)} games for {year}")
        except Exception as e:
            self.logger.error(f"Error saving game data for {year}: {str(e)}")

    def process_season(self, year: int) -> None:
        """Process all games for a specific season"""
        try:
            # Filter schedule for the given year
            season_schedule = self.schedule_df[self.schedule_df['season'] == year].copy()
            if len(season_schedule) == 0:
                self.logger.warning(f"No games found for {year} season")
                return

            self.logger.info(f"Processing {len(season_schedule)} games for {year} season")
            
            # Process each game with progress bar
            game_vectors = []
            for _, row in tqdm(season_schedule.iterrows(), total=len(season_schedule)):
                game_vector = self.process_game(row)
                if game_vector:
                    game_vectors.append(game_vector)
                time.sleep(1)  # Be nice to the server
            
            # Save the processed data
            if game_vectors:
                self._save_game_data(game_vectors, year)
            else:
                self.logger.warning(f"No game vectors were created for {year} season")
            
        except Exception as e:
            self.logger.error(f"Error processing {year} season: {str(e)}")