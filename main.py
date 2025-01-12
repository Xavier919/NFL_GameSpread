from nfl_data.src.scrapers.game_scraper import GameScraper
from nfl_data.src.processors.data_processor import NFLDataProcessor
from pathlib import Path
import pandas as pd
import logging
from nfl_data.src.utils.logger import setup_logger

def main():
    logger = setup_logger("main")
    
    # Create all required directories upfront
    data_dirs = [
        Path("data/raw/schedules"),
        Path("data/processed/game_data"),
        Path("logs")
    ]
    
    for directory in data_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        
    try:
        # Initialize scrapers and processors with proper GameScraper
        game_scraper = GameScraper()
        
        # Load schedule data
        schedule_df = pd.read_csv('data/raw/schedules/schedule_all.csv')
        
        data_processor = NFLDataProcessor(schedule_df, game_scraper)
        
        # Process each season
        logger.info("Processing game data for each season")
        for year in range(2023, 2024):
            logger.info(f"Processing {year} season")
            data_processor.process_season(year)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
