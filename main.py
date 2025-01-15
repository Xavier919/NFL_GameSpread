import argparse
import pandas as pd
from nfl_data.src.scrapers.game_scraper import GameScraper
from nfl_data.src.processors.data_processor import NFLDataProcessor
from nfl_data.src.utils.logger import setup_logger

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='NFL Game Data Scraper')
    parser.add_argument('--season', type=int, required=True,
                      help='Season year to scrape (e.g., 2023)')
    parser.add_argument('--start-week', type=int,
                      help='Starting week number (optional)')
    parser.add_argument('--end-week', type=int,
                      help='Ending week number (optional)')
    
    args = parser.parse_args()
    
    logger = setup_logger("main")
    logger.info(f"Processing {args.season} season" + 
               (f" weeks {args.start_week}-{args.end_week}" if args.start_week else ""))

    try:
        # Load the schedule
        schedule_df = pd.read_csv('data/raw/schedules/schedule_all.csv')
        
        # Convert week column to integer
        schedule_df['week'] = pd.to_numeric(schedule_df['week'], errors='coerce')
        
        # Filter for specified season
        season_schedule = schedule_df[schedule_df['season'] == args.season]
        
        if len(season_schedule) == 0:
            logger.error(f"No games found for season {args.season}")
            return
            
        # Create scraper and processor
        scraper = GameScraper()
        processor = NFLDataProcessor(
            schedule_df=season_schedule,
            scraper=scraper
        )
        
        # Filter schedule by week range if specified
        if args.start_week:
            season_schedule = season_schedule[season_schedule['week'] >= args.start_week]
        if args.end_week:
            season_schedule = season_schedule[season_schedule['week'] <= args.end_week]
        
        # Process the games
        games_data = []
        for _, row in season_schedule.iterrows():
            game_vector = processor.process_game(row)
            if game_vector:
                games_data.append(game_vector)
        
        # Save results
        if games_data:
            output_file = f'data/processed/game_data/nfl_boxscore_{args.season}'
            if args.start_week:
                output_file += f'_week_{args.start_week}'
                if args.end_week:
                    output_file += f'-{args.end_week}'
            output_file += '.csv'
            
            df = pd.DataFrame(games_data)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(games_data)} games to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
