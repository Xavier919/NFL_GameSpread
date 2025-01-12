import requests
from lxml import html
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, Tuple, Optional, Any
import requests
from lxml import html
import pandas as pd
import urllib3
import time
import random
from datetime import datetime


urllib3.disable_warnings()

class NFLScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        self.team_abbr = {
            'Texans': 'htx', 'Ravens': 'rav', 'Rams': 'ram', 'Titans': 'oti',
            'Cardinals': 'crd', 'Bears': 'chi', 'Bengals': 'cin', 'Bills': 'buf',
            'Broncos': 'den', 'Browns': 'cle', 'Buccaneers': 'tam', 'Chiefs': 'kan',
            'Colts': 'clt', 'Cowboys': 'dal', 'Dolphins': 'mia', 'Eagles': 'phi',
            'Falcons': 'atl', 'Giants': 'nyg', 'Jaguars': 'jax', 'Jets': 'nyj',
            'Lions': 'det', 'Packers': 'gnb', 'Panthers': 'car', 'Patriots': 'nwe',
            'Raiders': 'rai', 'Saints': 'nor', 'Seahawks': 'sea', 'Steelers': 'pit',
            'Vikings': 'min', 'Commanders': 'was', 'Chargers': 'sdg', '49ers': 'sfo',
            'Redskins': 'was', 'Football Team': 'was'
        }

    def make_request(self, url, max_retries=3):
        """Make request with retries, no proxy by default"""
        print(f"\nAttempting request to: {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10, verify=False)
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    print(f"Rate limited! Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Request failed with status: {response.status_code}")
                    time.sleep(2)
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(2)
        
        # If direct request fails, try with proxies as fallback
        return self.try_with_proxies(url)

    def try_with_proxies(self, url):
        """Fallback method using proxies"""
        proxies = self.get_working_proxies()
        for proxy in proxies[:5]:  # Try only 5 proxies max
            proxy_dict = {'http': f'http://{proxy}', 'https': f'http://{proxy}'}
            try:
                response = requests.get(url, headers=self.headers, proxies=proxy_dict, timeout=10, verify=False)
                if response.status_code == 200:
                    return response.content
            except:
                continue
            time.sleep(1)
        return None

    def get_working_proxies(self):
        """Get list of proxies"""
        urls = [
            "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
            "https://www.proxy-list.download/api/v1/get?type=http"
        ]
        working_proxies = []
        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                proxies = response.text.strip().split('\n')
                working_proxies.extend([p.strip() for p in proxies if p.strip()])
            except:
                continue
        return list(set(working_proxies))

    def generate_game_url(self, date_str, home_team):
        """Generate game URL from date and home team"""
        try:
            if 'Playoffs' in date_str:
                return None
                
            date_obj = pd.to_datetime(date_str)
            date_formatted = date_obj.strftime('%Y%m%d0')
            
            team_name = home_team.split()[-1]
            team_abbr = self.team_abbr.get(team_name, '').lower()
            if not team_abbr:
                print(f"Warning: No abbreviation found for team {home_team}")
                return None
            
            url = f"https://www.pro-football-reference.com/boxscores/{date_formatted}{team_abbr}.htm"
            return url
        except Exception as e:
            print(f"Error generating URL for {date_str}, {home_team}: {str(e)}")
            return None

    def scrape_schedule_year(self, year):
        """Scrape NFL schedule for a specific year."""
        url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
        content = self.make_request(url)
        
        if content:
            tree = html.fromstring(content)
            tables = tree.xpath('//table[@id="games"]')
            if tables:
                games = []
                for row in tables[0].xpath('.//tbody/tr[not(contains(@class, "thead"))]'):
                    try:
                        cols = row.xpath('.//th|.//td')
                        if len(cols) >= 10:
                            week = cols[0].text_content().strip()
                            if 'Week' in week or 'week' in week:
                                continue
                            
                            is_away_winner = '@' in cols[5].text_content()
                            winner = cols[4].find('a').text_content().strip() if cols[4].find('a') is not None else cols[4].text_content().strip()
                            loser = cols[6].find('a').text_content().strip() if cols[6].find('a') is not None else cols[6].text_content().strip()
                            
                            home_team = loser if is_away_winner else winner
                            away_team = winner if is_away_winner else loser
                            
                            game = {
                                'season': year,
                                'week': week,
                                'date': cols[2].text_content().strip(),
                                'time': cols[3].text_content().strip(),
                                'away_team': away_team,
                                'home_team': home_team
                            }
                            
                            game_url = self.generate_game_url(game['date'], game['home_team'])
                            if game_url:
                                game['game_url'] = game_url
                            
                            games.append(game)
                    except Exception as e:
                        print(f"Error processing row: {str(e)}")
                        continue
                
                return pd.DataFrame(games)
        return None

    def scrape_all_seasons(self, start_year=2020, end_year=2024):
        """Scrape schedules for multiple seasons"""
        all_seasons = []
        for year in range(start_year, end_year + 1):
            print(f"\nScraping {year} season...")
            schedule_df = self.scrape_schedule_year(year)
            if schedule_df is not None:
                all_seasons.append(schedule_df)
                schedule_df.to_csv(f"data/schedule_{year}.csv", index=False)
                print(f"Saved schedule for {year}")
                time.sleep(random.uniform(2, 5))
        
        if all_seasons:
            combined_df = pd.concat(all_seasons, ignore_index=True)
            combined_df = combined_df.dropna(subset=['week', 'game_url'])
            combined_df.to_csv("data/schedule_df.csv", index=False)
            return combined_df
        return None


# Create scraper and run
scraper = NFLScraper()
all_schedules = scraper.scrape_all_seasons(2010, 2024)

import requests
from lxml import html
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, Optional, Tuple, Any
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nfl_scraper.log')
    ]
)
logger = logging.getLogger(__name__)

class NFLGameScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_dynamic_content(self, url: str) -> Optional[str]:
        """
        Get dynamic content using Selenium when needed.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            Optional[str]: The page source if successful, None otherwise
        """
        driver = None
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            driver = webdriver.Chrome(options=options)
            
            logger.info(f"Fetching dynamic content from {url}")
            driver.get(url)
            
            # Wait for specific element to be present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "div_player_offense"))
            )
            return driver.page_source
            
        except Exception as e:
            logger.error(f"Error fetching dynamic content: {str(e)}")
            return None
            
        finally:
            if driver:
                driver.quit()

    def parse_table(self, table) -> Optional[pd.DataFrame]:
        """
        Parse HTML table into DataFrame with enhanced error handling.
        
        Args:
            table: The HTML table element to parse
            
        Returns:
            Optional[pd.DataFrame]: Parsed DataFrame or None if parsing fails
        """
        try:
            # Extract headers
            header_elements = table.xpath('.//thead//th|.//thead//td[@data-stat]')
            #if not header_elements:
            #    logger.warning("No header elements found in table")
            
            headers = []
            for h in header_elements:
                header = h.attrib.get('data-stat')
                if not header:
                    header = h.text_content().strip() if h.text_content() else f'Column_{len(headers)}'
                headers.append(header)
            
            # Extract rows
            rows = []
            for row in table.xpath('.//tbody/tr[not(contains(@class, "thead"))]'):
                if 'class' in row.attrib and 'hidden' in row.attrib['class']:
                    continue
                    
                cells = row.xpath('.//th|.//td')
                row_data = []
                for cell in cells:
                    value = cell.attrib.get('data-value', cell.text_content().strip())
                    row_data.append(value)
                
                if any(row_data):  # Only add non-empty rows
                    rows.append(row_data)
            
            if not rows:
                logger.warning("No data rows found in table")
                return None
            
            # Ensure headers match data
            if not headers:
                headers = [f'Column_{i}' for i in range(len(rows[0]))]
            elif len(headers) != len(rows[0]):
                logger.warning(f"Header count ({len(headers)}) doesn't match data column count ({len(rows[0])})")
                headers = headers[:len(rows[0])] if len(headers) > len(rows[0]) else headers + [f'Column_{i}' for i in range(len(headers), len(rows[0]))]
            
            return pd.DataFrame(rows, columns=headers)
            
        except Exception as e:
            logger.error(f"Error parsing table: {str(e)}")
            return None

    def safe_get_value(self, df: pd.DataFrame, row_condition: str, column: str) -> Optional[str]:
        """
        Safely extract value from DataFrame with error handling.
        
        Args:
            df (pd.DataFrame): The DataFrame to extract from
            row_condition (str): The condition to match in the row
            column (str): The column to extract from
            
        Returns:
            Optional[str]: The extracted value or None if not found
        """
        try:
            return df.loc[df['onecell'] == row_condition, column].iloc[0]
        except Exception as e:
            logger.warning(f"Could not extract {row_condition} from {column}: {str(e)}")
            return None

    def parse_game_info(self, game_info_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse game information including Vegas betting lines with standardized lowercase keys.
        """
        game_info = {}
        try:
            # Extract existing information (roof, surface, etc.)
            roof = self.safe_get_value(game_info_df, 'Roof', 'Column_1')
            game_info['roof'] = roof.lower() if roof else 'N/A'
            
            surface = self.safe_get_value(game_info_df, 'Surface', 'Column_1')
            game_info['surface'] = surface.lower() if surface else 'N/A'
            
            # Extract Vegas Line information
            vegas_line = self.safe_get_value(game_info_df, 'Vegas Line', 'Column_1')
            if vegas_line:
                try:
                    # Parse Vegas line (e.g., "Denver Broncos -2.5")
                    parts = vegas_line.split()
                    spread = float(parts[-1])
                    favored_team = ' '.join(parts[:-1])
                    game_info['spread'] = spread
                    game_info['favored_team'] = favored_team
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing Vegas line: {str(e)}")
                    game_info['spread'] = 'N/A'
                    game_info['favored_team'] = 'N/A'
            else:
                game_info['spread'] = 'N/A'
                game_info['favored_team'] = 'N/A'
    
            # Extract Over/Under
            over_under = self.safe_get_value(game_info_df, 'Over/Under', 'Column_1')
            if over_under:
                try:
                    # Parse over/under (e.g., "37.0 (over)")
                    total = float(over_under.split()[0])
                    game_info['vegas_total'] = total
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing over/under: {str(e)}")
                    game_info['vegas_total'] = 'N/A'
            else:
                game_info['vegas_total'] = 'N/A'
    
            # Extract attendance
            attendance = self.safe_get_value(game_info_df, 'Attendance', 'Column_1')
            if attendance:
                try:
                    game_info['attendance'] = int(attendance.replace(',', ''))
                except ValueError:
                    game_info['attendance'] = 'N/A'
            else:
                game_info['attendance'] = 'N/A'
                        
        except Exception as e:
            logger.error(f"Error parsing game info: {str(e)}")
            # Set default values if parsing fails
            default_fields = ['roof', 'surface', 'spread', 'favored_team', 'vegas_total', 'attendance']
            for field in default_fields:
                if field not in game_info:
                    game_info[field] = 'N/A'
                
        return game_info
    
    
    def parse_stat_value(self, value: str, stat_type: str) -> Optional[Any]:
        """
        Parse statistical values with error handling.
        
        Args:
            value (str): The value to parse
            stat_type (str): The type of statistic being parsed
            
        Returns:
            Optional[Any]: The parsed value or None if parsing fails
        """
        try:
            if '-' in value:
                parts = value.split('-')
                if stat_type == 'rush':
                    return {
                        'attempts': int(parts[0]),
                        'yards': int(parts[1]),
                        'tds': int(parts[2])
                    }
                elif stat_type == 'pass':
                    return {
                        'completions': int(parts[0]),
                        'attempts': int(parts[1]),
                        'yards': int(parts[2]),
                        'tds': int(parts[3]),
                        'ints': int(parts[4])
                    }
                elif stat_type in ['fumbles', 'penalties', 'downs']:
                    return {
                        'number': int(parts[0]),
                        'yards_or_lost': int(parts[1])
                    }
            return int(value)
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing stat value {value} of type {stat_type}: {str(e)}")
            return None

    def parse_team_stats(self, team_stats_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse team statistics with comprehensive error handling.
        """
        stats_dict = {}
        
        try:
            for _, row in team_stats_df.iterrows():
                stat_name = row['stat']
                vis_val = row['vis_stat']
                home_val = row['home_stat']
                
                if not all([stat_name, vis_val, home_val]):
                    continue
    
                # Time of Possession to possession percentage
                if stat_name == 'Time of Possession':
                    try:
                        # Convert MM:SS to seconds
                        def time_to_seconds(time_str):
                            minutes, seconds = map(int, time_str.split(':'))
                            return minutes * 60 + seconds
                        
                        away_seconds = time_to_seconds(vis_val)
                        home_seconds = time_to_seconds(home_val)
                        total_seconds = away_seconds + home_seconds
                        
                        stats_dict['away_possession_pct'] = round((away_seconds / total_seconds), 2)
                        stats_dict['home_possession_pct'] = round((home_seconds / total_seconds), 2)
                    except Exception as e:
                        logger.warning(f"Error parsing possession time: {str(e)}")
                
                # Third Down Conversions (format: X-Y)
                elif stat_name == 'Third Down Conv.':
                    try:
                        away_conv, away_att = map(int, vis_val.split('-'))
                        home_conv, home_att = map(int, home_val.split('-'))
                    
                        stats_dict.update({
                            'away_third_down_attempts': away_att,
                            'away_third_down_converted': away_conv,
                            'home_third_down_attempts': home_att,
                            'home_third_down_converted': home_conv
                        })
                    except Exception as e:
                        logger.warning(f"Error parsing third down conversions: {str(e)}")
                
                # Fourth Down Conversions (format: X-Y)
                elif stat_name == 'Fourth Down Conv.':
                    try:
                        away_conv, away_att = map(int, vis_val.split('-'))
                        home_conv, home_att = map(int, home_val.split('-'))
                        
                        stats_dict.update({
                            'away_fourth_down_attempts': away_att,
                            'away_fourth_down_converted': away_conv,
                            'home_fourth_down_attempts': home_att,
                            'home_fourth_down_converted': home_conv
                        })
                    except Exception as e:
                        logger.warning(f"Error parsing fourth down conversions: {str(e)}")
                
                # Penalties (format: X-Y)
                elif stat_name == 'Penalties-Yards':
                    try:
                        away_pen, away_yds = map(int, vis_val.split('-'))
                        home_pen, home_yds = map(int, home_val.split('-'))
                    
                        stats_dict.update({
                            'away_penalties': away_pen,
                            'away_penalties_yds': away_yds,
                            'home_penalties': home_pen,
                            'home_penalties_yds': home_yds
                        })
                    except Exception as e:
                        logger.warning(f"Error parsing penalties: {str(e)}")
                
                # Turnovers
                elif stat_name == 'Turnovers':
                    try:
                        stats_dict['away_turnovers'] = int(vis_val)
                        stats_dict['home_turnovers'] = int(home_val)
                    except Exception as e:
                        logger.warning(f"Error parsing turnovers: {str(e)}")
                
                # Sacked-Yards (format: X-Y)
                elif stat_name == 'Sacked-Yards':
                    try:
                        away_sacked, away_yds = map(int, vis_val.split('-'))
                        home_sacked, home_yds = map(int, home_val.split('-'))
                        
                        stats_dict.update({
                            'away_sacked': away_sacked,
                            'away_sacked_yds': away_yds,
                            'home_sacked': home_sacked,
                            'home_sacked_yds': home_yds
                        })
                    except Exception as e:
                        logger.warning(f"Error parsing sacks: {str(e)}")
            
                # Total Yards
                elif stat_name == 'Total Yards':
                    try:
                        stats_dict['away_total_yds'] = int(vis_val)
                        stats_dict['home_total_yds'] = int(home_val)
                    except Exception as e:
                        logger.warning(f"Error parsing total yards: {str(e)}")
                
                # Fumbles-Lost (format: X-Y)
                elif stat_name == 'Fumbles-Lost':
                    try:
                        away_fum, away_lost = map(int, vis_val.split('-'))
                        home_fum, home_lost = map(int, home_val.split('-'))
                        
                        stats_dict.update({
                            'away_fumbles': away_fum,
                            'away_fumbles_lost': away_lost,
                            'home_fumbles': home_fum,
                            'home_fumbles_lost': home_lost
                        })
                    except Exception as e:
                        logger.warning(f"Error parsing fumbles: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error parsing team stats: {str(e)}")
            
        return stats_dict
    


    def scrape_tables(self, url: str) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[str]]:
        """
        Scrape all NFL tables from the given URL with enhanced error handling.
        
        Args:
            url (str): The URL to scrape tables from
            
        Returns:
            Tuple[Optional[Dict[str, pd.DataFrame]], Optional[str]]: (Tables dict, error message if any)
        """
        try:
            logger.info(f"Scraping tables from {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            
            tree = html.fromstring(response.content)
            tables = tree.xpath('//table')
            
            if not tables:
                logger.warning("No tables found in initial HTML")
                return None, "No tables found in the page"
            
            if not any(table.attrib.get('id', '') == 'div_player_offense' for table in tables):
                logger.info("Attempting to get dynamic content")
                dynamic_content = self.get_dynamic_content(url)
                if dynamic_content:
                    tree = html.fromstring(dynamic_content)
                    tables = tree.xpath('//table')
            
            all_tables = {}
            for idx, table in enumerate(tables):
                table_id = table.attrib.get('id', f'Table_{idx}')
                df = self.parse_table(table)
                if df is not None and not df.empty:
                    all_tables[table_id] = df
            
            if not all_tables:
                return None, "No valid tables were parsed"
                
            logger.info(f"Successfully parsed {len(all_tables)} tables")
            return all_tables, None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return None, f"Failed to fetch URL: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None, str(e)


    def parse_drive_stats(self, home_drives_df: Optional[pd.DataFrame], away_drives_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Parse drive statistics for both teams.
        
        Args:
            home_drives_df: DataFrame containing home team drives
            away_drives_df: DataFrame containing away team drives
            
        Returns:
            Dict containing drive-related statistics
        """
        drive_stats = {}
    
        def process_drives(drives_df: pd.DataFrame, prefix: str) -> None:
            if drives_df is None or drives_df.empty:
                drive_stats.update({
                    f'{prefix}_drives': 0,
                    f'{prefix}_total_plays': 0,
                    f'{prefix}_punts': 0,
                    f'{prefix}_field_goals': 0,
                    f'{prefix}_touchdowns': 0
                })
                return
                
            try:
                # Count total drives
                drive_stats[f'{prefix}_drives'] = len(drives_df)
                
                # Sum total plays
                drive_stats[f'{prefix}_total_plays'] = drives_df['play_count_tip'].astype(int).sum()
                
                # Count end events
                end_events = drives_df['end_event'].value_counts()
                
                # Count specific events
                drive_stats[f'{prefix}_punts'] = end_events.get('Punt', 0)
                drive_stats[f'{prefix}_field_goals'] = end_events.get('Field Goal', 0)
                drive_stats[f'{prefix}_touchdowns'] = end_events.get('Touchdown', 0)
                
            except Exception as e:
                logger.error(f"Error processing {prefix} drives: {str(e)}")
                # Set default values if processing fails
                drive_stats.update({
                    f'{prefix}_drives': 0,
                    f'{prefix}_total_plays': 0,
                    f'{prefix}_punts': 0,
                    f'{prefix}_field_goals': 0,
                    f'{prefix}_touchdowns': 0
                })
        
        # Process home and away drives
        process_drives(home_drives_df, 'home')
        process_drives(away_drives_df, 'vis')
        
        return drive_stats


    def parse_starters(self, home_starters_df: Optional[pd.DataFrame], vis_starters_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Parse starting lineups for both teams, standardizing offensive line positions.
        """
        starters_dict = {}
        
        def process_starters(starters_df: pd.DataFrame, prefix: str) -> None:
            if starters_df is None or starters_df.empty:
                return
                
            try:
                # Track positions to handle duplicates
                position_counts = {}
                ol_count = 0  # Special counter for offensive line positions
                
                # Define offensive line positions to be standardized
                ol_positions = {'OL', 'T', 'OT', 'G', 'OG', 'C'}
                
                for _, row in starters_df.iterrows():
                    pos = row['pos'].strip().upper()  # Standardize to uppercase
                    player = row['player'].strip()
                    
                    # Handle offensive line positions
                    if pos in ol_positions:
                        ol_count += 1
                        final_pos = f'ol{ol_count}'
                    else:
                        # Handle other positions
                        if pos in position_counts:
                            position_counts[pos] += 1
                            final_pos = f"{pos.lower()}{position_counts[pos]}"
                        else:
                            position_counts[pos] = 1
                            final_pos = pos.lower()
                    
                    # Add to starters_dict with team prefix
                    starters_dict[f'{prefix}_{final_pos}'] = player
                        
            except Exception as e:
                logger.error(f"Error processing {prefix} starters: {str(e)}")
        
        # Process home and away starters
        process_starters(home_starters_df, 'home')
        process_starters(vis_starters_df, 'vis')
        
        return starters_dict


    def get_game_vector(self, tables_dict: Optional[Dict[str, pd.DataFrame]]) -> Optional[Dict[str, Any]]:
        """
        Create game vector with standardized lowercase keys and betting information.
        """
        try:
            if not tables_dict:
                raise ValueError("No tables dictionary provided")
                
            game_vector = {}
            
            # Get team names
            if 'Table_16' in tables_dict and tables_dict['Table_16'] is not None:
                teams_df = tables_dict['Table_16']
                if len(teams_df) >= 2:
                    try:
                        away_team_full = teams_df.iloc[0][1]
                        home_team_full = teams_df.iloc[1][1]
                        game_vector['away_team_id'] = away_team_full.split()[-1].lower()
                        game_vector['home_team_id'] = home_team_full.split()[-1].lower()
                    except Exception as e:
                        logger.warning(f"Error extracting team names: {str(e)}")
            
            # Get final scores
            if 'scoring' in tables_dict and tables_dict['scoring'] is not None:
                scoring_df = tables_dict['scoring']
                if not scoring_df.empty:
                    try:
                        final_row = scoring_df.iloc[-1]
                        game_vector['home_score'] = int(final_row['home_team_score'])
                        game_vector['away_score'] = int(final_row['vis_team_score'])
                    except Exception as e:
                        logger.warning(f"Error extracting scores: {str(e)}")
        
            # Get game info including Vegas lines
            if 'game_info' in tables_dict and tables_dict['game_info'] is not None:
                try:
                    game_info = self.parse_game_info(tables_dict['game_info'])
                    game_vector.update(game_info)
                except Exception as e:
                    logger.warning(f"Error extracting game info: {str(e)}")
            
            # Get team statistics (already using lowercase keys)
            if 'team_stats' in tables_dict and tables_dict['team_stats'] is not None:
                try:
                    team_stats = self.parse_team_stats(tables_dict['team_stats'])
                    # Convert team stats keys to lowercase
                    team_stats = {key.lower(): value for key, value in team_stats.items()}
                    game_vector.update(team_stats)
                except Exception as e:
                    logger.warning(f"Error extracting team stats: {str(e)}")
            
            # Get drive statistics
            try:
                home_drives_df = tables_dict.get('home_drives')
                away_drives_df = tables_dict.get('away_drives')
                drive_stats = self.parse_drive_stats(home_drives_df, away_drives_df)
                game_vector.update(drive_stats)
            except Exception as e:
                logger.warning(f"Error extracting drive stats: {str(e)}")
                
            # Get starters information
            try:
                home_starters_df = tables_dict.get('home_starters')
                vis_starters_df = tables_dict.get('vis_starters')
                starters_info = self.parse_starters(home_starters_df, vis_starters_df)
                game_vector.update(starters_info)
            except Exception as e:
                logger.warning(f"Error extracting starters info: {str(e)}")
            
            if not game_vector:
                logger.warning("No data could be extracted from the tables")
                return None
                    
            logger.info("Successfully created game vector")
            # Standardize all keys to lowercase before returning
            return {key.lower(): value for key, value in game_vector.items()}
                
        except Exception as e:
            logger.error(f"Error creating game vector: {str(e)}")
            return None



import pandas as pd
import json
from typing import Dict, Tuple, List, Any
import time
from tqdm import tqdm

class NFLDataProcessor:
    def __init__(self, schedule_df: pd.DataFrame, scraper: NFLGameScraper):
        self.schedule_df = schedule_df
        self.scraper = scraper
        
    def process_game(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single game row and combine with scraped game vector
        """
        try:
            # Scrape data from URL
            tables_dict, error = self.scraper.scrape_tables(row['game_url'])
            if error:
                print(f"Error scraping {row['game_url']}: {error}")
                return None
                
            # Get game vector
            game_vector = self.scraper.get_game_vector(tables_dict)
            
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
            print(f"Error processing game {row['game_url']}: {str(e)}")
            return None
    
    def process_all_games(self) -> List[Dict[str, Any]]:
        """
        Process all games in the schedule dataframe
        """
        all_game_vectors = []
        
        # Use tqdm for progress bar
        for _, row in tqdm(self.schedule_df.iterrows(), total=len(self.schedule_df)):
            game_vector = self.process_game(row)
            if game_vector:
                all_game_vectors.append(game_vector)
                
            # Add small delay to be nice to the server
            time.sleep(1)
            
        return all_game_vectors
    
    def save_to_csv(self, game_vectors: List[Dict[str, Any]], output_file: str):
        """
        Save all game vectors to a CSV file
        """
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(game_vectors)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved {len(game_vectors)} games to {output_file}")


schedule_df = pd.read_csv('data/schedule_df.csv')

for year in range(2010,2025):
    schedule_df_yr = schedule_df[schedule_df['season'] == year]
    scraper = NFLGameScraper()
    processor = NFLDataProcessor(schedule_df_yr, scraper)
    game_vectors = processor.process_all_games()
    processor.save_to_csv(game_vectors, f'data/nfl_boxscore_{year}.csv')