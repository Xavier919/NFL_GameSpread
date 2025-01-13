from nfl_data.src.scrapers.base_scraper import BaseScraper
import pandas as pd
from lxml import html
import requests
import time
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from nfl_data.src.utils.logger import setup_logger
import logging
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from urllib.parse import urlparse
import sqlite3
from contextlib import contextmanager
from tqdm import tqdm
from nfl_data.src.utils.logger import setup_logger


logger = setup_logger("GameScraper")

# Common modern user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0'
]

@dataclass
class ProxyStats:
    """Track proxy performance metrics"""
    address: str
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime = datetime.now()
    avg_response_time: float = 0.0
    last_error: str = ""
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

class DatabaseManager:
    def __init__(self, db_path: str = "scraper_data.db"):
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    domain TEXT PRIMARY KEY,
                    last_request TIMESTAMP,
                    request_count INTEGER,
                    reset_time TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proxy_performance (
                    proxy_address TEXT PRIMARY KEY,
                    success_count INTEGER,
                    failure_count INTEGER,
                    avg_response_time REAL,
                    last_used TIMESTAMP,
                    last_error TEXT
                )
            """)
            conn.commit()
    
    def save_proxy(self, proxy_stats: ProxyStats):
        """Save or update proxy stats in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO proxy_performance
                (proxy_address, success_count, failure_count, avg_response_time, last_used, last_error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                proxy_stats.address,
                proxy_stats.success_count,
                proxy_stats.failure_count,
                proxy_stats.avg_response_time,
                proxy_stats.last_used.isoformat(),
                proxy_stats.last_error
            ))
            conn.commit()
    
    def load_proxies(self) -> Dict[str, ProxyStats]:
        """Load proxy stats from database"""
        proxies = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM proxy_performance WHERE (julianday('now') - julianday(last_used)) < 1")
            for row in cursor.fetchall():
                stats = ProxyStats(
                    address=row[0],
                    success_count=row[1],
                    failure_count=row[2],
                    avg_response_time=row[3],
                    last_used=datetime.fromisoformat(row[4]),
                    last_error=row[5]
                )
                proxies[stats.address] = stats
        return proxies

class GameScraper:
    def __init__(self):
        self.logger = setup_logger("GameScraper")
        self.db_manager = DatabaseManager()
        self.active_proxies = {}
        self.proxies = []
        self.current_proxy = None
        self.session = requests.Session()
        # Define user agents list as instance variable
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0'
        ]
        self._load_proxies()
        self.recent_requests = {}
        self.domain_delays = {}
        self._rotate_session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_dynamic_content(self, url: str) -> Optional[str]:
        """Get dynamic content using Selenium when needed."""
        driver = None
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            driver = webdriver.Chrome(options=options)
            
            self.logger.info(f"Fetching dynamic content from {url}")
            driver.get(url)
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "div_player_offense"))
            )
            return driver.page_source
            
        except Exception as e:
            self.logger.error(f"Error fetching dynamic content: {str(e)}")
            return None
            
        finally:
            if driver:
                driver.quit()

    def parse_table(self, table) -> Optional[pd.DataFrame]:
        """Parse HTML table into DataFrame with enhanced error handling."""
        try:
            header_elements = table.xpath('.//thead//th|.//thead//td[@data-stat]')
            
            headers = []
            for h in header_elements:
                header = h.attrib.get('data-stat')
                if not header:
                    header = h.text_content().strip() if h.text_content() else f'Column_{len(headers)}'
                headers.append(header)
            
            rows = []
            for row in table.xpath('.//tbody/tr[not(contains(@class, "thead"))]'):
                if 'class' in row.attrib and 'hidden' in row.attrib['class']:
                    continue
                    
                cells = row.xpath('.//th|.//td')
                row_data = []
                for cell in cells:
                    value = cell.attrib.get('data-value', cell.text_content().strip())
                    row_data.append(value)
                
                if any(row_data):
                    rows.append(row_data)
            
            if not rows:
                self.logger.warning("No data rows found in table")
                return None
            
            if not headers:
                headers = [f'Column_{i}' for i in range(len(rows[0]))]
            elif len(headers) != len(rows[0]):
                self.logger.warning(f"Header count ({len(headers)}) doesn't match data column count ({len(rows[0])})")
                headers = headers[:len(rows[0])] if len(headers) > len(rows[0]) else headers + [f'Column_{i}' for i in range(len(headers), len(rows[0]))]
            
            return pd.DataFrame(rows, columns=headers)
            
        except Exception as e:
            self.logger.error(f"Error parsing table: {str(e)}")
            return None

    # Include all other methods from the original NFLGameScraper
    def safe_get_value(self, df: pd.DataFrame, row_condition: str, column: str) -> Optional[str]:
        """Safely extract value from DataFrame with error handling."""
        try:
            return df.loc[df['onecell'] == row_condition, column].iloc[0]
        except Exception as e:
            self.logger.warning(f"Could not extract {row_condition} from {column}: {str(e)}")
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

    def _load_proxies(self):
        """Load proxy list from file or initialize empty"""
        try:
            proxy_file = Path("data/proxies.txt")
            if proxy_file.exists():
                with open(proxy_file, 'r') as f:
                    self.proxies = [line.strip() for line in f if line.strip()]
                self.logger.info(f"Loaded {len(self.proxies)} proxies")
            else:
                self.logger.warning("No proxy file found at data/proxies.txt")
                self.proxies = []
        except Exception as e:
            self.logger.error(f"Error loading proxies: {str(e)}")
            self.proxies = []
            
    def _rotate_session(self):
        """Rotate session with new user agent and proxy"""
        try:
            # Reset session
            self.session = requests.Session()
            
            # Set new random user agent
            self.session.headers.update({
                'User-Agent': random.choice(self.user_agents)
            })
            
            # Rotate proxy if available
            if self.proxies:
                self.current_proxy = random.choice(self.proxies)
                self.session.proxies = {
                    'http': self.current_proxy,
                    'https': self.current_proxy
                }
                self.logger.debug(f"Rotated to proxy: {self.current_proxy}")
            
        except Exception as e:
            self.logger.error(f"Error rotating session: {str(e)}")
            # Ensure we have a working session even if rotation fails
            self.session = requests.Session()

