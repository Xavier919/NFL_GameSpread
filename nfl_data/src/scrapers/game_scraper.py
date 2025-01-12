from nfl_data.src.scrapers.base_scraper import BaseScraper
import pandas as pd
from lxml import html
import requests
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from nfl_data.src.utils.logger import setup_logger

class GameScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        
    def get_dynamic_content(self, url: str) -> Optional[str]:
        """Get dynamic content using Selenium"""
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
        """Parse HTML table into DataFrame"""
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
                return None
            
            if not headers:
                headers = [f'Column_{i}' for i in range(len(rows[0]))]
            elif len(headers) != len(rows[0]):
                headers = headers[:len(rows[0])] if len(headers) > len(rows[0]) else headers + [f'Column_{i}' for i in range(len(headers), len(rows[0]))]
            
            return pd.DataFrame(rows, columns=headers)
            
        except Exception as e:
            self.logger.error(f"Error parsing table: {str(e)}")
            return None

    def scrape_tables(self, url: str) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[str]]:
        """Scrape all tables from the game page"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            tree = html.fromstring(response.content)
            tables = tree.xpath('//table')
            
            if not tables:
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
            
            return all_tables, None
            
        except Exception as e:
            return None, str(e)

    def parse_game_info(self, game_info_df: pd.DataFrame) -> Dict[str, Any]:
        """Parse game information including Vegas betting lines"""
        game_info = {}
        try:
            roof = self.safe_get_value(game_info_df, 'Roof', 'Column_1')
            game_info['roof'] = roof.lower() if roof else 'N/A'
            
            surface = self.safe_get_value(game_info_df, 'Surface', 'Column_1')
            game_info['surface'] = surface.lower() if surface else 'N/A'
            
            vegas_line = self.safe_get_value(game_info_df, 'Vegas Line', 'Column_1')
            if vegas_line:
                try:
                    parts = vegas_line.split()
                    spread = float(parts[-1])
                    favored_team = ' '.join(parts[:-1])
                    game_info['spread'] = spread
                    game_info['favored_team'] = favored_team
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing Vegas line: {str(e)}")
                    game_info['spread'] = 'N/A'
                    game_info['favored_team'] = 'N/A'
            else:
                game_info['spread'] = 'N/A'
                game_info['favored_team'] = 'N/A'

            over_under = self.safe_get_value(game_info_df, 'Over/Under', 'Column_1')
            if over_under:
                try:
                    total = float(over_under.split()[0])
                    game_info['vegas_total'] = total
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing over/under: {str(e)}")
                    game_info['vegas_total'] = 'N/A'
            else:
                game_info['vegas_total'] = 'N/A'

            attendance = self.safe_get_value(game_info_df, 'Attendance', 'Column_1')
            if attendance:
                try:
                    game_info['attendance'] = int(attendance.replace(',', ''))
                except ValueError:
                    game_info['attendance'] = 'N/A'
            else:
                game_info['attendance'] = 'N/A'
                    
        except Exception as e:
            self.logger.error(f"Error parsing game info: {str(e)}")
            default_fields = ['roof', 'surface', 'spread', 'favored_team', 'vegas_total', 'attendance']
            for field in default_fields:
                if field not in game_info:
                    game_info[field] = 'N/A'
            
        return game_info

    def parse_team_stats(self, team_stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Parse team statistics"""
        stats_dict = {}
        try:
            for _, row in team_stats_df.iterrows():
                stat_name = row['stat']
                vis_val = row['vis_stat']
                home_val = row['home_stat']
                
                if not all([stat_name, vis_val, home_val]):
                    continue

                # Time of Possession to percentage
                if stat_name == 'Time of Possession':
                    try:
                        def time_to_seconds(time_str):
                            minutes, seconds = map(int, time_str.split(':'))
                            return minutes * 60 + seconds
                        
                        away_seconds = time_to_seconds(vis_val)
                        home_seconds = time_to_seconds(home_val)
                        total_seconds = away_seconds + home_seconds
                        
                        stats_dict['away_possession_pct'] = round((away_seconds / total_seconds), 3)
                        stats_dict['home_possession_pct'] = round((home_seconds / total_seconds), 3)
                    except Exception as e:
                        self.logger.warning(f"Error parsing possession time: {str(e)}")
                        continue

                # Parse other statistics
                stat_key = stat_name.lower().replace(' ', '_')
                stats_dict[f'away_{stat_key}'] = vis_val
                stats_dict[f'home_{stat_key}'] = home_val

        except Exception as e:
            self.logger.error(f"Error parsing team stats: {str(e)}")
        
        return stats_dict

    def parse_drive_stats(self, home_drives_df: Optional[pd.DataFrame], away_drives_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Parse drive statistics for both teams"""
        drive_stats = {}
        
        def process_drives(drives_df: Optional[pd.DataFrame], prefix: str) -> None:
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
                drive_stats[f'{prefix}_drives'] = len(drives_df)
                drive_stats[f'{prefix}_total_plays'] = drives_df['play_count_tip'].astype(int).sum()
                
                end_events = drives_df['end_event'].value_counts()
                drive_stats[f'{prefix}_punts'] = end_events.get('Punt', 0)
                drive_stats[f'{prefix}_field_goals'] = end_events.get('Field Goal', 0)
                drive_stats[f'{prefix}_touchdowns'] = end_events.get('Touchdown', 0)
                
            except Exception as e:
                self.logger.error(f"Error processing {prefix} drives: {str(e)}")
                drive_stats.update({
                    f'{prefix}_drives': 0,
                    f'{prefix}_total_plays': 0,
                    f'{prefix}_punts': 0,
                    f'{prefix}_field_goals': 0,
                    f'{prefix}_touchdowns': 0
                })
        
        process_drives(home_drives_df, 'home')
        process_drives(away_drives_df, 'away')
        
        return drive_stats

    def parse_starters(self, home_starters_df: Optional[pd.DataFrame], vis_starters_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Parse starting lineups for both teams"""
        starters_dict = {}
        
        def process_starters(starters_df: pd.DataFrame, prefix: str) -> None:
            if starters_df is None or starters_df.empty:
                return
                
            try:
                position_counts = {}
                ol_count = 0
                ol_positions = {'OL', 'T', 'OT', 'G', 'OG', 'C'}
                
                for _, row in starters_df.iterrows():
                    pos = row['pos'].strip().upper()
                    player = row['player'].strip()
                    
                    if pos in ol_positions:
                        ol_count += 1
                        final_pos = f'ol{ol_count}'
                    else:
                        if pos in position_counts:
                            position_counts[pos] += 1
                            final_pos = f"{pos.lower()}{position_counts[pos]}"
                        else:
                            position_counts[pos] = 1
                            final_pos = pos.lower()
                    
                    starters_dict[f'{prefix}_{final_pos}'] = player
                        
            except Exception as e:
                self.logger.error(f"Error processing {prefix} starters: {str(e)}")
        
        process_starters(home_starters_df, 'home')
        process_starters(vis_starters_df, 'vis')
        
        return starters_dict

    def safe_get_value(self, df: pd.DataFrame, row_condition: str, column: str) -> Optional[str]:
        """Safely extract value from DataFrame"""
        try:
            return df.loc[df['onecell'] == row_condition, column].iloc[0]
        except Exception as e:
            self.logger.warning(f"Could not extract {row_condition} from {column}: {str(e)}")
            return None

    def get_game_vector(self, tables_dict: Optional[Dict[str, pd.DataFrame]]) -> Optional[Dict[str, Any]]:
        """Create game vector with standardized lowercase keys and betting information"""
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
                        self.logger.warning(f"Error extracting team names: {str(e)}")
            
            # Get final scores
            if 'scoring' in tables_dict and tables_dict['scoring'] is not None:
                scoring_df = tables_dict['scoring']
                if not scoring_df.empty:
                    try:
                        final_row = scoring_df.iloc[-1]
                        game_vector['home_score'] = int(final_row['home_team_score'])
                        game_vector['away_score'] = int(final_row['vis_team_score'])
                    except Exception as e:
                        self.logger.warning(f"Error extracting scores: {str(e)}")
            
            # Get game info including Vegas lines
            if 'game_info' in tables_dict and tables_dict['game_info'] is not None:
                try:
                    game_info = self.parse_game_info(tables_dict['game_info'])
                    game_vector.update(game_info)
                except Exception as e:
                    self.logger.warning(f"Error extracting game info: {str(e)}")
            
            # Get team statistics
            if 'team_stats' in tables_dict and tables_dict['team_stats'] is not None:
                try:
                    team_stats = self.parse_team_stats(tables_dict['team_stats'])
                    team_stats = {key.lower(): value for key, value in team_stats.items()}
                    game_vector.update(team_stats)
                except Exception as e:
                    self.logger.warning(f"Error extracting team stats: {str(e)}")
            
            # Get drive statistics
            try:
                home_drives_df = tables_dict.get('home_drives')
                away_drives_df = tables_dict.get('away_drives')
                drive_stats = self.parse_drive_stats(home_drives_df, away_drives_df)
                game_vector.update(drive_stats)
            except Exception as e:
                self.logger.warning(f"Error extracting drive stats: {str(e)}")
            
            # Get starters information
            try:
                home_starters_df = tables_dict.get('home_starters')
                vis_starters_df = tables_dict.get('vis_starters')
                starters_info = self.parse_starters(home_starters_df, vis_starters_df)
                game_vector.update(starters_info)
            except Exception as e:
                self.logger.warning(f"Error extracting starters info: {str(e)}")
            
            if not game_vector:
                self.logger.warning("No data could be extracted from the tables")
                return None
                
            self.logger.info("Successfully created game vector")
            return {key.lower(): value for key, value in game_vector.items()}
            
        except Exception as e:
            self.logger.error(f"Error creating game vector: {str(e)}")
            return None