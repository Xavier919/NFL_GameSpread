from nfl_data.src.scrapers.base_scraper import BaseScraper
import pandas as pd
from lxml import html
from typing import Optional, Dict, Any
from pathlib import Path
import time
import random
import requests

class NFLScraper(BaseScraper):
    def __init__(self):
        super().__init__()
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

    def get_working_proxies(self):
        """Get list of working proxies"""
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

    def try_with_proxies(self, url):
        """Try request with proxies"""
        proxies = self.get_working_proxies()
        for proxy in proxies[:5]:
            proxy_dict = {'http': f'http://{proxy}', 'https': f'http://{proxy}'}
            try:
                response = requests.get(url, headers=self.headers, proxies=proxy_dict, timeout=10, verify=False)
                if response.status_code == 200:
                    return response.content
            except:
                continue
            time.sleep(1)
        return None

    def make_request(self, url, max_retries=3):
        """Make request with retries"""
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
        
        return self.try_with_proxies(url)

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
        """Scrape NFL schedule for a specific year"""
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
        self.ensure_data_directories()
        all_seasons = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nScraping {year} season...")
            schedule_df = self.scrape_schedule_year(year)
            if schedule_df is not None:
                all_seasons.append(schedule_df)
                output_path = Path("data/raw/schedules")
                output_path.mkdir(parents=True, exist_ok=True)
                schedule_df.to_csv(output_path / f"schedule_{year}.csv", index=False)
                print(f"Saved schedule for {year}")
                time.sleep(random.uniform(2, 5))
        
        if all_seasons:
            combined_df = pd.concat(all_seasons, ignore_index=True)
            combined_df = combined_df.dropna(subset=['week', 'game_url'])
            combined_df.to_csv("data/raw/schedules/schedule_all.csv", index=False)
            return combined_df
        return None