import requests
from typing import Dict, Optional, Tuple, Any
import time
import random
from pathlib import Path
from nfl_data.src.utils.logger import setup_logger

class BaseScraper:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

    def make_request(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Make HTTP request with retries"""
        self.logger.info(f"Requesting URL: {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10, verify=False)
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed: {response.status_code}")
                    time.sleep(2)
            except Exception as e:
                self.logger.error(f"Request error: {str(e)}")
                time.sleep(2)
        
        return None

    def ensure_data_directories(self):
        """Ensure all required data directories exist"""
        directories = [
            Path("data/raw/schedules"),
            Path("data/processed/game_data")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)