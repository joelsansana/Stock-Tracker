"""
ARK ETF Holdings Fetcher
Downloads and parses ARK Investment ETF holdings data.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ARK ETF ticker to CSV URL mapping
ARK_ETF_URLS = {
    'ARKK': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv',
    'ARKQ': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv',
    'ARKW': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv',
    'ARKG': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv',
    'ARKF': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv',
}


class ARKDataFetcher:
    """Fetches and caches ARK ETF holdings data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize ARK data fetcher.
        
        Args:
            data_dir: Directory to store downloaded CSV files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _download_csv(self, ticker: str, url: str) -> Optional[pd.DataFrame]:
        """
        Download CSV from URL with error handling.
        
        Args:
            ticker: ETF ticker symbol
            url: URL to download from
            
        Returns:
            DataFrame or None if download fails
        """
        try:
            logger.info(f"Downloading {ticker} from {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to file
            filepath = self.data_dir / f"{ticker}_holdings.csv"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Parse CSV
            df = pd.read_csv(filepath)
            logger.info(f"Successfully downloaded {ticker}: {len(df)} holdings")
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading {ticker}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading {ticker}: {e}")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {ticker}: {e}")
        
        return None
    
    def get_holding(self, ticker: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get holdings for a specific ETF.
        
        Args:
            ticker: ETF ticker symbol (e.g., 'ARKK')
            force_refresh: Force re-download even if cached file exists
            
        Returns:
            DataFrame of holdings or None if failed
        """
        ticker = ticker.upper()
        
        if ticker not in ARK_ETF_URLS:
            logger.error(f"Unknown ticker: {ticker}")
            return None
        
        # Check cache
        filepath = self.data_dir / f"{ticker}_holdings.csv"
        if not force_refresh and filepath.exists():
            try:
                df = pd.read_csv(filepath)
                logger.info(f"Loaded {ticker} from cache")
                return df
            except Exception as e:
                logger.warning(f"Error loading cached {ticker}: {e}")
        
        # Download fresh data
        url = ARK_ETF_URLS[ticker]
        return self._download_csv(ticker, url)
    
    def get_all_holdings(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download all ARK ETF holdings.
        
        Args:
            force_refresh: Force re-download
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in ARK_ETF_URLS:
            df = self.get_holding(ticker, force_refresh=force_refresh)
            if df is not None:
                results[ticker] = df
        
        return results
    
    def close(self):
        """Close the session."""
        self.session.close()


if __name__ == "__main__":
    # Example usage
    fetcher = ARKDataFetcher()
    
    # Download ARKK holdings
    arkk = fetcher.get_holding('ARKK')
    if arkk is not None:
        print(arkk.head())
    
    fetcher.close()
