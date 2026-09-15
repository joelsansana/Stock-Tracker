"""
ARK ETF Holdings Fetcher
========================

Downloads and parses ARK Invest ETF holdings data.

ARK publishes a CSV per ETF on ark-funds.com. These CSVs are cached
locally on disk and only re-downloaded on :meth:`get_holding` calls
with ``force_refresh=True``.

Note:
    ARK occasionally changes the exact URL slug for a fund. If a
    download returns 404, check ``ARK_ETF_URLS`` for the current slug.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

import pandas as pd
import requests

logger = logging.getLogger(__name__)


ARK_ETF_URLS: Dict[str, str] = {
    "ARKK": "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "ARKQ": "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
    "ARKW": "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
    "ARKG": "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv",
    "ARKF": "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
}

DEFAULT_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class ARKDataFetcher:
    """Fetches and caches ARK ETF holdings data.

    Args:
        data_dir: Directory used to cache downloaded CSVs. Created if missing.
        timeout: HTTP timeout in seconds for each download.
        max_workers: Max threads for :meth:`get_all_holdings`.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        timeout: int = DEFAULT_TIMEOUT,
        max_workers: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _safe_url(self, url: str) -> str:
        """Percent-encode characters in the URL path that must be encoded.

        ARK's CSV filenames contain literal ``&`` characters which must be
        sent as ``%26`` per RFC 3986.
        """
        return quote(url, safe=":/?=%")

    def _download_csv(self, ticker: str, url: str) -> Optional[pd.DataFrame]:
        """Download a single ETF's CSV and parse it.

        On parse failure we discard any partial file rather than caching it
        so the next call retries the network.
        """
        try:
            logger.info("Downloading %s", ticker)
            response = self.session.get(self._safe_url(url), timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("Timeout downloading %s", ticker)
            return None
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error downloading %s: %s", ticker, exc)
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Network error downloading %s: %s", ticker, exc)
            return None

        filepath = self.data_dir / f"{ticker}_holdings.csv"
        try:
            df = pd.read_csv(pd.io.common.BytesIO(response.content))
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            logger.error("Error parsing CSV for %s: %s", ticker, exc)
            return None

        # Only persist the cache after a successful parse.
        try:
            filepath.write_bytes(response.content)
        except OSError as exc:
            logger.warning("Could not cache %s to %s: %s", ticker, filepath, exc)

        logger.info("Successfully downloaded %s: %d holdings", ticker, len(df))
        return df

    def get_holding(
        self, ticker: str, force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """Return the holdings DataFrame for a single ETF.

        Args:
            ticker: ETF ticker symbol, case-insensitive (e.g. ``"ARKK"``).
            force_refresh: Re-download even if a cached CSV exists.

        Returns:
            Holdings DataFrame, or ``None`` if the ticker is unknown or
            the download failed.
        """
        ticker = ticker.upper()
        if ticker not in ARK_ETF_URLS:
            logger.error("Unknown ticker: %s", ticker)
            return None

        filepath = self.data_dir / f"{ticker}_holdings.csv"
        if not force_refresh and filepath.exists():
            try:
                df = pd.read_csv(filepath)
                logger.info("Loaded %s from cache (%d holdings)", ticker, len(df))
                return df
            except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
                logger.warning("Cache for %s unreadable, re-downloading: %s", ticker, exc)

        return self._download_csv(ticker, ARK_ETF_URLS[ticker])

    def get_all_holdings(
        self, force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Fetch holdings for every ETF in :data:`ARK_ETF_URLS`.

        Downloads are issued concurrently. ETFs that fail to download are
        omitted from the result rather than raising.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.get_holding, ticker, force_refresh): ticker
                for ticker in ARK_ETF_URLS
            }
            results: Dict[str, pd.DataFrame] = {}
            for future, ticker in futures.items():
                df = future.result()
                if df is not None:
                    results[ticker] = df
        return results

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> ARKDataFetcher:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


if __name__ == "__main__":
    with ARKDataFetcher() as fetcher:
        arkk = fetcher.get_holding("ARKK")
        if arkk is not None:
            print(arkk.head())