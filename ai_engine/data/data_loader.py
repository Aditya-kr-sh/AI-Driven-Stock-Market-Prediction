"""
Data loader module responsible for downloading stock datasets from Yahoo Finance.
Implements local cache hits, automatic retries with exponential backoff, and validates downloads.
"""

import time
from datetime import datetime, date
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from ai_engine.utils.config import settings
from ai_engine.utils.logging import logger
from ai_engine.data.tickers import is_valid_nifty_50_ticker
from ai_engine.data.cleaner import DataCleaner
from ai_engine.data.storage import DataStorage
from ai_engine.data.dataset import StockDataset
from ai_engine.data.exceptions import DownloadError, ValidationError

class DataLoader:
    """
    Data loading orchestrator that checks local cache availability, downloads
    data from yfinance with retries, and coordinates cleaning and storing.
    """

    def __init__(self, storage: Optional[DataStorage] = None, retry_limit: int = 3, retry_delay: float = 2.0):
        self.storage = storage or DataStorage()
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay

    def download_ticker(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_download: bool = False
    ) -> StockDataset:
        """
        Downloads stock data for a single stock.
        
        Args:
            ticker: Yahoo Finance ticker symbol (e.g. 'RELIANCE.NS').
            start: Start date string.
            end: End date string.
            force_download: Bypass cache check and redownload.
            
        Returns:
            StockDataset object.
        """
        return self.get_ticker_data(
            ticker=ticker,
            start_date=start,
            end_date=end,
            force_download=force_download
        )

    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False
    ) -> StockDataset:
        """
        Retrieves historical stock data for a given ticker.
        Checks cache hit feasibility before querying yfinance.
        
        Args:
            ticker: Yahoo Finance ticker symbol (e.g. 'RELIANCE.NS').
            start_date: Start date string (YYYY-MM-DD), defaults to settings default.
            end_date: End date string (YYYY-MM-DD), defaults to settings default.
            force_download: If True, bypasses cache and downloads a fresh copy.
            
        Returns:
            A populated StockDataset object.
            
        Raises:
            DownloadError: If download fails or ticker is invalid.
            ValidationError: If data constraints are violated.
        """
        ticker = ticker.strip().upper()
        start = start_date or settings.DEFAULT_START_DATE
        end = end_date or settings.DEFAULT_END_DATE

        # Log warning if ticker is not in central NIFTY 50 list
        if not is_valid_nifty_50_ticker(ticker):
            logger.warning(f"Ticker '{ticker}' is not present in the centralized NIFTY 50 tickers list.")

        # Validate date formats
        try:
            req_start_dt = pd.to_datetime(start)
            req_end_dt = pd.to_datetime(end)
            if req_start_dt >= req_end_dt:
                raise ValidationError(f"Start date ({start}) must be chronologically before end date ({end}).")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid date format passed to loader: {e}")

        # 1. Caching Check
        if not force_download and self.storage.raw_exists(ticker):
            try:
                metadata = self.storage.load_raw_metadata(ticker)
                cached_start_dt = pd.to_datetime(metadata["start_date"])
                cached_end_dt = pd.to_datetime(metadata["end_date"])
                
                # Check if the cached dataset covers the requested range
                if req_start_dt >= cached_start_dt and req_end_dt <= cached_end_dt:
                    logger.info(f"Cache Hit for {ticker}. Loading local dataset.")
                    df = self.storage.load_raw(ticker)
                    
                    # Filter local dataset to match requested range
                    filtered_df = df.loc[req_start_dt:req_end_dt].copy()
                    
                    if not filtered_df.empty:
                        return StockDataset(ticker, filtered_df, metadata)
                    else:
                        logger.warning(f"Cached data exists but filtering to [{start} : {end}] returned empty. Forcing redownload.")
                else:
                    logger.info(f"Cache Miss for {ticker}: requested range [{start} to {end}] exceeds cached range [{metadata['start_date']} to {metadata['end_date']}]. Fetching fresh data.")
            except Exception as e:
                logger.error(f"Error checking cache for {ticker}: {e}. Falling back to download.")

        # 2. Download from Yahoo Finance with retries
        logger.info(f"Downloading data for {ticker} from Yahoo Finance. Range: {start} to {end}")
        df = self._download_with_retry(ticker, start, end)

        # 3. Clean and Validate
        cleaned_df = DataCleaner.clean_dataframe(df)

        # 4. Save to Storage
        # Note: Save the full downloaded range to cache. Update dates in metadata.
        self.storage.save_raw(cleaned_df, ticker, start, end)
        metadata = self.storage.load_raw_metadata(ticker)

        return StockDataset(ticker, cleaned_df, metadata)

    def download_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False
    ) -> Dict[str, StockDataset]:
        """
        Downloads or retrieves cache data for multiple tickers sequentially.
        Displays terminal progress bars using tqdm.
        
        Args:
            tickers: List of stock ticker symbols.
            start_date: Start date string.
            end_date: End date string.
            force_download: Force downloading even if cached locally.
            
        Returns:
            Dict mapping ticker symbols to StockDataset objects.
        """
        results: Dict[str, StockDataset] = {}
        logger.info(f"Starting batch data load for {len(tickers)} tickers.")
        
        for ticker in tqdm(tickers, desc="Loading Stock Datasets", unit="ticker"):
            try:
                results[ticker] = self.get_ticker_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    force_download=force_download
                )
            except Exception as e:
                logger.error(f"Failed to load dataset for ticker '{ticker}': {e}")
                
        return results

    def _download_with_retry(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Downloads data from yfinance implementing exponential backoff retry loops."""
        attempts = 0
        last_error_msg = ""
        
        while attempts < self.retry_limit:
            try:
                # yfinance returns an empty DataFrame if no data is found or ticker is invalid
                df = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )
                
                # Check if download returned a valid dataset
                if df is None or df.empty:
                    attempts += 1
                    last_error_msg = "yfinance returned an empty DataFrame (invalid ticker or missing data)."
                    logger.warning(f"Download attempt {attempts}/{self.retry_limit} failed for {ticker}: {last_error_msg}")
                    if attempts < self.retry_limit:
                        time.sleep(self.retry_delay * attempts)
                    continue

                # Flatten MultiIndex columns if returned by yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Ensure 'Adj Close' is present if Close is available
                if "Adj Close" not in df.columns and "Close" in df.columns:
                    df["Adj Close"] = df["Close"]

                return df
            except Exception as e:
                attempts += 1
                last_error_msg = str(e)
                logger.warning(f"Download attempt {attempts}/{self.retry_limit} encountered error for {ticker}: {e}")
                if attempts < self.retry_limit:
                    time.sleep(self.retry_delay * attempts)

        raise DownloadError(
            f"Failed to download data for ticker '{ticker}' after {self.retry_limit} attempts. "
            f"Last Error: {last_error_msg}"
        )
