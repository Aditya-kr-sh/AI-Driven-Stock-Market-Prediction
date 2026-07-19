"""
Data loader module responsible for downloading stock datasets from Yahoo Finance.
Implements local cache hits, automatic retries with exponential backoff, and validates downloads.
"""

import time
from datetime import datetime, date
from typing import List, Dict, Optional
import pandas as pd
import os
# yfinance will be imported lazily within methods to avoid import-time SQLite errors.
# Fallback to pandas_datareader if yfinance cannot be imported due to missing sqlite3.
from tqdm import tqdm
from ai_engine.utils.logging import logger
from ai_engine.data.tickers import load_registry
from ai_engine.data.cleaner import DataCleaner
from ai_engine.data.storage import DataStorage
from ai_engine.data.dataset import StockDataset
from ai_engine.data.exceptions import DownloadError, ValidationError

# yfinance will be imported lazily within methods to avoid import-time SQLite errors.
# Fallback to pandas_datareader if yfinance cannot be imported due to missing sqlite3.


class DataLoader:
    """
    Data loading orchestrator that checks local cache availability, downloads
    data from yfinance with retries, and coordinates cleaning and storing.
    """

    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        retry_limit: int = 3,
        retry_delay: float = 2.0,
        index_name: str = "nifty50"
    ):
        self.storage = storage or DataStorage()
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self.index_name = index_name
        
        # Lazy import to break circular dependency at startup
        from ai_engine.utils.config import settings
        self.settings = settings

    def download_ticker(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_download: bool = False,
        interval: str = "1d"
    ) -> StockDataset:
        """
        Downloads stock data for a single stock.
        
        Args:
            ticker: Yahoo Finance ticker symbol (e.g. 'RELIANCE.NS').
            start: Start date string.
            end: End date string.
            force_download: Bypass cache check and redownload.
            interval: Data frequency interval (e.g. '1d', '1h', '30m', '15m', '5m', '1m').
            
        Returns:
            StockDataset object.
        """
        return self.get_ticker_data(
            ticker=ticker,
            start_date=start,
            end_date=end,
            force_download=force_download,
            interval=interval
        )

    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
        interval: str = "1d"
    ) -> StockDataset:
        """
        Retrieves historical stock data for a given ticker.
        Checks cache hit feasibility before querying yfinance.
        
        Args:
            ticker: Yahoo Finance ticker symbol (e.g. 'RELIANCE.NS').
            start_date: Start date string (YYYY-MM-DD), defaults to settings default.
            end_date: End date string (YYYY-MM-DD), defaults to settings default.
            force_download: If True, bypasses cache and downloads a fresh copy.
            interval: Data frequency interval (e.g. '1d', '1h', '30m', '15m', '5m', '1m').
            
        Returns:
            A populated StockDataset object.
            
        Raises:
            DownloadError: If download fails or ticker is invalid.
            ValidationError: If data constraints are violated.
        """
        ticker = ticker.strip().upper()
        start = start_date or self.settings.DEFAULT_START_DATE
        end = end_date or self.settings.DEFAULT_END_DATE

        # Validate index registry inclusion dynamically
        try:
            valid_tickers = load_registry(self.index_name)
            clean_t = ticker.split(".")[0]
            clean_registry = [vt.split(".")[0] for vt in valid_tickers]
            if ticker not in valid_tickers and clean_t not in clean_registry:
                logger.warning(f"Ticker '{ticker}' is not present in the index registry '{self.index_name}'.")
        except Exception as e:
            logger.warning(f"Could not check registry '{self.index_name}': {e}")

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
                cached_interval = metadata.get("interval", "1d")
                
                # Check if the cached dataset covers the requested range and matches the frequency interval
                if req_start_dt >= cached_start_dt and req_end_dt <= cached_end_dt and cached_interval == interval:
                    logger.info(f"Cache Hit for {ticker}. Loading local dataset.")
                    df = self.storage.load_raw(ticker)
                    
                    # Filter local dataset to match requested range
                    filtered_df = df.loc[req_start_dt:req_end_dt].copy()
                    
                    if not filtered_df.empty:
                        return StockDataset(ticker, filtered_df, metadata)
                    else:
                        logger.warning(f"Cached data exists but filtering to [{start} : {end}] returned empty. Forcing redownload.")
                else:
                    logger.info(f"Cache Miss for {ticker}. Fetching fresh data.")
            except Exception as e:
                logger.error(f"Error checking cache for {ticker}: {e}. Falling back to download.")

        # 2. Download from Yahoo Finance with retries
        logger.info(f"Downloading data for {ticker} from Yahoo Finance (Interval: {interval}). Range: {start} to {end}")
        df = self._download_with_retry(ticker, start, end, interval)

        # 3. Clean and Validate
        cleaned_df = DataCleaner.clean_dataframe(df)

        # 4. Save to Storage
        self.storage.save_raw(cleaned_df, ticker, start, end, interval)
        metadata = self.storage.load_raw_metadata(ticker)

        return StockDataset(ticker, cleaned_df, metadata)

    def download_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
        interval: str = "1d"
    ) -> Dict[str, StockDataset]:
        """
        Downloads or retrieves cache data for multiple tickers in parallel where available.
        
        Args:
            tickers: List of stock ticker symbols.
            start_date: Start date string.
            end_date: End date string.
            force_download: Force downloading even if cached locally.
            interval: Data frequency interval.
            
        Returns:
            Dict mapping ticker symbols to StockDataset objects.
        """
        results: Dict[str, StockDataset] = {}
        logger.info(f"Starting batch data load for {len(tickers)} tickers (Interval: {interval}).")
        
        # Segment tickers to run parallel cache loading separately from sequential downloads
        cached_tickers = []
        download_tickers = []
        
        if not force_download:
            for ticker in tickers:
                if self.storage.raw_exists(ticker.strip().upper()):
                    cached_tickers.append(ticker)
                else:
                    download_tickers.append(ticker)
        else:
            download_tickers = tickers

        # 1. Load cached files in parallel (using ThreadPoolExecutor)
        if cached_tickers:
            logger.info(f"Loading {len(cached_tickers)} cached tickers in parallel...")
            import concurrent.futures
            import os
            max_workers = min(16, os.cpu_count() or 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(
                        self.get_ticker_data,
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        force_download=False,
                        interval=interval
                    ): ticker for ticker in cached_tickers
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_ticker), total=len(cached_tickers), desc="Loading Cache", unit="ticker"):
                    ticker = future_to_ticker[future]
                    try:
                        results[ticker] = future.result()
                    except Exception as e:
                        logger.error(f"Failed loading cache for '{ticker}': {e}")
                        # Move to download queue as fallback
                        download_tickers.append(ticker)

        # 2. Download remaining tickers sequentially
        if download_tickers:
            logger.info(f"Downloading {len(download_tickers)} tickers sequentially...")
            for ticker in tqdm(download_tickers, desc="Downloading Tickers", unit="ticker"):
                try:
                    results[ticker] = self.get_ticker_data(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        force_download=True,
                        interval=interval
                    )
                except Exception as e:
                    logger.error(f"Failed to load dataset for ticker '{ticker}': {e}")
                    
        return results

    def _download_with_retry(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Downloads data using yfinance if available, otherwise falls back to pandas_datareader.
        Implements exponential backoff retry loops.
        """
        attempts = 0
        last_error_msg = ""
        
        # Lazy import yfinance to avoid import-time SQLite errors.
        try:
            import yfinance as yf  # type: ignore
        except Exception as import_err:
            logger.warning(f"yfinance import failed ({import_err}); falling back to pandas_datareader.")
            yf = None
        
        while attempts < self.retry_limit:
            try:
                if yf is not None:
                    # Use yfinance download
                    df = yf.download(
                        tickers=ticker,
                        start=start,
                        end=end,
                        interval=interval,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                    )
                else:
                    # Fallback using pandas_datareader
                    from pandas_datareader import data as pdr
                    df = pdr.get_data_yahoo(ticker, start=start, end=end, interval=interval)
                
                # Check if download returned a valid dataset
                if df is None or df.empty:
                    attempts += 1
                    last_error_msg = "download returned an empty DataFrame (invalid ticker or missing data)."
                    logger.warning(
                        f"Download attempt {attempts}/{self.retry_limit} failed for ticker '{ticker}' "
                        f"at timestamp {datetime.utcnow().isoformat()}Z. Reason: {last_error_msg}"
                    )
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
                logger.warning(
                    f"Download attempt {attempts}/{self.retry_limit} failed for ticker '{ticker}' "
                    f"at timestamp {datetime.utcnow().isoformat()}Z. Exception: {last_error_msg}"
                )
                if attempts < self.retry_limit:
                    time.sleep(self.retry_delay * attempts)

        raise DownloadError(
            f"Failed to download data for ticker '{ticker}' after {self.retry_limit} attempts. "
            f"Last Error: {last_error_msg}"
        )
