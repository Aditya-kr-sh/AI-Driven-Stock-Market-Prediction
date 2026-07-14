"""
Data storage manager handling saving and loading of raw/processed datasets.
Supports configurable file formats (CSV or Parquet) and generates structured JSON metadata companions.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from ai_engine.utils.config import settings
from ai_engine.utils.logging import logger
from ai_engine.data.exceptions import StorageError

class DataStorage:
    """
    Manages physical file I/O operations for stock datasets.
    Supports CSV and Parquet formats, and creates companion metadata JSON files.
    """

    def __init__(self, raw_dir: Path = None, processed_dir: Path = None, file_format: str = None):
        self.raw_dir = raw_dir or settings.DATA_RAW_DIR
        self.processed_dir = processed_dir or settings.DATA_PROCESSED_DIR
        self.file_format = (file_format or settings.STORAGE_FORMAT).lower()

        if self.file_format not in ["csv", "parquet"]:
            logger.warning(f"Unsupported storage format '{self.file_format}'. Defaulting to 'csv'.")
            self.file_format = "csv"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> Path:
        """
        Saves a cleaned DataFrame to the raw storage directory along with its companion metadata JSON.
        
        Args:
            df: Cleaned stock price DataFrame.
            ticker: The stock ticker (e.g. 'RELIANCE.NS').
            start_date: Request start date.
            end_date: Request end date.
            
        Returns:
            The Path where the dataset was saved.
            
        Raises:
            StorageError: If file write operations fail.
        """
        filename = f"{ticker}.{self.file_format}"
        filepath = self.raw_dir / filename
        
        try:
            # 1. Save data file
            if self.file_format == "parquet":
                df.to_parquet(filepath, index=True)
            else:
                df.to_csv(filepath, index=True)
            logger.info(f"Saved raw dataset to: {filepath}")
            
            # 2. Write companion metadata JSON
            metadata = {
                "ticker": ticker,
                "download_timestamp": datetime.utcnow().isoformat() + "Z",
                "start_date": start_date,
                "end_date": end_date,
                "row_count": len(df),
                "source": "Yahoo Finance",
                "file_format": self.file_format
            }
            self._write_metadata(filepath, metadata)
            
            return filepath
        except Exception as e:
            raise StorageError(f"Failed to save raw dataset for {ticker}: {e}")

    def load_raw(self, ticker: str) -> pd.DataFrame:
        """
        Loads a raw dataset from local storage.
        
        Args:
            ticker: The stock ticker (e.g. 'RELIANCE.NS').
            
        Returns:
            A pandas DataFrame of the stock history.
            
        Raises:
            StorageError: If the file does not exist, is corrupted, or fails to parse.
        """
        filename = f"{ticker}.{self.file_format}"
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise StorageError(f"Raw dataset for ticker {ticker} not found at {filepath}")

        try:
            if self.file_format == "parquet":
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
            
            # Formitting check to ensure index is a DatetimeIndex
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            raise StorageError(f"Failed to read or parse corrupted dataset file at {filepath}: {e}")

    def load_raw_metadata(self, ticker: str) -> Dict[str, Any]:
        """Loads and returns the companion metadata JSON for the given raw ticker dataset."""
        filename = f"{ticker}.{self.file_format}"
        filepath = self.raw_dir / filename
        metadata_path = filepath.with_suffix(filepath.suffix + ".metadata.json")
        
        if not metadata_path.exists():
            raise StorageError(f"Metadata file for ticker {ticker} not found at {metadata_path}")

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to load metadata file at {metadata_path}: {e}")

    def raw_exists(self, ticker: str) -> bool:
        """Checks if both the data file and its companion metadata JSON exist in raw storage."""
        filename = f"{ticker}.{self.file_format}"
        filepath = self.raw_dir / filename
        metadata_path = filepath.with_suffix(filepath.suffix + ".metadata.json")
        return filepath.exists() and metadata_path.exists()

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """Saves a processed feature DataFrame to the processed directory."""
        filename = f"{name}.{self.file_format}"
        filepath = self.processed_dir / filename
        
        try:
            if self.file_format == "parquet":
                df.to_parquet(filepath, index=True)
            else:
                df.to_csv(filepath, index=True)
            logger.info(f"Saved processed dataset to: {filepath}")
            return filepath
        except Exception as e:
            raise StorageError(f"Failed to save processed dataset '{name}': {e}")

    def load_processed(self, name: str) -> pd.DataFrame:
        """Loads a processed dataset from local storage."""
        filename = f"{name}.{self.file_format}"
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            raise StorageError(f"Processed dataset '{name}' not found at {filepath}")

        try:
            if self.file_format == "parquet":
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            raise StorageError(f"Failed to read or parse processed dataset at {filepath}: {e}")

    def processed_exists(self, name: str) -> bool:
        """Checks if the processed dataset exists in local storage."""
        filename = f"{name}.{self.file_format}"
        return (self.processed_dir / filename).exists()

    def _write_metadata(self, filepath: Path, metadata: Dict[str, Any]) -> None:
        """Writes a companion JSON file for a given dataset path."""
        metadata_path = filepath.with_suffix(filepath.suffix + ".metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved dataset metadata companion to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to write metadata JSON file: {e}")
