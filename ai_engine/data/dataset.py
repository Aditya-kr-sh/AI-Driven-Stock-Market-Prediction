"""
Core StockDataset class representing a standardized stock price history dataset.
Wraps the pandas DataFrame and its companion metadata in a clean, typed OOP container.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd

class StockDataset:
    """
    Object-oriented container representing a single stock's historical data.
    Provides easy access to dataset attributes, date bounds, features, and metadata.
    """

    def __init__(self, ticker: str, df: pd.DataFrame, metadata: Dict[str, Any] = None):
        """
        Initializes the StockDataset container.
        
        Args:
            ticker: The stock ticker (e.g. 'RELIANCE.NS').
            df: Standardized, validated price history DataFrame.
            metadata: Structured metadata companion dictionary.
        """
        self._ticker = ticker.strip().upper()
        self._df = df
        self._metadata = metadata or {}

    @property
    def ticker(self) -> str:
        """Returns the uppercase stock ticker symbol."""
        return self._ticker

    @property
    def df(self) -> pd.DataFrame:
        """Returns the underlying pandas DataFrame containing columns: Open, High, Low, Close, Adj Close, Volume."""
        return self._df

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the structured companion metadata dict (ISO timestamps, source, format)."""
        return self._metadata

    @property
    def columns(self) -> List[str]:
        """Returns list of column names in the dataset."""
        return list(self._df.columns)

    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Returns the first and last timestamps covered by this dataset.
        
        Returns:
            Tuple of (start_date, end_date) as pandas Timestamps.
        """
        if self._df.empty:
            raise ValueError(f"Cannot determine date range: StockDataset '{self._ticker}' is empty.")
        return self._df.index[0], self._df.index[-1]

    @property
    def start_date_str(self) -> str:
        """Returns the starting date of the dataset as a string (YYYY-MM-DD)."""
        return self.date_range[0].strftime("%Y-%m-%d")

    @property
    def end_date_str(self) -> str:
        """Returns the ending date of the dataset as a string (YYYY-MM-DD)."""
        return self.date_range[1].strftime("%Y-%m-%d")

    def __len__(self) -> int:
        """Returns the number of trading rows in the dataset."""
        return len(self._df)

    def __repr__(self) -> str:
        rows = len(self)
        if rows > 0:
            start_str = self.start_date_str
            end_str = self.end_date_str
            return f"StockDataset(ticker='{self._ticker}', rows={rows}, range=[{start_str} to {end_str}])"
        return f"StockDataset(ticker='{self._ticker}', rows=0)"
