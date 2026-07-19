"""
Data cleaning and validation engine for stock market datasets.
Standardizes headers, filters duplicates, fills missing data, and validates mathematical constraints.
Supports optional RAPIDS cuDF GPU acceleration when available.
"""

import pandas as pd
import numpy as np
from ai_engine.data.exceptions import ValidationError
from ai_engine.utils.logging import logger

try:
    import cudf
    CUDF_AVAILABLE = True
    logger.info("RAPIDS cuDF is available. GPU-accelerated dataframe cleaning active.")
except ImportError:
    CUDF_AVAILABLE = False

class DataCleaner:
    """
    Data cleaner and validator that processes raw Yahoo Finance DataFrames,
    ensures proper date formatting, fills gaps, and enforces core validation constraints.
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame. Uses cuDF if available.
        
        Args:
            df: Raw DataFrame downloaded from Yahoo Finance.
        """
        if df.empty:
            raise ValidationError("DataFrame is empty; cleaning cannot proceed.")

        if CUDF_AVAILABLE:
            try:
                return cls._clean_with_cudf(df)
            except Exception as e:
                logger.warning(f"cuDF cleaning failed: {e}. Falling back to Pandas.")
                
        return cls._clean_with_pandas(df)

    @classmethod
    def _clean_with_cudf(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Accelerated cleaning implementation using RAPIDS cuDF."""
        # Convert to GPU DataFrame
        gdf = cudf.DataFrame.from_pandas(df)
        
        # Standardize index
        if "Date" in gdf.columns:
            gdf = gdf.set_index("Date")
        
        # Sort and remove duplicates
        gdf = gdf.sort_index()
        gdf = gdf[~gdf.index.duplicated(keep="first")]
        
        # Check required columns
        for col in cls.REQUIRED_COLUMNS:
            if col not in gdf.columns:
                raise ValidationError(f"Missing required column in dataset: {col}")
                
        # Fill missing values
        gdf[cls.REQUIRED_COLUMNS] = gdf[cls.REQUIRED_COLUMNS].ffill().bfill()
        
        # Validate constraints
        # 1. Price columns must be positive
        price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in price_cols:
            if (gdf[col] <= 0).any():
                raise ValidationError(f"Found non-positive values in price column '{col}' on GPU.")
                
        # 2. Volume must be non-negative
        if (gdf["Volume"] < 0).any():
            raise ValidationError("Found negative volume records on GPU.")
            
        # 3. High must be >= Low
        if (gdf["High"] < gdf["Low"]).any():
            raise ValidationError("High price is less than Low price on GPU.")
            
        # Convert back to standard pandas
        cleaned_df = gdf.to_pandas()
        cleaned_df.index.name = "Date"
        cleaned_df.index = pd.to_datetime(cleaned_df.index)
        return cleaned_df

    @classmethod
    def _clean_with_pandas(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standard pandas cleaning implementation."""
        cleaned_df = df.copy()

        # 1. Standardize and Format Date Index
        cleaned_df = cls._standardize_date_index(cleaned_df)

        # 2. Check and Remove Duplicate Dates
        cleaned_df = cls._remove_duplicate_dates(cleaned_df)

        # 3. Verify Required Columns Existence
        cls._verify_required_columns(cleaned_df)

        # 4. Numeric Datatype Checks
        cls._validate_numeric_types(cleaned_df)

        # 5. Handle Missing/Null Values
        cleaned_df = cls._handle_missing_values(cleaned_df)

        # 6. Validate Mathematical & Financial Boundaries
        cls._validate_financial_constraints(cleaned_df)

        logger.info(f"DataFrame successfully cleaned and validated. Row count: {len(cleaned_df)}")
        return cleaned_df

    @classmethod
    def _standardize_date_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes the DataFrame index to be a chronological DatetimeIndex."""
        if "Date" in df.columns:
            df = df.set_index("Date")
        
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValidationError(f"Failed to convert index to datetime: {e}")

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index.name = "Date"
        df = df.sort_index()
        return df

    @classmethod
    def _remove_duplicate_dates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies and removes duplicate dates from the index."""
        duplicate_mask = df.index.duplicated(keep="first")
        duplicate_count = duplicate_mask.sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate date rows. Removing duplicates and keeping first.")
            df = df[~duplicate_mask]
        return df

    @classmethod
    def _verify_required_columns(cls, df: pd.DataFrame) -> None:
        """Verifies that all required yfinance columns exist in the DataFrame."""
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValidationError(f"Missing required columns in dataset: {missing}")

    @classmethod
    def _validate_numeric_types(cls, df: pd.DataFrame) -> None:
        """Ensures that all pricing and volume columns contain numeric types only."""
        for col in cls.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    raise ValidationError(f"Column '{col}' contains non-numeric data types and cannot be parsed.")

    @classmethod
    def _handle_missing_values(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using forward fill (to keep daily continuity)
        and falls back to backward fill for boundary values.
        """
        null_count = df[cls.REQUIRED_COLUMNS].isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Dataset contains {null_count} null cells. Performing forward-fill and backward-fill.")
            df[cls.REQUIRED_COLUMNS] = df[cls.REQUIRED_COLUMNS].ffill().bfill()

        # Re-check for remaining nulls
        remaining_nulls = df[cls.REQUIRED_COLUMNS].isnull().sum().sum()
        if remaining_nulls > 0:
            raise ValidationError(f"Dataset still contains {remaining_nulls} null values after fill operations.")
            
        return df

    @classmethod
    def _validate_financial_constraints(cls, df: pd.DataFrame) -> None:
        """
        Enforces strict mathematical relationships:
        - Open, High, Low, Close, Adj Close must be > 0.
        - Volume must be >= 0.
        - High must be >= Low.
        """
        price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in price_cols:
            invalid_price_count = (df[col] <= 0).sum()
            if invalid_price_count > 0:
                raise ValidationError(f"Found {invalid_price_count} records in column '{col}' with negative or zero price.")

        invalid_volume_count = (df["Volume"] < 0).sum()
        if invalid_volume_count > 0:
            raise ValidationError(f"Found {invalid_volume_count} records in column 'Volume' with negative values.")

        high_low_violation = (df["High"] < df["Low"]).sum()
        if high_low_violation > 0:
            violation_dates = df[df["High"] < df["Low"]].index.tolist()
            raise ValidationError(
                f"Found {high_low_violation} records where High price is less than Low price. "
                f"Dates: {violation_dates[:5]}"
            )
