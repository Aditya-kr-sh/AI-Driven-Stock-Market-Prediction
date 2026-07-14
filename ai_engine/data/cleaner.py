"""
Data cleaning and validation engine for stock market datasets.
Standardizes headers, filters duplicates, fills missing data, and validates mathematical constraints.
"""

import pandas as pd
import numpy as np
from ai_engine.data.exceptions import ValidationError
from ai_engine.utils.logging import logger

class DataCleaner:
    """
    Data cleaner and validator that processes raw Yahoo Finance DataFrames,
    ensures proper date formatting, fills gaps, and enforces core validation constraints.
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by:
        1. Formitting the Date index.
        2. Dropping duplicate index values.
        3. Filling missing rows/values.
        4. Validating data types and mathematical boundaries.
        
        Args:
            df: Raw DataFrame downloaded from Yahoo Finance.
            
        Returns:
            A cleaned and validated DataFrame.
            
        Raises:
            ValidationError: If required columns are missing, data is corrupted, or numerical constraints are violated.
        """
        if df.empty:
            raise ValidationError("DataFrame is empty; cleaning cannot proceed.")

        # Create a copy to prevent SettingWithCopy warnings
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
        # Check if Date is a column or index
        if "Date" in df.columns:
            df = df.set_index("Date")
        
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValidationError(f"Failed to convert index to datetime: {e}")

        # Ensure index is timezone-naive to prevent compatibility issues (e.g. Parquet serialization)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Name index 'Date'
        df.index.name = "Date"

        # Sort chronologically
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
                # Attempt conversion
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
        # Constraint 1: Prices > 0
        price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in price_cols:
            invalid_price_count = (df[col] <= 0).sum()
            if invalid_price_count > 0:
                raise ValidationError(f"Found {invalid_price_count} records in column '{col}' with negative or zero price.")

        # Constraint 2: Volume >= 0
        invalid_volume_count = (df["Volume"] < 0).sum()
        if invalid_volume_count > 0:
            raise ValidationError(f"Found {invalid_volume_count} records in column 'Volume' with negative values.")

        # Constraint 3: High >= Low
        high_low_violation = (df["High"] < df["Low"]).sum()
        if high_low_violation > 0:
            # Let's inspect the violations
            violation_dates = df[df["High"] < df["Low"]].index.tolist()
            raise ValidationError(
                f"Found {high_low_violation} records where High price is less than Low price. "
                f"Dates: {violation_dates[:5]}"
            )
