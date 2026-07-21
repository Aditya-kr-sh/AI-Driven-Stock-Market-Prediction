import os
from pathlib import Path
from typing import List
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

# Base Directory of the workspace (stockproject/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Central Configuration management utilizing Pydantic Settings.
    Environment variables are automatically parsed from a local .env file.
    """
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # =========================================================================
    # Environment & Logging Configuration
    # =========================================================================
    ENV: str = Field(default="development", description="Execution environment (development, production, test)")
    DEBUG: bool = Field(default=True, description="Enable debug level outputs")
    LOG_LEVEL: str = Field(default="INFO", description="Global logging filter level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging layout string"
    )

    # =========================================================================
    # Project Paths
    # =========================================================================
    BASE_PATH: Path = BASE_DIR
    MODEL_DIR: Path = BASE_DIR / "saved_models"
    DATABASE_URL: str = Field(
        default=f"sqlite:///{BASE_DIR}/stock_market.db",
        description="SQLAlchemy database URI (SQLite default, PostgreSQL compatible)"
    )
    
    # Raw and processed data storage folders
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "stocks"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    BENCHMARK_RESULTS_PATH: Path = BASE_DIR / "data" / "benchmark_results.json"

    # =========================================================================
    # Storage Format Options
    # =========================================================================
    # Supports "csv" or "parquet"
    STORAGE_FORMAT: str = Field(default="csv", description="File storage format (csv or parquet)")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    RANDOM_SEED: int = Field(default=42, description="Global reproducibility random seed")

    # =========================================================================
    # Device Selection
    # =========================================================================
    # Fallback to GPU if CUDA is available, otherwise CPU
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Default target device for DL workloads (cpu/cuda)"
    )

    # =========================================================================
    # Yahoo Finance & Data Settings
    # =========================================================================
    # Imports list of all 50 tickers directly from the tickers module
    TICKERS: List[str] = Field(
        default_factory=list,
        description="Centralized NIFTY 50 stock tickers list"
    )
    DEFAULT_START_DATE: str = Field(default="2018-01-01", description="yfinance start date")
    DEFAULT_END_DATE: str = Field(default="2026-01-01", description="yfinance end date")
    MAX_BENCHMARK_RECORDS: int = Field(default=100, description="Max benchmark records retained on disk")

    # =========================================================================
    # AI Models Training Configuration
    # =========================================================================
    TRAIN_TEST_SPLIT: float = Field(default=0.8, description="Data split percentage for train")
    
    # LSTM Hyperparameters
    LSTM_EPOCHS: int = Field(default=50, description="LSTM train iterations")
    LSTM_BATCH_SIZE: int = Field(default=64, description="LSTM batch size")
    LSTM_LEARNING_RATE: float = Field(default=0.001, description="LSTM gradient step multiplier")
    LSTM_HIDDEN_SIZE: int = Field(default=64, description="LSTM internal hidden size")
    LSTM_NUM_LAYERS: int = Field(default=2, description="LSTM depth size")
    LSTM_SEQUENCE_LENGTH: int = Field(default=30, description="LSTM lookback days")

    # Transformer Hyperparameters
    TRANSFORMER_EPOCHS: int = Field(default=50, description="Transformer train iterations")
    TRANSFORMER_BATCH_SIZE: int = Field(default=64, description="Transformer batch size")
    TRANSFORMER_LEARNING_RATE: float = Field(default=0.0005, description="Transformer gradient step multiplier")
    TRANSFORMER_D_MODEL: int = Field(default=64, description="Transformer token dimensions")
    TRANSFORMER_NHEAD: int = Field(default=4, description="Transformer multihead attention counts")
    TRANSFORMER_NUM_LAYERS: int = Field(default=2, description="Transformer encoder layers depth")
    TRANSFORMER_SEQUENCE_LENGTH: int = Field(default=30, description="Transformer lookback days")

    # XGBoost Hyperparameters
    XGB_MAX_DEPTH: int = Field(default=6, description="XGBoost tree depth limit")
    XGB_N_ESTIMATORS: int = Field(default=100, description="XGBoost boosting round numbers")
    XGB_LEARNING_RATE: float = Field(default=0.05, description="XGBoost shrinkage multiplier")

    # =========================================================================
    # Portfolio Optimization Configuration
    # =========================================================================
    PORTFOLIO_MC_SIMULATIONS: int = Field(default=10000, description="Monte Carlo execution loops")
    RISK_FREE_RATE: float = Field(default=0.07, description="Annualized risk free rate (7% Default)")
    MIN_COMMON_TRADING_DAYS: int = Field(default=252, description="Minimum required overlapping trading days for portfolio optimization")
    DOWNLOAD_WORKERS: int = Field(default=5, description="Number of parallel workers for ticker downloads (configurable via env var)")

    @model_validator(mode="after")
    def populate_tickers(self) -> "Settings":
        """Lazily populates NIFTY 50 tickers list from tickers.py if empty."""
        if not self.TICKERS:
            from ai_engine.data.tickers import NIFTY_50_TICKERS
            self.TICKERS = NIFTY_50_TICKERS
        return self

# Instantiate Settings Object
settings = Settings()
