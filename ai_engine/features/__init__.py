"""
Feature engineering module including technical indicators and parallel C/OpenMP extensions.
"""

from ai_engine.features.pipeline import add_technical_indicators
from ai_engine.features.engine import get_active_engine, is_openmp_available
from ai_engine.features.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_daily_returns,
    compute_log_returns,
    compute_rolling_volatility,
    HAS_OPENMP_ACCELERATION
)

__all__ = [
    "add_technical_indicators",
    "get_active_engine",
    "is_openmp_available",
    "compute_sma",
    "compute_ema",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_daily_returns",
    "compute_log_returns",
    "compute_rolling_volatility",
    "HAS_OPENMP_ACCELERATION"
]
