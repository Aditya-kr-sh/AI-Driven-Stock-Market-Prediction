"""
Feature engineering pipeline module.
Applies mathematical technical indicators to cleaned stock price DataFrames.
"""

from typing import List, Optional
import pandas as pd
from ai_engine.features import indicators
from ai_engine.features.engine import get_active_engine

def add_technical_indicators(df: pd.DataFrame, indicators_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Appends standard technical indicators directly to the stock price DataFrame.
    Operates on a copy of the input DataFrame to prevent unexpected side effects.
    
    Supported indicators:
        - SMA_20, SMA_50: Simple Moving Averages (Python CPU)
        - EMA_12, EMA_26: Exponential Moving Averages (Python CPU)
        - RSI_14: Relative Strength Index (Wilder's smoothing) (Python CPU)
        - MACD, MACD_Signal, MACD_Hist: Moving Average Convergence Divergence (Python CPU)
        - BB_Middle, BB_Upper, BB_Lower: Bollinger Bands (Strategy: OpenMP/Python)
        - ATR_14: Average True Range (Strategy: OpenMP/Python)
        - Daily_Returns: Simple percentage returns (Python CPU)
        - Log_Returns: Continuously compounded returns (Python CPU)
        - Rolling_Volatility: Annualized volatility of log returns (Strategy: OpenMP/Python)
        
    Args:
        df: Cleaned pandas DataFrame containing standard columns:
            ['Open', 'High', 'Low', 'Close', 'Volume']
        indicators_list: Optional list of indicator filter strings to compute.
                         If None (default), all indicators are computed.
            
    Returns:
        A new DataFrame copy containing the original price records and the computed indicators.
    """
    # Enforce copy to keep the pipeline side-effect free
    df_out = df.copy()
    
    # Extract numerical arrays for computational calls
    close = df_out["Close"].values
    high = df_out["High"].values
    low = df_out["Low"].values

    engine = get_active_engine()
    
    # Helper to check if a specific indicator should be computed
    def should_calc(name: str) -> bool:
        if indicators_list is None:
            return True
        name_lower = name.lower()
        for ind in indicators_list:
            ind_lower = ind.lower()
            if ind_lower == name_lower:
                return True
            # Match common prefixes (e.g. 'SMA' matches 'SMA_20')
            if name_lower.startswith(ind_lower + "_") or ind_lower.startswith(name_lower):
                return True
        return False

    # 1. Simple Moving Averages
    if should_calc("SMA"):
        df_out["SMA_20"] = indicators.compute_sma(close, 20)
        df_out["SMA_50"] = indicators.compute_sma(close, 50)

    # 2. Exponential Moving Averages
    if should_calc("EMA"):
        df_out["EMA_12"] = indicators.compute_ema(close, 12)
        df_out["EMA_26"] = indicators.compute_ema(close, 26)

    # 3. Relative Strength Index
    if should_calc("RSI"):
        df_out["RSI_14"] = indicators.compute_rsi(close, 14)

    # 4. Moving Average Convergence Divergence (MACD)
    if should_calc("MACD"):
        macd, signal, hist = indicators.compute_macd(close, 12, 26, 9)
        df_out["MACD"] = macd
        df_out["MACD_Signal"] = signal
        df_out["MACD_Hist"] = hist

    # 5. Bollinger Bands (Delegated to Strategy Engine: OMP or Python Fallback)
    if should_calc("BB") or should_calc("Bollinger"):
        bb_middle, bb_upper, bb_lower = engine.compute_bollinger_bands(close, 20, 2.0)
        df_out["BB_Middle"] = bb_middle
        df_out["BB_Upper"] = bb_upper
        df_out["BB_Lower"] = bb_lower

    # 6. Average True Range (ATR) (Delegated to Strategy Engine: OMP or Python Fallback)
    if should_calc("ATR"):
        df_out["ATR_14"] = engine.compute_atr(high, low, close, 14)

    # 7. Daily & Log Returns
    if should_calc("Daily_Returns"):
        df_out["Daily_Returns"] = indicators.compute_daily_returns(close)
    if should_calc("Log_Returns") or should_calc("Rolling_Volatility"):
        df_out["Log_Returns"] = indicators.compute_log_returns(close)

    # 8. Annualized Rolling Volatility (Delegated to Strategy Engine: OMP or Python Fallback)
    if should_calc("Rolling_Volatility") or should_calc("Volatility"):
        if "Log_Returns" not in df_out.columns:
            log_ret = indicators.compute_log_returns(close)
        else:
            log_ret = df_out["Log_Returns"].values
        df_out["Rolling_Volatility"] = engine.compute_rolling_volatility(log_ret, 21, 252)

    return df_out
