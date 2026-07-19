"""
Feature engineering pipeline module.
Applies mathematical technical indicators to cleaned stock price DataFrames.
Switches dynamically between CPU Python and multithreaded OpenMP execution hooks.
"""

from typing import List, Optional
import pandas as pd
from ai_engine.features.engine import get_active_engine

def add_technical_indicators(df: pd.DataFrame, indicators_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Appends standard technical indicators directly to the stock price DataFrame.
    Operates on a copy of the input DataFrame to prevent unexpected side effects.
    
    Supported indicators:
        - SMA_20, SMA_50: Simple Moving Averages
        - EMA_12, EMA_26: Exponential Moving Averages
        - RSI_14: Relative Strength Index
        - MACD, MACD_Signal, MACD_Hist: Moving Average Convergence Divergence
        - BB_Middle, BB_Upper, BB_Lower: Bollinger Bands
        - ATR_14: Average True Range
        - OBV: On-Balance Volume
        - Momentum: Price Momentum
        - Daily_Returns: Simple percentage returns
        - Log_Returns: Continuously compounded returns
        - Rolling_Mean: 20-day rolling average
        - Rolling_Std: 20-day rolling standard deviation
        - Rolling_Volatility: Annualized volatility of log returns
        
    Args:
        df: Cleaned pandas DataFrame containing standard columns:
            ['Open', 'High', 'Low', 'Close', 'Volume']
        indicators_list: Optional list of indicator filter strings to compute.
                         If None (default), all indicators are computed.
            
    Returns:
        A new DataFrame copy containing the original price records and the computed indicators.
    """
    df_out = df.copy()
    
    close = df_out["Close"].values
    high = df_out["High"].values
    low = df_out["Low"].values
    volume = df_out["Volume"].values

    engine = get_active_engine()
    
    def should_calc(name: str) -> bool:
        if indicators_list is None:
            return True
        name_lower = name.lower()
        for ind in indicators_list:
            ind_lower = ind.lower()
            if ind_lower == name_lower:
                return True
            if name_lower.startswith(ind_lower + "_") or ind_lower.startswith(name_lower):
                return True
        return False

    # 1. Simple Moving Averages
    if should_calc("SMA"):
        df_out["SMA_20"] = engine.compute_sma(close, 20)
        df_out["SMA_50"] = engine.compute_sma(close, 50)

    # 2. Exponential Moving Averages
    if should_calc("EMA"):
        df_out["EMA_12"] = engine.compute_ema(close, 12)
        df_out["EMA_26"] = engine.compute_ema(close, 26)

    # 3. Relative Strength Index
    if should_calc("RSI"):
        df_out["RSI_14"] = engine.compute_rsi(close, 14)

    # 4. Moving Average Convergence Divergence (MACD)
    if should_calc("MACD"):
        macd, signal, hist = engine.compute_macd(close, 12, 26, 9)
        df_out["MACD"] = macd
        df_out["MACD_Signal"] = signal
        df_out["MACD_Hist"] = hist

    # 5. Bollinger Bands
    if should_calc("BB") or should_calc("Bollinger"):
        bb_middle, bb_upper, bb_lower = engine.compute_bollinger_bands(close, 20, 2.0)
        df_out["BB_Middle"] = bb_middle
        df_out["BB_Upper"] = bb_upper
        df_out["BB_Lower"] = bb_lower

    # 6. Average True Range (ATR)
    if should_calc("ATR"):
        df_out["ATR_14"] = engine.compute_atr(high, low, close, 14)

    # 7. On-Balance Volume (OBV)
    if should_calc("OBV"):
        df_out["OBV"] = engine.compute_obv(close, volume)

    # 8. Momentum
    if should_calc("Momentum"):
        df_out["Momentum"] = engine.compute_momentum(close, 10)

    # 9. Daily & Log Returns
    if should_calc("Daily_Returns"):
        df_out["Daily_Returns"] = engine.compute_daily_returns(close)
    if should_calc("Log_Returns") or should_calc("Rolling_Volatility"):
        df_out["Log_Returns"] = engine.compute_log_returns(close)

    # 10. Rolling Mean (SMA_20 alias / standalone)
    if should_calc("Rolling_Mean"):
        df_out["Rolling_Mean"] = engine.compute_rolling_mean(close, 20)

    # 11. Rolling Standard Deviation
    if should_calc("Rolling_Std"):
        df_out["Rolling_Std"] = engine.compute_rolling_std(close, 20)

    # 12. Annualized Rolling Volatility
    if should_calc("Rolling_Volatility") or should_calc("Volatility"):
        if "Log_Returns" not in df_out.columns:
            log_ret = engine.compute_log_returns(close)
        else:
            log_ret = df_out["Log_Returns"].values
        df_out["Rolling_Volatility"] = engine.compute_rolling_volatility(log_ret, 21, 252)

    return df_out
