"""
Feature engineering pipeline module.
Applies mathematical technical indicators to cleaned stock price DataFrames.
"""

import pandas as pd
from ai_engine.features import indicators

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends standard technical indicators directly to the stock price DataFrame.
    Operates on a copy of the input DataFrame to prevent unexpected side effects.
    
    The following columns are appended:
        - SMA_20, SMA_50: Simple Moving Averages
        - EMA_12, EMA_26: Exponential Moving Averages
        - RSI_14: Relative Strength Index (Wilder's smoothing)
        - MACD, MACD_Signal, MACD_Hist: Moving Average Convergence Divergence
        - BB_Middle, BB_Upper, BB_Lower: Bollinger Bands
        - ATR_14: Average True Range (Wilder's smoothing)
        - Daily_Returns: Simple percentage returns
        - Log_Returns: Continuously compounded returns
        - Rolling_Volatility: Annualized rolling standard deviation of log returns
        
    Args:
        df: Cleaned pandas DataFrame containing standard columns:
            ['Open', 'High', 'Low', 'Close', 'Volume']
            
    Returns:
        A new DataFrame copy containing the original price records and the computed indicators.
    """
    # Enforce copy to keep the pipeline side-effect free
    df_out = df.copy()
    
    # Extract numerical arrays for computational calls
    close = df_out["Close"].values
    high = df_out["High"].values
    low = df_out["Low"].values

    # 1. Simple Moving Averages
    df_out["SMA_20"] = indicators.compute_sma(close, 20)
    df_out["SMA_50"] = indicators.compute_sma(close, 50)

    # 2. Exponential Moving Averages
    df_out["EMA_12"] = indicators.compute_ema(close, 12)
    df_out["EMA_26"] = indicators.compute_ema(close, 26)

    # 3. Relative Strength Index
    df_out["RSI_14"] = indicators.compute_rsi(close, 14)

    # 4. Moving Average Convergence Divergence (MACD)
    macd, signal, hist = indicators.compute_macd(close, 12, 26, 9)
    df_out["MACD"] = macd
    df_out["MACD_Signal"] = signal
    df_out["MACD_Hist"] = hist

    # 5. Bollinger Bands
    bb_middle, bb_upper, bb_lower = indicators.compute_bollinger_bands(close, 20, 2.0)
    df_out["BB_Middle"] = bb_middle
    df_out["BB_Upper"] = bb_upper
    df_out["BB_Lower"] = bb_lower

    # 6. Average True Range (ATR)
    df_out["ATR_14"] = indicators.compute_atr(high, low, close, 14)

    # 7. Daily & Log Returns
    df_out["Daily_Returns"] = indicators.compute_daily_returns(close)
    df_out["Log_Returns"] = indicators.compute_log_returns(close)

    # 8. Annualized Rolling Volatility (21 trading days window, 252 annualized factor)
    # Volatility is calculated standardly on log returns
    log_returns = df_out["Log_Returns"].values
    df_out["Rolling_Volatility"] = indicators.compute_rolling_volatility(log_returns, 21, 252)

    return df_out
