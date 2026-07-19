"""
Pure Python/NumPy implementations of technical indicators.
Designed to be modular so that implementations can be swapped with parallel O(N) 
OpenMP extensions in future phases without modifying public APIs.
"""

import numpy as np
from typing import Tuple

# Global flag to track if C/OpenMP acceleration is active
HAS_OPENMP_ACCELERATION = False

# Reserved hook for loading OpenMP compiled libraries in future phases
try:
    # Placeholder for Phase 3B compiled Cython/C extensions:
    # from ai_engine.features._openmp_backend import compute_sma_omp, compute_ema_omp, ...
    pass
except ImportError:
    pass


def compute_sma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).
    Time Complexity: O(N) sliding window sum.
    """
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[:period - 1] = np.nan
    if n < period:
        return out

    # Compute first window sum
    window_sum = float(np.sum(values[:period]))
    out[period - 1] = window_sum / period

    # Sliding window O(N) updates
    for i in range(period, n):
        window_sum += float(values[i] - values[i - period])
        out[i] = window_sum / period

    return out


def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    Time Complexity: O(N)
    """
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[:period - 1] = np.nan
    if n < period:
        return out

    # Standard practice: Seed EMA with initial SMA value
    sma_seed = float(np.mean(values[:period]))
    out[period - 1] = sma_seed

    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = float(values[i] * alpha + out[i - 1] * (1.0 - alpha))

    return out


def compute_rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) using Wilder's smoothing.
    Time Complexity: O(N)
    """
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[:period] = np.nan
    if n <= period:
        return out

    # Compute daily gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        diff = float(values[i] - values[i - 1])
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    # Initial average gains and losses (SMA)
    avg_gain = float(np.mean(gains[1:period + 1]))
    avg_loss = float(np.mean(losses[1:period + 1]))

    # Store first RSI calculation
    if avg_loss == 0.0:
        out[period] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's O(N) smoothed calculation loop
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            out[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


def compute_macd(
    values: np.ndarray, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    Returns:
        macd_line: Fast EMA - Slow EMA
        signal_line: EMA(macd_line, signal_period)
        histogram: macd_line - signal_line
    """
    fast_ema = compute_ema(values, fast_period)
    slow_ema = compute_ema(values, slow_period)
    macd_line = fast_ema - slow_ema

    # Signal line is EMA of the MACD line
    valid_start = slow_period - 1
    signal_line = np.full_like(macd_line, np.nan)

    if len(values) >= (slow_period + signal_period - 1):
        # Slice valid MACD parts to avoid seeding EMA with initial NaNs
        macd_valid = macd_line[valid_start:]
        signal_valid = compute_ema(macd_valid, signal_period)
        signal_line[valid_start:] = signal_valid

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    values: np.ndarray, 
    period: int = 20, 
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands (Middle Band, Upper Band, Lower Band).
    Time Complexity: O(N) rolling statistics.
    """
    n = len(values)
    middle_band = compute_sma(values, period)

    rolling_std = np.empty(n, dtype=np.float64)
    rolling_std[:period - 1] = np.nan

    # Calculate sliding window standard deviation
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        rolling_std[i] = float(np.std(window))

    upper_band = middle_band + num_std * rolling_std
    lower_band = middle_band - num_std * rolling_std

    return middle_band, upper_band, lower_band


def compute_atr(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray, 
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR) using Wilder's True Range smoothing.
    Time Complexity: O(N)
    """
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    out[:period] = np.nan
    if n <= period:
        return out

    # Compute True Range (TR)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = float(max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        ))

    # Seed ATR with the mean of the first 'period' elements
    atr = float(np.mean(tr[:period]))
    out[period - 1] = atr

    # Smooth TR over the remaining series
    for i in range(period, n):
        atr = (atr * (period - 1) + tr[i]) / period
        out[i] = atr

    return out


def compute_daily_returns(close: np.ndarray) -> np.ndarray:
    """Calculate Simple Daily Returns percentage change."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan
    for i in range(1, n):
        out[i] = float((close[i] - close[i - 1]) / close[i - 1])
    return out


def compute_log_returns(close: np.ndarray) -> np.ndarray:
    """Calculate Log Returns."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan
    for i in range(1, n):
        out[i] = float(np.log(close[i] / close[i - 1]))
    return out


def compute_rolling_volatility(
    returns: np.ndarray, 
    period: int = 21, 
    trading_days: int = 252
) -> np.ndarray:
    """
    Calculate Rolling Volatility (annualized standard deviation of returns).
    """
    n = len(returns)
    out = np.empty(n, dtype=np.float64)
    out[:period - 1] = np.nan

    for i in range(period - 1, n):
        window = returns[i - period + 1 : i + 1]
        if np.isnan(window).any():
            out[i] = np.nan
        else:
            out[i] = float(np.std(window) * np.sqrt(trading_days))

    return out


def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Calculate On-Balance Volume (OBV) cumulative technical indicator."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


def compute_momentum(values: np.ndarray, period: int = 10) -> np.ndarray:
    """Calculate price Momentum over a given period."""
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[:period] = np.nan
    for i in range(period, n):
        out[i] = float(values[i] - values[i - period])
    return out


def compute_rolling_std(values: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate simple rolling standard deviation."""
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[:period - 1] = np.nan
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        out[i] = float(np.std(window))
    return out

