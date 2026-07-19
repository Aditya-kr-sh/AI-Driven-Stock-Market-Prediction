"""
Technical Indicator Computation Strategy Engines.
Provides unified interfaces for running quantitative calculations on NumPy arrays,
allowing seamless switching between pure Python/NumPy and Cython/OpenMP.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class IndicatorEngine(ABC):
    """Abstract base interface defining the strategy pattern contract for indicators."""
    
    @abstractmethod
    def set_threads(self, num_threads: int) -> None:
        """Sets the thread limit for parallel calculation blocks (no-op for Python)."""
        pass

    @abstractmethod
    def compute_sma(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculates Simple Moving Average (SMA)."""
        pass

    @abstractmethod
    def compute_ema(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculates Exponential Moving Average (EMA)."""
        pass

    @abstractmethod
    def compute_rsi(self, values: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculates Relative Strength Index (RSI)."""
        pass

    @abstractmethod
    def compute_macd(
        self, 
        values: np.ndarray, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates MACD (MACD line, Signal line, Histogram)."""
        pass

    @abstractmethod
    def compute_bollinger_bands(
        self, 
        values: np.ndarray, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates Bollinger Bands (Middle, Upper, Lower)."""
        pass

    @abstractmethod
    def compute_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """Calculates Average True Range (ATR)."""
        pass

    @abstractmethod
    def compute_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculates On-Balance Volume (OBV)."""
        pass

    @abstractmethod
    def compute_momentum(self, values: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculates Price Momentum."""
        pass

    @abstractmethod
    def compute_daily_returns(self, close: np.ndarray) -> np.ndarray:
        """Calculates Simple Daily Returns."""
        pass

    @abstractmethod
    def compute_log_returns(self, close: np.ndarray) -> np.ndarray:
        """Calculates continuously compounded Log Returns."""
        pass

    @abstractmethod
    def compute_rolling_mean(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculates Rolling Mean (identical to SMA)."""
        pass

    @abstractmethod
    def compute_rolling_std(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculates Rolling Standard Deviation."""
        pass

    @abstractmethod
    def compute_rolling_volatility(
        self, 
        returns: np.ndarray, 
        period: int = 21, 
        trading_days: int = 252
    ) -> np.ndarray:
        """Calculates annualized rolling volatility."""
        pass


class PythonEngine(IndicatorEngine):
    """Pure Python / NumPy indicator calculations strategy implementation."""
    
    def set_threads(self, num_threads: int) -> None:
        pass

    def compute_sma(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        from ai_engine.features.indicators import compute_sma as py_sma
        return py_sma(values, period)

    def compute_ema(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        from ai_engine.features.indicators import compute_ema as py_ema
        return py_ema(values, period)

    def compute_rsi(self, values: np.ndarray, period: int = 14) -> np.ndarray:
        from ai_engine.features.indicators import compute_rsi as py_rsi
        return py_rsi(values, period)

    def compute_macd(
        self, 
        values: np.ndarray, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ai_engine.features.indicators import compute_macd as py_macd
        return py_macd(values, fast_period, slow_period, signal_period)

    def compute_bollinger_bands(
        self, 
        values: np.ndarray, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ai_engine.features.indicators import compute_bollinger_bands as py_bb
        return py_bb(values, period, num_std)

    def compute_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        from ai_engine.features.indicators import compute_atr as py_atr
        return py_atr(high, low, close, period)

    def compute_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        from ai_engine.features.indicators import compute_obv as py_obv
        return py_obv(close, volume)

    def compute_momentum(self, values: np.ndarray, period: int = 10) -> np.ndarray:
        from ai_engine.features.indicators import compute_momentum as py_mom
        return py_mom(values, period)

    def compute_daily_returns(self, close: np.ndarray) -> np.ndarray:
        from ai_engine.features.indicators import compute_daily_returns as py_ret
        return py_ret(close)

    def compute_log_returns(self, close: np.ndarray) -> np.ndarray:
        from ai_engine.features.indicators import compute_log_returns as py_lret
        return py_lret(close)

    def compute_rolling_mean(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        return self.compute_sma(values, period)

    def compute_rolling_std(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        from ai_engine.features.indicators import compute_rolling_std as py_rstd
        return py_rstd(values, period)

    def compute_rolling_volatility(
        self, 
        returns: np.ndarray, 
        period: int = 21, 
        trading_days: int = 252
    ) -> np.ndarray:
        from ai_engine.features.indicators import compute_rolling_volatility as py_vol
        return py_vol(returns, period, trading_days)


class OpenMPEngine(IndicatorEngine):
    """OpenMP multi-threaded parallel indicators strategy implementation."""
    
    def __init__(self, backend_module):
        self.backend = backend_module
        self._threads = 1  # Default to single thread baseline
        
    def set_threads(self, num_threads: int) -> None:
        """Sets the number of parallel OpenMP threads globally in the backend."""
        self._threads = num_threads
        if hasattr(self.backend, "set_openmp_threads"):
            self.backend.set_openmp_threads(num_threads)

    def _to_c_array(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.float64)

    def compute_sma(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        c_vals = self._to_c_array(values)
        return self.backend.compute_sma_omp(c_vals, period, self._threads)

    def compute_ema(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        c_vals = self._to_c_array(values)
        return self.backend.compute_ema_omp(c_vals, period, self._threads)

    def compute_rsi(self, values: np.ndarray, period: int = 14) -> np.ndarray:
        c_vals = self._to_c_array(values)
        return self.backend.compute_rsi_omp(c_vals, period, self._threads)

    def compute_macd(
        self, 
        values: np.ndarray, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_vals = self._to_c_array(values)
        return self.backend.compute_macd_omp(
            c_vals, fast_period, slow_period, signal_period, self._threads
        )

    def compute_bollinger_bands(
        self, 
        values: np.ndarray, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_vals = self._to_c_array(values)
        middle_band, upper_band, lower_band = self.backend.compute_bollinger_bands_omp(
            c_vals, period, num_std, self._threads
        )
        return middle_band, upper_band, lower_band

    def compute_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        c_high = self._to_c_array(high)
        c_low = self._to_c_array(low)
        c_close = self._to_c_array(close)
        return self.backend.compute_atr_omp(
            c_high, c_low, c_close, period, self._threads
        )

    def compute_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        c_close = self._to_c_array(close)
        c_vol = self._to_c_array(volume)
        return self.backend.compute_obv_omp(c_close, c_vol, self._threads)

    def compute_momentum(self, values: np.ndarray, period: int = 10) -> np.ndarray:
        c_vals = self._to_c_array(values)
        return self.backend.compute_momentum_omp(c_vals, period, self._threads)

    def compute_daily_returns(self, close: np.ndarray) -> np.ndarray:
        c_close = self._to_c_array(close)
        return self.backend.compute_daily_returns_omp(c_close, self._threads)

    def compute_log_returns(self, close: np.ndarray) -> np.ndarray:
        c_close = self._to_c_array(close)
        return self.backend.compute_log_returns_omp(c_close, self._threads)

    def compute_rolling_mean(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        return self.compute_sma(values, period)

    def compute_rolling_std(self, values: np.ndarray, period: int = 20) -> np.ndarray:
        c_vals = self._to_c_array(values)
        return self.backend.compute_rolling_std_omp(c_vals, period, self._threads)

    def compute_rolling_volatility(
        self, 
        returns: np.ndarray, 
        period: int = 21, 
        trading_days: int = 252
    ) -> np.ndarray:
        c_returns = self._to_c_array(returns)
        return self.backend.compute_rolling_volatility_omp(
            c_returns, period, trading_days, self._threads
        )


# Global package acceleration hook selection
_ACTIVE_ENGINE: IndicatorEngine = PythonEngine()
_OPENMP_AVAILABLE = False

try:
    # On Windows, add PATH dirs containing DLL dependencies
    import sys
    import os
    if sys.platform.startswith("win32") and hasattr(os, "add_dll_directory"):
        for p in os.environ.get("PATH", "").split(";"):
            p = p.strip()
            if p and os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass

    # Try importing Cython compiled openmp_backend extension
    import ai_engine.features.openmp_backend as omp
    _ACTIVE_ENGINE = OpenMPEngine(omp)
    _OPENMP_AVAILABLE = True
except ImportError:
    pass

def get_active_engine() -> IndicatorEngine:
    """Returns the currently configured indicator strategy engine (OpenMP or Python fallback)."""
    return _ACTIVE_ENGINE

def is_openmp_available() -> bool:
    """Returns whether the OpenMP Cython extension was successfully compiled and loaded."""
    return _OPENMP_AVAILABLE
