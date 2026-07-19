"""
Technical Indicator Computation Strategy Engines.
Provides unified interfaces for running quantitative calculations on NumPy arrays,
allowing seamless switching between pure Python/NumPy, Cython/OpenMP, and future CUDA implementations.
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
    def compute_bollinger_bands(
        self, 
        values: np.ndarray, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates Bollinger Bands (Middle, Upper, Lower)."""
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


class PythonEngine(IndicatorEngine):
    """Pure Python / NumPy indicator calculations strategy implementation."""
    
    def set_threads(self, num_threads: int) -> None:
        # NumPy/Python sequential fallback is a no-op for threading controls
        pass

    def compute_bollinger_bands(
        self, 
        values: np.ndarray, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ai_engine.features.indicators import compute_bollinger_bands as py_bb
        return py_bb(values, period, num_std)

    def compute_rolling_volatility(
        self, 
        returns: np.ndarray, 
        period: int = 21, 
        trading_days: int = 252
    ) -> np.ndarray:
        from ai_engine.features.indicators import compute_rolling_volatility as py_vol
        return py_vol(returns, period, trading_days)

    def compute_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        from ai_engine.features.indicators import compute_atr as py_atr
        return py_atr(high, low, close, period)


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

    def compute_bollinger_bands(
        self, 
        values: np.ndarray, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Cython OpenMP backend expects contiguous float64 memory views
        c_vals = np.ascontiguousarray(values, dtype=np.float64)
        middle_band, upper_band, lower_band = self.backend.compute_bollinger_bands_omp(
            c_vals, period, num_std, self._threads
        )
        return middle_band, upper_band, lower_band

    def compute_rolling_volatility(
        self, 
        returns: np.ndarray, 
        period: int = 21, 
        trading_days: int = 252
    ) -> np.ndarray:
        c_returns = np.ascontiguousarray(returns, dtype=np.float64)
        return self.backend.compute_rolling_volatility_omp(
            c_returns, period, trading_days, self._threads
        )

    def compute_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        c_high = np.ascontiguousarray(high, dtype=np.float64)
        c_low = np.ascontiguousarray(low, dtype=np.float64)
        c_close = np.ascontiguousarray(close, dtype=np.float64)
        return self.backend.compute_atr_omp(
            c_high, c_low, c_close, period, self._threads
        )


# Global package acceleration hook selection
_ACTIVE_ENGINE: IndicatorEngine = PythonEngine()
_OPENMP_AVAILABLE = False

try:
    # On Windows, Python 3.8+ does not search PATH for DLLs, so we must add existing folders
    # on PATH containing DLL dependencies (like MinGW/MSYS) manually to prevent load failures.
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
