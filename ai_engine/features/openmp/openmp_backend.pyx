# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Cython OpenMP accelerated implementations of heavy rolling technical indicators.
Releases Python GIL and maps directly to double-precision raw memory views.
Uses const double[:] read-only memory views to interface directly with Pandas columns.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt, log

# Resolve NaN access within nogil blocks by using a pre-declared C-double constant
cdef double c_nan = float('nan')
cdef int GLOBAL_NUM_THREADS = 1

def set_openmp_threads(int num_threads):
    """Sets global default thread count for OpenMP loops."""
    global GLOBAL_NUM_THREADS
    GLOBAL_NUM_THREADS = num_threads


# =========================================================================
# Inline C Helper Functions (Thread-Private, No GIL, Const Views)
# =========================================================================

cdef inline double c_mean(const double[:] values, int start, int end) nogil:
    """Computes the mean of a slice in a thread-private context."""
    cdef double sum_val = 0.0
    cdef int k
    for k in range(start, end):
        sum_val += values[k]
    return sum_val / (end - start)


cdef inline double c_std(const double[:] values, int start, int end, double mean_val) nogil:
    """Computes the standard deviation of a slice in a thread-private context."""
    cdef double sq_diff_sum = 0.0
    cdef int k
    for k in range(start, end):
        sq_diff_sum += (values[k] - mean_val) * (values[k] - mean_val)
    return sqrt(sq_diff_sum / (end - start))


cdef inline double c_volatility(
    const double[:] returns, 
    int start, 
    int end, 
    double annualize_factor
) nogil:
    """Computes annualized standard deviation of returns in a thread-private context."""
    cdef double sum_val = 0.0
    cdef double mean_val = 0.0
    cdef double sq_diff_sum = 0.0
    cdef int k
    
    for k in range(start, end):
        sum_val += returns[k]
    mean_val = sum_val / (end - start)
    
    for k in range(start, end):
        sq_diff_sum += (returns[k] - mean_val) * (returns[k] - mean_val)
        
    return sqrt(sq_diff_sum / (end - start)) * annualize_factor


# =========================================================================
# Parallel Indicators Functions (Accepting Const Views)
# =========================================================================

def compute_sma_omp(const double[:] values, int period, int num_threads):
    """Parallel Simple Moving Average (SMA) calculation using OpenMP."""
    cdef int n = values.shape[0]
    cdef cnp.ndarray[double, ndim=1] sma_arr = np.empty(n, dtype=np.float64)
    cdef double[:] sma = sma_arr
    cdef int i
    
    for i in range(period - 1):
        sma[i] = c_nan
        
    if n < period:
        return sma_arr
        
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    for i in prange(period - 1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        sma[i] = c_mean(values, i - period + 1, i + 1)
        
    return sma_arr


def compute_ema_omp(const double[:] values, int period, int num_threads):
    """Exponential Moving Average (EMA) calculation with C loop optimizations."""
    cdef int n = values.shape[0]
    cdef cnp.ndarray[double, ndim=1] ema_arr = np.empty(n, dtype=np.float64)
    cdef double[:] ema = ema_arr
    cdef int i
    
    for i in range(period - 1):
        ema[i] = c_nan
        
    if n < period:
        return ema_arr
        
    cdef double alpha = 2.0 / (period + 1)
    
    # Seeding with simple mean
    cdef double sum_val = 0.0
    for i in range(period):
        sum_val += values[i]
    cdef double current_ema = sum_val / period
    ema[period - 1] = current_ema
    
    # Sequential computation due to recursive temporal dependency
    for i in range(period, n):
        current_ema = (values[i] * alpha) + (current_ema * (1.0 - alpha))
        ema[i] = current_ema
        
    return ema_arr


def compute_rsi_omp(const double[:] values, int period, int num_threads):
    """Parallel RSI computation using Wilder's smoothing loop structures."""
    cdef int n = values.shape[0]
    cdef cnp.ndarray[double, ndim=1] rsi_arr = np.empty(n, dtype=np.float64)
    cdef double[:] rsi = rsi_arr
    cdef int i
    
    for i in range(period):
        rsi[i] = c_nan
        
    if n <= period:
        return rsi_arr
        
    cdef cnp.ndarray[double, ndim=1] gains_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] losses_arr = np.empty(n, dtype=np.float64)
    cdef double[:] gains = gains_arr
    cdef double[:] losses = losses_arr
    
    gains[0] = 0.0
    losses[0] = 0.0
    
    cdef double diff
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    
    # 1. Parallel changes separation loop
    for i in prange(1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gains[i] = diff
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -diff
            
    # 2. Seeding averages
    cdef double sum_gain = 0.0
    cdef double sum_loss = 0.0
    for i in range(1, period + 1):
        sum_gain += gains[i]
        sum_loss += losses[i]
        
    cdef double avg_gain = sum_gain / period
    cdef double avg_loss = sum_loss / period
    
    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
        
    # 3. Smoothed sequential loops
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
            
    return rsi_arr


def compute_macd_omp(
    const double[:] values, 
    int fast_period, 
    int slow_period, 
    int signal_period, 
    int num_threads
):
    """Computes MACD, signal line, and histogram using OpenMP optimized EMAs."""
    cdef int n = values.shape[0]
    
    # Pre-allocate output arrays
    cdef cnp.ndarray[double, ndim=1] macd_line_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] signal_line_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] hist_arr = np.empty(n, dtype=np.float64)
    
    cdef double[:] macd_line = macd_line_arr
    cdef double[:] signal_line = signal_line_arr
    cdef double[:] hist = hist_arr
    
    # Compute EMAs
    cdef cnp.ndarray[double, ndim=1] ema_fast_arr = compute_ema_omp(values, fast_period, num_threads)
    cdef cnp.ndarray[double, ndim=1] ema_slow_arr = compute_ema_omp(values, slow_period, num_threads)
    
    cdef double[:] ema_fast = ema_fast_arr
    cdef double[:] ema_slow = ema_slow_arr
    
    cdef int i
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    
    # 1. Parallel compute MACD line
    for i in prange(n, nogil=True, num_threads=threads_to_use, schedule='static'):
        if ema_fast[i] != ema_fast[i] or ema_slow[i] != ema_slow[i]:
            macd_line[i] = c_nan
        else:
            macd_line[i] = ema_fast[i] - ema_slow[i]
            
    # Remove leading NaNs for signal EMA seeding
    cdef int start_idx = slow_period - 1
    if n <= start_idx + signal_period:
        for i in range(n):
            signal_line[i] = c_nan
            hist[i] = c_nan
        return macd_line_arr, signal_line_arr, hist_arr
        
    cdef double alpha_sig = 2.0 / (signal_period + 1)
    for i in range(start_idx + signal_period - 1):
        signal_line[i] = c_nan
        
    cdef double sum_macd = 0.0
    for i in range(start_idx, start_idx + signal_period):
        sum_macd += macd_line[i]
    cdef double current_signal = sum_macd / signal_period
    signal_line[start_idx + signal_period - 1] = current_signal
    
    # 2. Sequential Wilder's EMA loop for signal line
    for i in range(start_idx + signal_period, n):
        current_signal = (macd_line[i] * alpha_sig) + (current_signal * (1.0 - alpha_sig))
        signal_line[i] = current_signal
        
    # 3. Parallel compute Histogram loop
    for i in prange(n, nogil=True, num_threads=threads_to_use, schedule='static'):
        if macd_line[i] != macd_line[i] or signal_line[i] != signal_line[i]:
            hist[i] = c_nan
        else:
            hist[i] = macd_line[i] - signal_line[i]
            
    return macd_line_arr, signal_line_arr, hist_arr


def compute_bollinger_bands_omp(
    const double[:] values, 
    int period, 
    double num_std, 
    int num_threads
):
    """
    Parallel Bollinger Bands calculation utilizing OpenMP.
    Loops are executed in parallel without Python GIL.
    """
    cdef int n = values.shape[0]
    
    cdef cnp.ndarray[double, ndim=1] middle_band_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] upper_band_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] lower_band_arr = np.empty(n, dtype=np.float64)
    
    cdef double[:] middle_band = middle_band_arr
    cdef double[:] upper_band = upper_band_arr
    cdef double[:] lower_band = lower_band_arr
    
    cdef int i
    
    for i in range(period - 1):
        middle_band[i] = c_nan
        upper_band[i] = c_nan
        lower_band[i] = c_nan
        
    if n < period:
        return middle_band_arr, upper_band_arr, lower_band_arr

    cdef double mean_val, std_val
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS

    for i in prange(period - 1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        mean_val = c_mean(values, i - period + 1, i + 1)
        std_val = c_std(values, i - period + 1, i + 1, mean_val)
        
        middle_band[i] = mean_val
        upper_band[i] = mean_val + num_std * std_val
        lower_band[i] = mean_val - num_std * std_val
        
    return middle_band_arr, upper_band_arr, lower_band_arr


def compute_atr_omp(
    const double[:] high, 
    const double[:] low, 
    const double[:] close, 
    int period, 
    int num_threads
):
    """
    Calculates Average True Range (ATR) using OpenMP.
    The True Range loop is parallelized; smoothing is computed sequentially.
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[double, ndim=1] atr_arr = np.empty(n, dtype=np.float64)
    cdef double[:] atr = atr_arr
    cdef int i
    
    for i in range(period):
        atr[i] = c_nan
        
    if n <= period:
        return atr_arr

    cdef cnp.ndarray[double, ndim=1] tr_arr = np.empty(n, dtype=np.float64)
    cdef double[:] tr = tr_arr
    tr[0] = high[0] - low[0]
    
    cdef double val1, val2, val3, max_val
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS

    # Parallel True Range Loop
    for i in prange(1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        val1 = high[i] - low[i]
        val2 = high[i] - close[i - 1]
        val3 = low[i] - close[i - 1]
        
        if val2 < 0.0:
            val2 = -val2
        if val3 < 0.0:
            val3 = -val3
            
        max_val = val1
        if val2 > max_val:
            max_val = val2
        if val3 > max_val:
            max_val = val3
            
        tr[i] = max_val

    # Seeding ATR
    cdef double tr_sum = 0.0
    for i in range(period):
        tr_sum += tr[i]
    cdef double atr_val = tr_sum / period
    atr[period - 1] = atr_val

    # Wilder's smoothing loop (sequential)
    for i in range(period, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        atr[i] = atr_val

    return atr_arr


def compute_obv_omp(const double[:] close, const double[:] volume, int num_threads):
    """Computes On-Balance Volume (OBV) cumulative indicator."""
    cdef int n = close.shape[0]
    cdef cnp.ndarray[double, ndim=1] obv_arr = np.empty(n, dtype=np.float64)
    cdef double[:] obv = obv_arr
    cdef int i
    
    if n == 0:
        return obv_arr
        
    obv[0] = volume[0]
    cdef double current_obv = volume[0]
    
    # Cumulative calculation (sequential)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            current_obv += volume[i]
        elif close[i] < close[i - 1]:
            current_obv -= volume[i]
        obv[i] = current_obv
        
    return obv_arr


def compute_momentum_omp(const double[:] values, int period, int num_threads):
    """Parallel momentum calculator using OpenMP loops."""
    cdef int n = values.shape[0]
    cdef cnp.ndarray[double, ndim=1] mom_arr = np.empty(n, dtype=np.float64)
    cdef double[:] mom = mom_arr
    cdef int i
    
    for i in range(period):
        mom[i] = c_nan
        
    if n < period:
        return mom_arr
        
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    for i in prange(period, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        mom[i] = values[i] - values[i - period]
        
    return mom_arr


def compute_daily_returns_omp(const double[:] close, int num_threads):
    """Parallel Daily returns percentage calculation using OpenMP."""
    cdef int n = close.shape[0]
    cdef cnp.ndarray[double, ndim=1] ret_arr = np.empty(n, dtype=np.float64)
    cdef double[:] ret = ret_arr
    cdef int i
    
    if n > 0:
        ret[0] = c_nan
        
    if n <= 1:
        return ret_arr
        
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    for i in prange(1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        if close[i - 1] == 0.0:
            ret[i] = 0.0
        else:
            ret[i] = (close[i] - close[i - 1]) / close[i - 1]
            
    return ret_arr


def compute_log_returns_omp(const double[:] close, int num_threads):
    """Parallel Log returns percentage calculation using OpenMP."""
    cdef int n = close.shape[0]
    cdef cnp.ndarray[double, ndim=1] ret_arr = np.empty(n, dtype=np.float64)
    cdef double[:] ret = ret_arr
    cdef int i
    
    if n > 0:
        ret[0] = c_nan
        
    if n <= 1:
        return ret_arr
        
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    for i in prange(1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        if close[i - 1] <= 0.0 or close[i] <= 0.0:
            ret[i] = 0.0
        else:
            ret[i] = log(close[i] / close[i - 1])
            
    return ret_arr


def compute_rolling_std_omp(const double[:] values, int period, int num_threads):
    """Parallel rolling standard deviation calculation utilizing OpenMP."""
    cdef int n = values.shape[0]
    cdef cnp.ndarray[double, ndim=1] std_arr = np.empty(n, dtype=np.float64)
    cdef double[:] std = std_arr
    cdef int i
    
    for i in range(period - 1):
        std[i] = c_nan
        
    if n < period:
        return std_arr

    cdef double mean_val
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS

    for i in prange(period - 1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        mean_val = c_mean(values, i - period + 1, i + 1)
        std[i] = c_std(values, i - period + 1, i + 1, mean_val)
        
    return std_arr


def compute_rolling_volatility_omp(
    const double[:] returns, 
    int period, 
    int trading_days, 
    int num_threads
):
    """
    Parallel Rolling Volatility calculation utilizing OpenMP.
    Loops are executed in parallel without Python GIL.
    """
    cdef int n = returns.shape[0]
    cdef cnp.ndarray[double, ndim=1] volatility_arr = np.empty(n, dtype=np.float64)
    cdef double[:] volatility = volatility_arr
    cdef int i
    
    for i in range(period - 1):
        volatility[i] = c_nan
        
    if n < period:
        return volatility_arr

    cdef double annualize_factor = sqrt(<double>trading_days)
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS
    
    cdef int contains_nan, k

    for i in prange(period - 1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        contains_nan = 0
        
        # OMP NaN detection
        for k in range(i - period + 1, i + 1):
            if returns[k] != returns[k]:
                contains_nan = 1
                break
                
        if contains_nan == 1:
            volatility[i] = c_nan
        else:
            volatility[i] = c_volatility(returns, i - period + 1, i + 1, annualize_factor)
            
    return volatility_arr
