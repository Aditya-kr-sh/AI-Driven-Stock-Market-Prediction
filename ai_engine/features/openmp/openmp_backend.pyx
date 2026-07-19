# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Cython OpenMP accelerated implementations of heavy rolling technical indicators.
Releases Python GIL and maps directly to double-precision raw memory views.
Uses const double[:] read-only memory views to interface directly with Pandas columns.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt

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
    
    # Pre-allocate NumPy arrays and get views
    cdef cnp.ndarray[double, ndim=1] middle_band_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] upper_band_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] lower_band_arr = np.empty(n, dtype=np.float64)
    
    cdef double[:] middle_band = middle_band_arr
    cdef double[:] upper_band = upper_band_arr
    cdef double[:] lower_band = lower_band_arr
    
    cdef int i
    
    # Set leading indices to NaN using C-constant
    for i in range(period - 1):
        middle_band[i] = c_nan
        upper_band[i] = c_nan
        lower_band[i] = c_nan
        
    if n < period:
        return middle_band_arr, upper_band_arr, lower_band_arr

    cdef double mean_val, std_val
    cdef int threads_to_use = num_threads if num_threads > 0 else GLOBAL_NUM_THREADS

    # Parallel sliding window calculation (thread variables are private automatically)
    for i in prange(period - 1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        mean_val = c_mean(values, i - period + 1, i + 1)
        std_val = c_std(values, i - period + 1, i + 1, mean_val)
        
        middle_band[i] = mean_val
        upper_band[i] = mean_val + num_std * std_val
        lower_band[i] = mean_val - num_std * std_val
        
    return middle_band_arr, upper_band_arr, lower_band_arr


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

    # Parallel sliding window standard deviation
    for i in prange(period - 1, n, nogil=True, num_threads=threads_to_use, schedule='static'):
        contains_nan = 0
        
        # OMP NaN detection using self-comparison (returns[k] != returns[k])
        for k in range(i - period + 1, i + 1):
            if returns[k] != returns[k]:
                contains_nan = 1
                break
                
        if contains_nan == 1:
            volatility[i] = c_nan
        else:
            volatility[i] = c_volatility(returns, i - period + 1, i + 1, annualize_factor)
            
    return volatility_arr


def compute_atr_omp(
    const double[:] high, 
    const double[:] low, 
    const double[:] close, 
    int period, 
    int num_threads
):
    """
    Calculates Average True Range (ATR) using OpenMP.
    The True Range computation loop is parallelized; the subsequent Wilder's 
    smoothing is computed sequentially due to recursive temporal dependency.
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

    # Step 1: Parallel True Range Calculation (with thread-private variables)
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

    # Step 2: Seeding ATR (sequential mean of the first 'period' elements)
    cdef double tr_sum = 0.0
    for i in range(period):
        tr_sum += tr[i]
    cdef double atr_val = tr_sum / period
    atr[period - 1] = atr_val

    # Step 3: Wilder's smoothing loop (sequential)
    for i in range(period, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        atr[i] = atr_val

    return atr_arr
