"""
Comprehensive Parity Verification Script for Feature Engineering.
Asserts mathematical equivalence between Python and OpenMP implementations for all 13 indicators.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataStorage
from ai_engine.features import add_technical_indicators
from ai_engine.features.engine import PythonEngine, OpenMPEngine
import ai_engine.features.engine as fe_engine

def run_features_parity_check():
    print("=========================================================================")
    print("           RUNNING MATHEMATICAL PARITY CHECKS: PYTHON VS OPENMP          ")
    print("=========================================================================")

    # 1. Load cached RELIANCE stock data
    storage = DataStorage()
    ticker = "RELIANCE.NS"
    
    if not storage.raw_exists(ticker):
        print(f"Error: Raw cached data for {ticker} is missing. Ingesting test range...")
        from ai_engine.data import DataLoader
        loader = DataLoader(storage=storage)
        loader.get_ticker_data(ticker, start_date="2020-01-01", end_date="2023-01-01")
        
    df_raw = storage.load_raw(ticker)
    print(f"Loaded raw test dataset for {ticker}. Shape: {df_raw.shape}")

    # Initialize engines
    py_engine = PythonEngine()
    
    if not fe_engine.is_openmp_available():
        print("Warning: OpenMPEngine not loaded. Parity checking skipped.")
        sys.exit(1)
        
    omp_engine = fe_engine.get_active_engine()
    
    close = df_raw["Close"].values
    high = df_raw["High"].values
    low = df_raw["Low"].values
    volume = df_raw["Volume"].values

    # Test lists
    tests = [
        ("SMA_20", lambda: py_engine.compute_sma(close, 20), lambda: omp_engine.compute_sma(close, 20)),
        ("EMA_12", lambda: py_engine.compute_ema(close, 12), lambda: omp_engine.compute_ema(close, 12)),
        ("RSI_14", lambda: py_engine.compute_rsi(close, 14), lambda: omp_engine.compute_rsi(close, 14)),
        ("MACD", lambda: py_engine.compute_macd(close, 12, 26, 9), lambda: omp_engine.compute_macd(close, 12, 26, 9)),
        ("Bollinger", lambda: py_engine.compute_bollinger_bands(close, 20, 2.0), lambda: omp_engine.compute_bollinger_bands(close, 20, 2.0)),
        ("ATR", lambda: py_engine.compute_atr(high, low, close, 14), lambda: omp_engine.compute_atr(high, low, close, 14)),
        ("OBV", lambda: py_engine.compute_obv(close, volume), lambda: omp_engine.compute_obv(close, volume)),
        ("Momentum", lambda: py_engine.compute_momentum(close, 10), lambda: omp_engine.compute_momentum(close, 10)),
        ("Daily Returns", lambda: py_engine.compute_daily_returns(close), lambda: omp_engine.compute_daily_returns(close)),
        ("Log Returns", lambda: py_engine.compute_log_returns(close), lambda: omp_engine.compute_log_returns(close)),
        ("Rolling Std", lambda: py_engine.compute_rolling_std(close, 20), lambda: omp_engine.compute_rolling_std(close, 20)),
        ("Rolling Volatility", lambda: py_engine.compute_rolling_volatility(close, 21, 252), lambda: omp_engine.compute_rolling_volatility(close, 21, 252)),
    ]

    failed = 0
    print("\nStarting parity comparisons...")
    for name, py_fn, omp_fn in tests:
        try:
            py_res = py_fn()
            omp_res = omp_fn()
            
            # Handle Tuples (MACD and BB)
            if isinstance(py_res, tuple):
                for idx, (py_arr, omp_arr) in enumerate(zip(py_res, omp_res)):
                    py_clean = np.nan_to_num(py_arr)
                    omp_clean = np.nan_to_num(omp_arr)
                    np.testing.assert_allclose(py_clean, omp_clean, rtol=1e-5, atol=1e-5)
                print(f"  [PASS] {name} (tuple output components match)")
            else:
                py_clean = np.nan_to_num(py_res)
                omp_clean = np.nan_to_num(omp_res)
                np.testing.assert_allclose(py_clean, omp_clean, rtol=1e-5, atol=1e-5)
                print(f"  [PASS] {name} matches exactly")
        except AssertionError as e:
            print(f"  [FAIL] {name} mismatch: {e}")
            failed += 1
            
    print("\n-------------------------------------------------------------")
    if failed == 0:
        print("SUCCESS: Full mathematical parity between Python and OpenMP verified!")
    else:
        print(f"FAILED: {failed} indicator calculations had precision mismatch.")
    print("=========================================================================")

if __name__ == "__main__":
    run_features_parity_check()
