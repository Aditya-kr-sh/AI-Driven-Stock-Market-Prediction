"""
Verification script for testing parallel cache data loading.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataLoader, DataStorage, NIFTY_50_TICKERS

def main():
    print("=========================================================================")
    print("                TESTING PARALLEL CACHE DATA LOADING STATUS               ")
    print("=========================================================================")
    
    storage = DataStorage()
    loader = DataLoader(storage=storage, index_name="nifty50")
    
    # 1. First confirm we have cache files
    cached_count = sum(1 for t in NIFTY_50_TICKERS if storage.raw_exists(t))
    print(f"Total cached tickers found: {cached_count} / {len(NIFTY_50_TICKERS)}")
    
    if cached_count == 0:
        print("Error: No cached files found. Run data ingestion first.")
        sys.exit(1)
        
    # 2. Run parallel load
    print(f"Starting parallel load of {len(NIFTY_50_TICKERS)} tickers...")
    start_time = time.perf_counter()
    datasets = loader.download_multiple(
        tickers=NIFTY_50_TICKERS,
        start_date="2026-01-01",
        end_date="2026-01-10",
        force_download=False
    )
    duration = time.perf_counter() - start_time
    
    print("\n-------------------------------------------------------------")
    print(f"Parallel Cache Load Completed in: {duration:.4f} seconds")
    print(f"Successfully loaded datasets count: {len(datasets)}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
