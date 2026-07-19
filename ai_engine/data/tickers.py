"""
Centralized constituent ticker registry management.
Loads constituent listings dynamically from local JSON files, supporting
multiple market indices (NIFTY, S&P 500, NASDAQ, Dow Jones) and custom lists.
"""

import json
from pathlib import Path
from typing import List

REGISTRIES_DIR = Path(__file__).resolve().parent / "registries"

# Complete list of NIFTY 50 constituent stocks (retained for backward compatibility)
NIFTY_50_TICKERS: List[str] = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "LTM.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
    "TMCV.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS"
]

def load_registry(index_name: str) -> List[str]:
    """
    Loads constituent ticker list from locally stored registry JSON files.
    Allows loading custom registries if a path to a custom JSON file is provided.
    
    Args:
        index_name: The name of the built-in index (e.g. 'nifty100', 'sp500') 
                    or the path to a custom JSON constituent file.
    """
    index_clean = index_name.strip().lower().replace(" ", "").replace("-", "")
    
    # Check if index_name is an existing JSON filepath directly (custom registry)
    custom_path = Path(index_name)
    if custom_path.is_file():
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(t).strip().upper() for t in data]
        except Exception as e:
            raise ValueError(f"Failed to load custom registry file from {index_name}: {e}")
            
    # Try looking in our built-in registries folder
    reg_file = REGISTRIES_DIR / f"{index_clean}.json"
    if not reg_file.exists():
        # Handle some variations/aliases
        aliases = {
            "nifty50": "nifty50.json",
            "nifty100": "nifty100.json",
            "nifty200": "nifty200.json",
            "nifty500": "nifty500.json",
            "sp500": "sp500.json",
            "nasdaq100": "nasdaq100.json",
            "dowjones30": "dowjones30.json"
        }
        filename = aliases.get(index_clean)
        if filename:
            reg_file = REGISTRIES_DIR / filename
            
    if reg_file.exists():
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [str(t).strip().upper() for t in data]
        except Exception as e:
            raise ValueError(f"Failed to load built-in registry file {reg_file}: {e}")
            
    # Fallback to NIFTY 50 if registry name is unknown
    return NIFTY_50_TICKERS

def is_valid_nifty_50_ticker(ticker: str) -> bool:
    """
    Validates if a given ticker symbol belongs to the centralized NIFTY 50 tickers list.
    
    Args:
        ticker: The stock ticker symbol (e.g. 'RELIANCE.NS')
    """
    return ticker.strip().upper() in NIFTY_50_TICKERS
