"""
Centralized list of NIFTY 50 ticker symbols for Yahoo Finance.
Provides validation and categorization utilities.
"""

from typing import List

# Complete list of NIFTY 50 constituent stocks formatted with the '.NS' suffix for National Stock Exchange (NSE).
NIFTY_50_TICKERS: List[str] = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    # LTIMindtree transitioned its stock ticker from 'LTIM' to 'LTM' (effective Feb 27, 2026)
    # to align with its unified brand transition.
    "LTM.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    # Tata Motors restructured its business, listing the passenger vehicle segment
    # under the trading ticker 'TMCV' on the NSE.
    "TMCV.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS"
]

def is_valid_nifty_50_ticker(ticker: str) -> bool:
    """
    Validates if a given ticker symbol belongs to the centralized NIFTY 50 tickers list.
    
    Args:
        ticker: The stock ticker symbol (e.g. 'RELIANCE.NS')
        
    Returns:
        True if the ticker is valid, False otherwise.
    """
    return ticker.strip().upper() in NIFTY_50_TICKERS
