"""
Verification script for Vectorized Monte Carlo Portfolio Optimization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataStorage
from ai_engine.portfolio import MonteCarloSimulator

def verify_portfolio_optimization():
    print("=========================================================================")
    print("                 VERIFYING MONTE CARLO PORTFOLIO OPTIMIZATION            ")
    print("=========================================================================")

    # 1. Load price histories for 3 test stocks (RELIANCE, TCS, SBIN)
    storage = DataStorage()
    tickers = ["RELIANCE", "TCS", "SBIN"]
    
    price_series = {}
    for ticker in tickers:
        if not storage.raw_exists(ticker):
            # Fallback if no NS data exists
            if storage.raw_exists(ticker + ".NS"):
                df = storage.load_raw(ticker + ".NS")
            else:
                raise FileNotFoundError(f"Missing test data for ticker {ticker}")
        else:
            df = storage.load_raw(ticker)
        
        # Keep Close price
        price_series[ticker] = df["Close"]
        
    price_df = pd.DataFrame(price_series).dropna()
    print(f"Loaded portfolio price histories. Shape: {price_df.shape}")
    print(f"Tickers in portfolio: {list(price_df.columns)}")

    # 2. Run simulation
    simulator = MonteCarloSimulator(risk_free_rate=0.05)
    print("\nSimulating 50,000 portfolio allocations...")
    res = simulator.simulate(price_df, num_simulations=50000)

    # 3. Validations
    print("\nRunning validations...")
    assert len(res["tickers"]) == 3, "Assets mismatch"
    assert len(res["sim_returns"]) == 50000, "Simulation count mismatch"
    assert len(res["sim_volatilities"]) == 50000, "Volatility count mismatch"
    assert res["weights_matrix"].shape == (50000, 3), "Weights matrix dimension mismatch"

    # Assert weights sum to 1.0 within close tolerance
    np.testing.assert_allclose(res["weights_matrix"].sum(axis=1), 1.0, rtol=1e-5)
    print("- Weights normalization verified: Sum to 1.0.")

    max_s = res["max_sharpe"]
    min_v = res["min_vol"]
    
    print(f"\nMax Sharpe Portfolio allocation:")
    for ticker, weight in max_s["weights"].items():
        print(f"  {ticker}: {weight*100:.2f}%")
    print(f"  Expected Return     : {max_s['expected_return']*100:.2f}%")
    print(f"  Expected Volatility : {max_s['expected_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio        : {max_s['sharpe_ratio']:.4f}")

    print(f"\nMin Volatility Portfolio allocation:")
    for ticker, weight in min_v["weights"].items():
        print(f"  {ticker}: {weight*100:.2f}%")
    print(f"  Expected Return     : {min_v['expected_return']*100:.2f}%")
    print(f"  Expected Volatility : {min_v['expected_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio        : {min_v['sharpe_ratio']:.4f}")

    # Validations on ratios
    assert max_s["sharpe_ratio"] >= min_v["sharpe_ratio"], "Max Sharpe Sharpe ratio should be greater than Min Vol Sharpe ratio."
    assert min_v["expected_volatility"] <= max_s["expected_volatility"], "Min Vol volatility should be lower than Max Sharpe volatility."
    print("\n- Comparative portfolio bounds check: PASS.")
    
    print("=========================================================================")
    print("PORTFOLIO_SUCCESS")

if __name__ == "__main__":
    verify_portfolio_optimization()
