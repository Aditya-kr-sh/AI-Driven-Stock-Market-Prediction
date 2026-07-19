"""
Vectorized Monte Carlo Portfolio Optimization Engine.
Calculates expected portfolio returns, volatilities, and Sharpe ratios
over thousands of random weight combinations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class MonteCarloSimulator:
    """Vectorized Monte Carlo simulation engine for portfolio optimization."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def simulate(
        self, 
        price_df: pd.DataFrame, 
        num_simulations: int = 100000,
        trading_days: int = 252
    ) -> Dict[str, Any]:
        """
        Runs a vectorized Monte Carlo simulation to compute optimal asset allocations.
        
        Args:
            price_df: DataFrame of historical closing prices (columns as asset tickers).
            num_simulations: Number of random portfolios to simulate.
            trading_days: Annualized trading days multiplier (default 252).
            
        Returns:
            Dict containing maximum Sharpe and minimum volatility portfolios,
            along with simulation results array.
        """
        tickers = list(price_df.columns)
        num_assets = len(tickers)
        
        if num_assets < 2:
            raise ValueError("Portfolio must contain at least 2 assets for optimization.")

        # 1. Compute daily returns and statistics
        returns_df = price_df.pct_change().dropna()
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values

        # 2. Vectorized weights generation: shape (num_simulations, num_assets)
        # Uniformly generate weights and normalize each row to sum to 1
        raw_weights = np.random.default_rng().uniform(0.0, 1.0, size=(num_simulations, num_assets))
        weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

        # 3. Vectorized Return computation
        # Expected annualized returns = weight * mean_returns * trading_days
        port_returns = np.dot(weights, mean_returns) * trading_days

        # 4. Vectorized Volatility computation
        # Expected annualized volatility = sqrt(weights.T * cov_matrix * weights) * sqrt(trading_days)
        # We compute this efficiently using Einstein summation convention to avoid huge matrix multiples
        port_vols = np.sqrt(np.einsum('ij,ji->i', np.dot(weights, cov_matrix), weights.T)) * np.sqrt(trading_days)

        # 5. Sharpe Ratio computation
        sharpe_ratios = (port_returns - self.risk_free_rate) / port_vols

        # 6. Extract optimal configurations
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(port_vols)

        max_sharpe_portfolio = {
            "weights": {tickers[i]: float(weights[max_sharpe_idx, i]) for i in range(num_assets)},
            "expected_return": float(port_returns[max_sharpe_idx]),
            "expected_volatility": float(port_vols[max_sharpe_idx]),
            "sharpe_ratio": float(sharpe_ratios[max_sharpe_idx])
        }

        min_vol_portfolio = {
            "weights": {tickers[i]: float(weights[min_vol_idx, i]) for i in range(num_assets)},
            "expected_return": float(port_returns[min_vol_idx]),
            "expected_volatility": float(port_vols[min_vol_idx]),
            "sharpe_ratio": float(sharpe_ratios[min_vol_idx])
        }

        return {
            "tickers": tickers,
            "sim_returns": port_returns,
            "sim_volatilities": port_vols,
            "sim_sharpe_ratios": sharpe_ratios,
            "weights_matrix": weights,
            "max_sharpe": max_sharpe_portfolio,
            "min_vol": min_vol_portfolio
        }
