"""
AI-Driven Stock Market Prediction and Portfolio Optimization Core Engine.

This package contains all reusable core logic for:
- Data Ingestion (SQLite, Yahoo Finance)
- Feature Engineering (Indicators, including parallelized OpenMP/C engines)
- AI Modeling (XGBoost, LSTM, Transformer architectures)
- Deep Learning Model Training & Evaluation
- Portfolio Optimization (Sharpe, Monte Carlo simulations)
- Infrastructure Utilities (logging, configuration)
"""

from ai_engine.__version__ import __version__

__all__ = ["__version__"]
