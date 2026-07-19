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
from ai_engine.data import DataStorage, DataLoader, NIFTY_50_TICKERS
from ai_engine.features import add_technical_indicators, get_active_engine, is_openmp_available
from ai_engine.models import BasePredictor, XGBoostPredictor, LSTMPredictor, TransformerPredictor
from ai_engine.training import prepare_stock_data, set_all_seeds, get_computation_device
from ai_engine.evaluation import evaluate_predictions

__all__ = [
    "__version__",
    "DataStorage",
    "DataLoader",
    "NIFTY_50_TICKERS",
    "add_technical_indicators",
    "get_active_engine",
    "is_openmp_available",
    "BasePredictor",
    "XGBoostPredictor",
    "LSTMPredictor",
    "TransformerPredictor",
    "prepare_stock_data",
    "set_all_seeds",
    "get_computation_device",
    "evaluate_predictions"
]
