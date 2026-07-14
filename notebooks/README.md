# Jupyter Notebooks

Jupyter notebooks executing data ingestion, indicator calculations, model training, and portfolio allocation scripts. The notebooks import functions from the shared core package `ai_engine` instead of duplicating execution parameters.

- `01_data_ingestion.ipynb`: Downloads stock datasets and populates the SQLite database.
- `02_feature_engineering.ipynb`: Technical analysis features calculation (OpenMP accelerated on the RU cluster).
- `03_model_training.ipynb`: PyTorch LSTM, PyTorch Transformer, and XGBoost training procedures.
- `04_portfolio_optimization.ipynb`: Allocation metrics and Monte Carlo simulation outputs.
