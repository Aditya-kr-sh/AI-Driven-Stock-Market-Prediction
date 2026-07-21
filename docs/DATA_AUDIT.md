# System Dataset & Model Checkpoint Audit Report

This report presents a technical audit of the cached stock datasets, trained machine learning model checkpoints, index registries, and the fixes implemented to resolve the NIFTY 500 dataset discrepancy.

---

## 📊 Summary Audit Metrics

| Metric | Count | Purpose / Description |
| :--- | :---: | :--- |
| **Local Stock CSVs** | 51 | Historical daily close prices stored locally under `data/stocks/` |
| **Local CSV Metadata** | 51 | JSON companion files containing metadata schema definitions |
| **Trained Models (`.model`)** | 662 | Neural networks & boosting models (LSTM, Transformer, XGBoost) |
| **Model Metadata (`.json`)** | 662 | Checkpoint training statistics, hyperparameters, and feature orders |
| **Model Scalers (`.pkl`)** | 661 | Feature standardizers (StandardScaler) for scaling prediction inputs |
| **Supported Prediction Tickers** | 222 | Unique stocks with valid pre-trained checkpoints in `saved_models/` |
| **Supported Portfolio Tickers** | 500 | All NIFTY 500 constituent stocks (resolved via dynamic download) |

---

## 🔍 Root Cause Analysis

1. **HPC Training Run Scope**: The HPC cluster training pipeline successfully compiled, engineered features, and trained models on **222 unique tickers** from the NIFTY 500 index registry. This resulted in the 662 checkpoint files packaged inside the official release assets.
2. **Local Data Cache Scope**: During the initial local repository setup/bootstrapping, the data download script `scripts/download_data.py` defaulted to `--index nifty50`. This only downloaded and cached daily prices for the **50 Nifty 50 constituents**, saving them in `data/stocks/`.
3. **Hardcoded Backend Checks**: The FastAPI backend `/predict` and `/portfolio` endpoints checked for file existence exclusively under `data/stocks/TICKER.csv`. When a user requested a valid trained NIFTY 500 constituent (such as `TATACOMM`), the backend failed with a `404 Not Found` error because the historical CSV file was not cached locally.

---

## 🛠️ Fixes Implemented

### 1. Dynamic Dataset Downloader (Backend)
Updated `backend/main.py` to automatically detect when a requested stock's historical CSV is missing from the local cache. Instead of raising a `404` exception, the backend triggers the `DataLoader` utility to dynamically fetch the historical daily price series from Yahoo Finance, write the CSV and metadata companion files to `data/stocks/`, and proceed with inference or simulation.

### 2. Supported Prediction Tickers Filter (Frontend)
Created a new FastAPI endpoint `@app.get("/api/supported_prediction_tickers")` that scans the `saved_models/` directory for trained checkpoints. 
The frontend **Next-Day Prediction** ticker selector now dynamically loads from this endpoint, ensuring that the dropdown only displays stocks that are 100% supported by pre-trained checkpoints (the 222 tickers), completely eliminating the "model weights not found" error.

### 3. All Registry Autocomplete (Frontend)
Configured the **Monte Carlo Allocations** input box to suggest any of the 500 registry constituents. If the user adds a NIFTY 500 constituent (e.g. `TATACOMM`), the backend automatically downloads the price series and runs the simulation seamlessly.

---

## 📸 Screenshots & Verification

### Dynamic Autocomplete & Toast Warnings
Missing stock alerts are now shown using a sliding floating notification alert in the top-right corner. When entering multiple tickers, invalid inputs (such as trailing commas) are stripped dynamically.

### Verification Command Executed
To verify the dynamic download, the portfolio simulation API was queried for `TATACOMM`:
```bash
python scratch/test_portfolio_err.py
```

*Output:*
```json
Response Status: 200
Response JSON: {
  "tickers": ["RELIANCE", "TCS", "SBIN", "TATACOMM"],
  "max_sharpe": {
    "weights": {
      "RELIANCE": 0.173,
      "TCS": 0.047,
      "SBIN": 0.237,
      "TATACOMM": 0.541
    }
  }
}
```
*TATACOMM.csv was automatically saved under `data/stocks/TATACOMM.csv` and successfully simulated.*
