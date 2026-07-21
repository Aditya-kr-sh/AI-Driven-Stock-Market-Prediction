import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append("d:/stockproject")

from ai_engine import DataStorage, add_technical_indicators, prepare_stock_data
from ai_engine.models.xgboost_model import XGBoostPredictor

def main():
    storage = DataStorage()
    model_dir = Path("saved_models")
    
    # Find all xgboost models
    model_paths = list(model_dir.glob("hpc_*_xgboost.model"))
    print(f"Found {len(model_paths)} XGBoost models to retrain.")
    
    success_count = 0
    start_time = time.perf_counter()
    
    for path in model_paths:
        # Extract ticker name from filename, e.g. hpc_ADANIPORTS.NS_xgboost.model
        filename = path.name
        # Remove 'hpc_' prefix and '_xgboost.model' suffix
        ticker = filename[4:-14]
        
        from datetime import datetime
        from ai_engine.data import DataLoader
        
        # Check if raw CSV file exists, if not, download it dynamically
        clean_filename = storage._get_clean_filename(ticker)
        csv_filepath = storage.raw_dir / clean_filename
        if not csv_filepath.exists():
            try:
                print(f"Raw CSV not found for {ticker}. Fetching dynamically from Yahoo Finance...")
                download_ticker = ticker if ticker.endswith(".NS") else ticker + ".NS"
                loader = DataLoader(storage=storage)
                loader.get_ticker_data(
                    ticker=download_ticker,
                    start_date="2018-01-01",
                    end_date=datetime.now().strftime("%Y-%m-%d"),
                    force_download=True,
                    interval="1d"
                )
            except Exception as dl_err:
                print(f"Skipping {ticker}: Failed to dynamically download raw data: {str(dl_err)}")
                continue
            
        print(f"Retraining XGBoost for {ticker}...")
        try:
            # Load metadata to get hyperparameters
            meta_path = path.with_suffix(path.suffix + ".metadata.json")
            hparams = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    hparams = meta.get("hyperparameters", {})
            
            # Load data and build features
            df_raw = storage.load_raw(ticker)
            df_features = add_technical_indicators(df_raw)
            
            # Prepare splits
            splits = prepare_stock_data(df_features, target_col="Log_Returns", seq_length=1, seed=42)
            
            # Train model
            predictor = XGBoostPredictor(**hparams)
            predictor.scaler = splits["scaler"]
            predictor.feature_order = splits["feature_order"]
            predictor.target_col = "Log_Returns"
            
            predictor.fit(
                (splits["X_train"], splits["y_train"]),
                (splits["X_val"], splits["y_val"])
            )
            
            # Save model (overwrites model, metadata, and scaler)
            predictor.save(str(path))
            success_count += 1
            
        except Exception as e:
            print(f"Failed to retrain {ticker}: {str(e)}")
            
    elapsed = time.perf_counter() - start_time
    print(f"\nSuccessfully retrained {success_count}/{len(model_paths)} XGBoost models in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
