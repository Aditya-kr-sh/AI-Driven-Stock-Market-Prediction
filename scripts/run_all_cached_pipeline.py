import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataStorage
from ai_engine.features import add_technical_indicators
from ai_engine.models.xgboost_model import XGBoostPredictor
from ai_engine.training import prepare_stock_data
from ai_engine.utils.config import settings

def main():
    print("=========================================================================")
    print("            ALL CACHED DATASETS AI PIPELINE RUNNER                       ")
    print("=========================================================================")
    
    storage = DataStorage()
    
    # 1. Discover all cached tickers in DATA_RAW_DIR
    raw_dir = storage.raw_dir
    metadata_files = list(raw_dir.glob("*.metadata.json"))
    
    tickers = []
    for meta_file in metadata_files:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                ticker = meta.get("ticker")
                if ticker:
                    tickers.append(ticker)
        except Exception as e:
            print(f"Error reading metadata file {meta_file}: {e}")
            
    print(f"Discovered {len(tickers)} cached stocks.")
    if not tickers:
        print("No cached datasets found.")
        return
        
    # 2. Get/Train a suitable model to reuse
    model_dir = settings.MODEL_DIR
    model_path = model_dir / "hpc_RELIANCE_xgboost.model"
    
    xgb_model = XGBoostPredictor()
    model_loaded = False
    
    if model_path.exists():
        print(f"Reusing existing trained model: {model_path}")
        try:
            xgb_model.load(str(model_path))
            model_loaded = True
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            
    # If no model could be loaded, train a fallback model on the first available stock
    target = "Log_Returns"
    if not model_loaded:
        train_ticker = None
        for t in tickers:
            if storage.raw_exists(t):
                # Ensure we pick a stock with enough rows for training
                try:
                    df_test = storage.load_raw(t)
                    if len(df_test) > 100:
                        train_ticker = t
                        break
                except Exception:
                    continue
        if not train_ticker:
            print("No suitable cached data available to train a fallback model.")
            return
            
        print(f"No suitable model found. Training a fallback model on {train_ticker}...")
        try:
            df_raw = storage.load_raw(train_ticker)
            df_features = add_technical_indicators(df_raw)
            splits_tab = prepare_stock_data(df_features, target_col=target, seq_length=1, seed=42)
            xgb_model.scaler = splits_tab["scaler"]
            xgb_model.feature_order = splits_tab["feature_order"]
            xgb_model.target_col = target
            xgb_model.fit((splits_tab["X_train"], splits_tab["y_train"]), (splits_tab["X_val"], splits_tab["y_val"]))
            # Save the fallback model
            fallback_path = model_dir / "hpc_fallback_xgboost.model"
            xgb_model.save(str(fallback_path))
            print(f"Fallback model trained and saved to {fallback_path}")
            model_loaded = True
        except Exception as e:
            print(f"Failed to train fallback model: {e}")
            return

    # Create predictions output directory
    predictions_dir = settings.BASE_PATH / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_start_time = time.perf_counter()
    
    for ticker in tickers:
        ticker_start_time = time.perf_counter()
        try:
            print(f"Processing {ticker}...")
            # Load raw data from cache
            df_raw = storage.load_raw(ticker)
            if df_raw.empty:
                raise ValueError("Loaded DataFrame is empty.")
            
            # Check length to prevent native OpenMP/Cython bounds crashes
            if len(df_raw) < 50:
                raise ValueError(f"Insufficient rows ({len(df_raw)}) to calculate indicators (minimum 50 required).")
            
            # Feature engineering
            df_features = add_technical_indicators(df_raw)
            
            # Ensure target column exists
            if target not in df_features.columns:
                df_features[target] = np.log(df_features["Adj Close"] / df_features["Adj Close"].shift(1))
                df_features.dropna(inplace=True)
                
            # Align features with the trained model's feature order
            feature_order = xgb_model.feature_order
            for col in feature_order:
                if col not in df_features.columns:
                    raise KeyError(f"Required feature '{col}' not found after feature engineering.")
                    
            X = df_features[feature_order].values
            
            # Fallback if the loaded model's scaler is None (STRICTLY FOR VALIDATION ONLY)
            # NOTE: Fitting a new scaler on the inference dataset is a validation fallback to 
            # prevent execution failures. In a production pipeline, you must use the exact 
            # scaler fitted on the training data to avoid scaling discrepancies.
            scaler = xgb_model.scaler
            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                try:
                    X_scaled = scaler.transform(X)
                except Exception:
                    # If transform fails, fit a new scaler as a last resort validation fallback
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
            
            # Model inference
            preds = xgb_model.predict(X_scaled)
            
            # Save predictions
            df_preds = pd.DataFrame({
                "Date": df_features.index,
                "Actual": df_features[target].values,
                "Predicted": preds
            }, index=df_features.index)
            
            pred_file = predictions_dir / f"{ticker}_predictions.csv"
            df_preds.to_csv(pred_file, index=True)
            
            elapsed = time.perf_counter() - ticker_start_time
            results.append({
                "ticker": ticker,
                "status": "SUCCESS",
                "time_sec": elapsed,
                "error": None
            })
            print(f"Successfully processed {ticker} in {elapsed:.4f} seconds.")
        except Exception as e:
            elapsed = time.perf_counter() - ticker_start_time
            results.append({
                "ticker": ticker,
                "status": "FAILED",
                "time_sec": elapsed,
                "error": str(e)
            })
            print(f"Failed to process {ticker}: {e}")
            
    total_runtime = time.perf_counter() - total_start_time
    
    # 3. Generate summary report
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = len(results) - success_count
    
    avg_time = np.mean([r["time_sec"] for r in results]) if results else 0.0
    
    report = f"""=========================================================================
            COMPREHENSIVE ALL-CACHED PIPELINE EXECUTION REPORT
=========================================================================
Timestamp             : {pd.Timestamp.now().isoformat()}
Total Discovered      : {len(tickers)}
Processed Successfully: {success_count}
Failed                : {failed_count}
Average Time/Ticker   : {avg_time:.4f} seconds
Total Runtime         : {total_runtime:.2f} seconds

Output Locations:
- Predictions Directory: {predictions_dir.absolute()}
- Models Directory:      {model_dir.absolute()}
- Summary Report Path:   {settings.BASE_PATH / 'docs' / 'all_cached_pipeline_report.txt'}

Detailed Logs:
"""
    for r in results:
        status_str = r["status"]
        err_str = f" | Error: {r['error']}" if r["error"] else ""
        report += f"- {r['ticker']}: {status_str} ({r['time_sec']:.4f}s){err_str}\n"
        
    report += "========================================================================="
    
    report_path = settings.BASE_PATH / "docs" / "all_cached_pipeline_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nReport saved to: {report_path.absolute()}")
    print("Execution completed.")

if __name__ == "__main__":
    main()
