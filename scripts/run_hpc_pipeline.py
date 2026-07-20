"""
HPC Core Pipeline Execution Script.
Executes the full-scale OpenMP scaling benchmarks on the complete dataset,
trains XGBoost, LSTM, and Transformer with full hyperparameter settings,
profiles hardware device scaling, and serializes academic evaluation reports.
"""

import time
import os
import gc
import json
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path to enable importing ai_engine
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ai_engine.data import DataStorage, NIFTY_50_TICKERS
from ai_engine.data.tickers import load_registry
from ai_engine.features import add_technical_indicators, get_active_engine, is_openmp_available
from ai_engine.features.engine import OpenMPEngine, PythonEngine
import ai_engine.features.engine as fe_engine
from ai_engine import XGBoostPredictor, LSTMPredictor, TransformerPredictor, prepare_stock_data, evaluate_predictions
from ai_engine.utils.config import settings

def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash for reproducibility logging."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "Unknown"

def get_cpu_model() -> str:
    """Queries CPU model info."""
    try:
        if platform_name := platform_system():
            if platform_name == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            elif platform_name == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
    except Exception:
        pass
    return "Unknown CPU"

def platform_system() -> str:
    try:
        import platform
        return platform.system()
    except Exception:
        return "Unknown"

# =========================================================================
# 1. LARGE-SCALE OPENMP SCALING BENCHMARK
# =========================================================================
def run_large_scale_openmp_benchmark(storage: DataStorage, index_name: str = "nifty50") -> dict:
    print("\n=============================================================")
    print(" 1. EXECUTING FULL-SCALE OPENMP SCALING BENCHMARK ")
    print("=============================================================")
    
    # Load all cached historical data to create a large dataset for testing
    loaded_dfs = []
    tickers = load_registry(index_name)
    for ticker in tickers:
        if storage.raw_exists(ticker):
            loaded_dfs.append(storage.load_raw(ticker))
            
    if not loaded_dfs:
        raise FileNotFoundError("No cached raw datasets found. Ensure data is downloaded first.")
        
    # Combine all historical datasets into one massive DataFrame
    df_combined = pd.concat(loaded_dfs, ignore_index=True)
    total_records = len(df_combined)
    print(f"Combined Large-Scale Dataset Size: {total_records:,} rows")
    
    active_engine = get_active_engine()
    py_engine = PythonEngine()
    
    # Target columns for warm-up & benchmark
    close = df_combined["Close"].values
    
    # A. Python / NumPy Baseline
    print("Profiling Python/NumPy Baseline...")
    gc.collect()
    start_time = time.perf_counter()
    # Execute technical indicators using PythonEngine fallback
    fe_engine._ACTIVE_ENGINE = py_engine
    _ = add_technical_indicators(df_combined, indicators_list=["BB", "ATR", "Volatility"])
    py_duration = time.perf_counter() - start_time
    print(f"  Python Baseline Time: {py_duration:.5f}s")
    
    # Restoring active engine
    fe_engine._ACTIVE_ENGINE = active_engine
    
    if not is_openmp_available():
        print("OpenMPEngine unavailable. Skipping OpenMP thread scaling benchmarks.")
        return {"records": total_records, "python_time": py_duration, "omp_results": []}
        
    # Test thread configurations: 1, 2, 4, 8, 12, 16 (limited to host CPU count)
    cpu_cores = os.cpu_count() or 1
    thread_settings = [1, 2, 4, 8, 16]
    thread_settings = [t for t in thread_settings if t <= cpu_cores]
    if cpu_cores not in thread_settings:
        thread_settings.append(cpu_cores)
    thread_settings = sorted(list(set(thread_settings)))
    
    omp_results = []
    print(f"Starting OpenMP thread scaling on {cpu_cores} available CPU cores...")
    
    for t in thread_settings:
        active_engine.set_threads(t)
        gc.collect()
        t_start = time.perf_counter()
        _ = add_technical_indicators(df_combined, indicators_list=["BB", "ATR", "Volatility"])
        t_duration = time.perf_counter() - t_start
        speedup = py_duration / t_duration
        print(f"  OpenMP ({t} thread{'s' if t > 1 else ''}): Time={t_duration:.5f}s, Speedup={speedup:.2f}x")
        omp_results.append({"threads": t, "duration_sec": t_duration, "speedup": speedup})
        
    # Serialize benchmark report
    git_hash = get_git_commit_hash()
    timestamp = pd.Timestamp.now().isoformat()
    cpu_model = get_cpu_model()
    
    report = f"""=========================================================================
            OFFICIAL HPC OPENMP FEATURE PIPELINE BENCHMARK REPORT
=========================================================================
Timestamp           : {timestamp}
Git Commit Hash     : {git_hash}
CPU Architecture    : {cpu_model} (Available logical cores: {cpu_cores})
Dataset Scale       : {total_records:,} rows (combined NIFTY-50 historical series)

Benchmark Table:
-------------------------------------------------------------
Implementation              Duration (sec)  Speedup
-------------------------------------------------------------
Python (NumPy Baseline)     {py_duration:.5f}s        1.00x (Baseline)
"""
    for res in omp_results:
        threads = res["threads"]
        label = f"OpenMP ({threads} thread{'s' if threads > 1 else ''})"
        report += f"{label:<28} {res['duration_sec']:.5f}s        {res['speedup']:.2f}x\n"
        
    report += "============================================================="
    
    report_path = settings.BASE_PATH / "docs" / "openmp_hpc_benchmark_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"HPC OpenMP benchmark report saved to: {report_path.absolute()}")
    
    return {
        "records": total_records,
        "python_time": py_duration,
        "omp_results": omp_results
    }


# =========================================================================
# 2. FULL-SCALE MODEL TRAINING & EVALUATIONS
# =========================================================================
def run_large_scale_model_training(storage: DataStorage) -> dict:
    print("\n=============================================================")
    print(" 2. EXECUTING HYBRID AI MODELS TRAINING & EVALUATIONS ")
    print("=============================================================")
    
    # Discover all cached tickers in raw storage dynamically
    raw_dir = storage.raw_dir
    metadata_files = list(raw_dir.glob("*.metadata.json"))
    representative_stocks = []
    for meta_file in metadata_files:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                ticker = meta.get("ticker")
                if ticker:
                    representative_stocks.append(ticker)
        except Exception:
            pass
            
    if not representative_stocks:
        # Fallback to the original hardcoded set if discovery fails
        representative_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    else:
        # Sort for deterministic processing order
        representative_stocks = sorted(list(set(representative_stocks)))
        
    print(f"Discovered {len(representative_stocks)} cached stocks for full-scale training.")
    
    # Configurable hyperparameters (suited for full-scale HPC runs)
    # Exposing these dictionary variables directly to configurations
    hparams_lstm = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 50,
        "random_state": 42
    }
    
    hparams_transformer = {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 50,
        "random_state": 42
    }
    
    hparams_xgb = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    }
    
    evaluation_records = []
    
    for ticker in representative_stocks:
        try:
            if not storage.raw_exists(ticker):
                print(f"Skipping {ticker}: Cache file missing.")
                continue
                
            print(f"\n--- Loading and preprocessing full history for {ticker} ---")
            df_raw = storage.load_raw(ticker)
            
            # Check length to prevent native OpenMP/Cython bounds crashes
            if len(df_raw) < 50:
                print(f"Skipping {ticker}: Insufficient rows ({len(df_raw)}) to calculate indicators (minimum 50 required).")
                continue
                
            df_features = add_technical_indicators(df_raw)
        
            # Prepare datasets temporally using Log_Returns target
            target = "Log_Returns"
            
            # A. Preprocessing splits for Tabular XGBoost
            splits_tab = prepare_stock_data(df_features, target_col=target, seq_length=1, seed=42)
            X_test_tab = splits_tab["X_test"]
            y_test_tab = splits_tab["y_test"]
            
            # B. Preprocessing splits for Sequential PyTorch Models (Sequence window = 20 days)
            seq_length = 20
            splits_seq = prepare_stock_data(df_features, target_col=target, seq_length=seq_length, seed=42)
            X_test_seq = splits_seq["X_test"]
            y_test_seq = splits_seq["y_test"]
            
            # Model 1: XGBoost tabular model
            print("Training XGBoost...")
            xgb_model = XGBoostPredictor(**hparams_xgb)
            xgb_model.scaler = splits_tab["scaler"]
            xgb_model.feature_order = splits_tab["feature_order"]
            xgb_model.target_col = target
            
            t_start = time.perf_counter()
            xgb_model.fit((splits_tab["X_train"], splits_tab["y_train"]), (splits_tab["X_val"], splits_tab["y_val"]))
            xgb_fit_time = time.perf_counter() - t_start
            
            t_start = time.perf_counter()
            xgb_preds = xgb_model.predict(X_test_tab)
            xgb_inf_time = time.perf_counter() - t_start
            xgb_metrics = evaluate_predictions(y_test_tab, xgb_preds)
            
            # Save model
            xgb_path = settings.BASE_PATH / "saved_models" / f"hpc_{ticker}_xgboost.model"
            xgb_model.save(str(xgb_path))
            
            evaluation_records.append({
                "ticker": ticker,
                "model_type": "XGBoost",
                "train_time_sec": xgb_fit_time,
                "inference_time_sec": xgb_inf_time,
                "device": "cpu",
                "metrics": xgb_metrics
            })
            
            # Model 2: PyTorch LSTM
            print("Training LSTM...")
            lstm_model = LSTMPredictor(**hparams_lstm)
            lstm_model.scaler = splits_seq["scaler"]
            lstm_model.feature_order = splits_seq["feature_order"]
            lstm_model.target_col = target
            
            t_start = time.perf_counter()
            lstm_model.fit((splits_seq["X_train"], splits_seq["y_train"]), (splits_seq["X_val"], splits_seq["y_val"]))
            lstm_fit_time = time.perf_counter() - t_start
            
            t_start = time.perf_counter()
            lstm_preds = lstm_model.predict(X_test_seq)
            lstm_inf_time = time.perf_counter() - t_start
            lstm_metrics = evaluate_predictions(y_test_seq, lstm_preds)
            
            # Save model
            lstm_path = settings.BASE_PATH / "saved_models" / f"hpc_{ticker}_lstm.model"
            lstm_model.save(str(lstm_path))
            
            evaluation_records.append({
                "ticker": ticker,
                "model_type": "LSTM",
                "train_time_sec": lstm_fit_time,
                "inference_time_sec": lstm_inf_time,
                "device": str(lstm_model.device),
                "metrics": lstm_metrics
            })
            
            # Model 3: PyTorch Transformer
            print("Training Transformer...")
            transformer_model = TransformerPredictor(**hparams_transformer)
            transformer_model.scaler = splits_seq["scaler"]
            transformer_model.feature_order = splits_seq["feature_order"]
            transformer_model.target_col = target
            
            t_start = time.perf_counter()
            transformer_model.fit((splits_seq["X_train"], splits_seq["y_train"]), (splits_seq["X_val"], splits_seq["y_val"]))
            transformer_fit_time = time.perf_counter() - t_start
            
            t_start = time.perf_counter()
            transformer_preds = transformer_model.predict(X_test_seq)
            transformer_inf_time = time.perf_counter() - t_start
            transformer_metrics = evaluate_predictions(y_test_seq, transformer_preds)
            
            # Save model
            trans_path = settings.BASE_PATH / "saved_models" / f"hpc_{ticker}_transformer.model"
            transformer_model.save(str(trans_path))
            
            evaluation_records.append({
                "ticker": ticker,
                "model_type": "Transformer",
                "train_time_sec": transformer_fit_time,
                "inference_time_sec": transformer_inf_time,
                "device": str(transformer_model.device),
                "metrics": transformer_metrics
            })
        except Exception as e:
            print(f"Error training models for {ticker}: {e}. Continuing to next ticker.")
            
    # Generate final evaluations report
    git_hash = get_git_commit_hash()
    timestamp = pd.Timestamp.now().isoformat()
    cpu_model = get_cpu_model()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    
    report = f"""=========================================================================
                 HPC MODELS TRAINING & EVALUATION SUMMARY REPORT
=========================================================================
Timestamp           : {timestamp}
Git Commit Hash     : {git_hash}
Hardware Config     : CPU = {cpu_model} | GPU = {gpu_name}

Evaluation Index Results:
"""
    for rec in evaluation_records:
        m = rec["metrics"]
        report += f"\n-------------------------------------------------------------\n"
        report += f"Stock: {rec['ticker']} | Model: {rec['model_type']} ({rec['device']})\n"
        report += f"-------------------------------------------------------------\n"
        report += f"  Train Duration     : {rec['train_time_sec']:.4f}s\n"
        report += f"  Inference Duration : {rec['inference_time_sec']:.4f}s\n"
        report += f"  MSE                : {m['mse']:.6f}\n"
        report += f"  RMSE               : {m['rmse']:.6f}\n"
        report += f"  MAE                : {m['mae']:.6f}\n"
        report += f"  R² Score           : {m['r2']:.4f}\n"
        report += f"  Directional Acc (%) : {m['directional_accuracy']:.2f}%\n"
        report += f"  Pearson Correlation: {m['correlation']:.4f}\n"

    report += "\n============================================================="
    
    report_path = settings.BASE_PATH / "docs" / "final_model_evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"HPC models evaluation report successfully saved to: {report_path.absolute()}")
    
    return {"records": evaluation_records}


# =========================================================================
# MAIN EXECUTION ROUTINE
# =========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="HPC AI Engine Training Pipeline")
    parser.add_argument(
        "--index",
        type=str,
        default="nifty50",
        help="Registry index name to train models on (e.g. nifty50, nifty100, nifty500) or path to custom JSON file."
    )
    args = parser.parse_args()
    
    print("=========================================================================")
    print("            RAMANUJAN UNIVERSE HPC CORE EXECUTION PIPELINE               ")
    print(f"            Training Index: {args.index.upper()}                          ")
    print("=========================================================================")
    
    # Isolate storage by index name to match custom dataset folders
    raw_dir_path = settings.BASE_PATH / "data" / args.index
    storage = DataStorage(raw_dir=raw_dir_path)
    
    # 1. Run full-scale OpenMP benchmark
    run_large_scale_openmp_benchmark(storage, index_name=args.index)
    
    # 2. Run model training and evaluations
    run_large_scale_model_training(storage)
    
    print("\n=========================================================================")
    print("                 HPC EXECUTION COMPLETED SUCCESSFULLY                    ")
    print("=========================================================================")

if __name__ == "__main__":
    main()
