"""
HPC Analytics Platform Performance Benchmarking Framework.
Evaluates OpenMP scaling, NumPy vectorization speedups, and hardware efficiency.
Outputs multi-format plots (PNG, SVG, PDF) and structured text logs.
"""

import os
import sys
import time
import json
import random
import platform
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Setup project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.features.engine import PythonEngine, get_active_engine, is_openmp_available
from ai_engine.portfolio import MonteCarloSimulator
from ai_engine.data import DataStorage
from ai_engine.data.tickers import load_registry
from ai_engine.utils.config import settings

def get_git_commit() -> str:
    """Safely retrieves the current Git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "Unknown"

def get_hardware_info() -> dict:
    """Retrieves processor and device acceleration metadata."""
    import torch
    info = {
        "os": platform.system() + " " + platform.release(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
    return info

def generate_mock_ohlcv(num_rows: int) -> pd.DataFrame:
    """Generates clean synthetic OHLCV records for high-volume benchmark testing."""
    rng = np.random.default_rng(seed=42)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=num_rows))
    high = close + rng.uniform(0.0, 2.0, size=num_rows)
    low = close - rng.uniform(0.0, 2.0, size=num_rows)
    open_val = close + rng.normal(0.0, 0.5, size=num_rows)
    volume = rng.uniform(1000, 100000, size=num_rows)
    
    return pd.DataFrame({
        "Open": open_val,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })

def run_benchmarks(index_name: str = "nifty50", output_dir: str = "docs/benchmarks"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    from datetime import timezone
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "git_commit": get_git_commit(),
        "random_seed": 42,
        "hardware": get_hardware_info()
    }
    
    print("=========================================================================")
    print("            HPC FINANCIAL PLATFORM: BENCHMARKING FRAMEWORK               ")
    print("=========================================================================")
    print(f"Timestamp   : {metadata['timestamp']}")
    print(f"Git Commit  : {metadata['git_commit']}")
    print(f"Hardware    : CPU={metadata['hardware']['processor']}, GPU={metadata['hardware']['gpu_name']}")
    print(f"Cores       : {metadata['hardware']['logical_cores']} threads detected")
    # Load and benchmark the real dataset for the selected index
    raw_dir_path = settings.BASE_PATH / "data" / index_name
    storage = DataStorage(raw_dir=raw_dir_path)
    
    print(f"\n--- Loading Real Dataset for {index_name.upper()} ---")
    tickers = load_registry(index_name)
    loaded_dfs = []
    for ticker in tickers:
        if storage.raw_exists(ticker):
            try:
                loaded_dfs.append(storage.load_raw(ticker))
            except Exception:
                pass
                
    real_bench = {}
    if loaded_dfs:
        df_real = pd.concat(loaded_dfs, ignore_index=True)
        total_stocks = len(loaded_dfs)
        total_rows = len(df_real)
        print(f"Loaded {total_stocks} stocks, total {total_rows:,} rows of price series.")
        
        close = df_real["Close"].values
        high = df_real["High"].values
        low = df_real["Low"].values
        
        # Benchmark Python
        t0 = time.perf_counter()
        py_engine = PythonEngine()
        _ = py_engine.compute_sma(close, 20)
        _ = py_engine.compute_ema(close, 12)
        _ = py_engine.compute_rsi(close, 14)
        _ = py_engine.compute_bollinger_bands(close, 20, 2.0)
        _ = py_engine.compute_atr(high, low, close, 14)
        real_py_time = time.perf_counter() - t0
        
        # Benchmark OpenMP
        omp_engine = get_active_engine()
        omp_active = is_openmp_available()
        max_threads = os.cpu_count() or 4
        
        if omp_active:
            omp_engine.set_threads(max_threads)
            t0 = time.perf_counter()
            _ = omp_engine.compute_sma(close, 20)
            _ = omp_engine.compute_ema(close, 12)
            _ = omp_engine.compute_rsi(close, 14)
            _ = omp_engine.compute_bollinger_bands(close, 20, 2.0)
            _ = omp_engine.compute_atr(high, low, close, 14)
            real_omp_time = time.perf_counter() - t0
            real_speedup = real_py_time / real_omp_time
        else:
            real_omp_time = None
            real_speedup = None
            
        real_bench = {
            "dataset_name": index_name.upper(),
            "stocks": total_stocks,
            "total_rows": total_rows,
            "python_time_sec": real_py_time,
            "openmp_time_sec": real_omp_time,
            "speedup": real_speedup
        }
        
        print(f"Python Time: {real_py_time:.5f}s | OpenMP ({max_threads} threads) Time: {real_omp_time:.5f}s | Speedup: {real_speedup:.2f}x" if real_omp_time is not None else f"Python Time: {real_py_time:.5f}s | OpenMP Offline")
    else:
        print(f"Warning: No cached data files found for registry {index_name.upper()} under {raw_dir_path.absolute()}")
        real_bench = {
            "dataset_name": index_name.upper(),
            "stocks": 0,
            "total_rows": 0,
            "python_time_sec": None,
            "openmp_time_sec": None,
            "speedup": None
        }

    # -------------------------------------------------------------
    # 1. Feature Engineering Benchmark
    # -------------------------------------------------------------
    print("\n--- Running Feature Engineering Scaling Benchmarks ---")
    sizes = [10000, 50000, 100000, 250000, 500000]
    threads_to_test = [1, 2, 4, 8]
    
    py_engine = PythonEngine()
    omp_engine = get_active_engine()
    omp_active = is_openmp_available()
    
    # Data storage for results
    results_df_sizes = []
    
    for size in sizes:
        print(f"Evaluating dataset size: {size:,} rows")
        df = generate_mock_ohlcv(size)
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values
        
        # A. Benchmark Python
        t0 = time.perf_counter()
        # Run a representative selection of indicators
        _ = py_engine.compute_sma(close, 20)
        _ = py_engine.compute_ema(close, 12)
        _ = py_engine.compute_rsi(close, 14)
        _ = py_engine.compute_bollinger_bands(close, 20, 2.0)
        _ = py_engine.compute_atr(high, low, close, 14)
        t_py = time.perf_counter() - t0
        
        row_res = {"size": size, "python_time": t_py}
        
        # B. Benchmark OpenMP threads
        if omp_active:
            for t in threads_to_test:
                omp_engine.set_threads(t)
                t0 = time.perf_counter()
                _ = omp_engine.compute_sma(close, 20)
                _ = omp_engine.compute_ema(close, 12)
                _ = omp_engine.compute_rsi(close, 14)
                _ = omp_engine.compute_bollinger_bands(close, 20, 2.0)
                _ = omp_engine.compute_atr(high, low, close, 14)
                t_omp = time.perf_counter() - t0
                row_res[f"omp_{t}_threads_time"] = t_omp
        else:
            for t in threads_to_test:
                row_res[f"omp_{t}_threads_time"] = np.nan
                
        results_df_sizes.append(row_res)
        
    feat_bench_df = pd.DataFrame(results_df_sizes)
    print(feat_bench_df.to_string(index=False))

    # -------------------------------------------------------------
    # 2. Monte Carlo Portfolio Simulation Benchmark
    # -------------------------------------------------------------
    print("\n--- Running Monte Carlo Portfolio Vectorization Benchmarks ---")
    sim_sizes = [5000, 10000, 50000, 100000, 250000]
    
    # Create sample price DataFrame for 5 assets
    num_days = 1000
    df_prices = pd.DataFrame({
        f"Asset_{i}": 100.0 + np.cumsum(np.random.normal(0.01, 1.0, size=num_days))
        for i in range(5)
    })
    
    mc_results = []
    simulator = MonteCarloSimulator(risk_free_rate=0.05)
    
    for count in sim_sizes:
        print(f"Simulating {count:,} portfolio iterations...")
        
        # A. Vectorized Run
        t0 = time.perf_counter()
        _ = simulator.simulate(df_prices, num_simulations=count)
        t_vec = time.perf_counter() - t0
        
        # B. Iterative Python Loop Run (Representative simulation of same formulas in Python)
        t0 = time.perf_counter()
        # We perform a small sample loop to extrapolate iterative runtime safely
        # to avoid freezing the system on huge counts
        sample_size = min(count, 1000)
        returns_df = df_prices.pct_change().dropna()
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        tickers = list(df_prices.columns)
        
        for _ in range(sample_size):
            raw_w = np.random.uniform(0.0, 1.0, size=5)
            w = raw_w / raw_w.sum()
            # Mean return
            _ = np.dot(w, mean_returns) * 252
            # Volatility
            _ = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
            
        t_loop_sample = time.perf_counter() - t0
        t_loop_extrapolated = t_loop_sample * (count / sample_size)
        
        mc_results.append({
            "simulations": count,
            "vectorized_time": t_vec,
            "iterative_time": t_loop_extrapolated,
            "speedup": t_loop_extrapolated / t_vec
        })
        
    mc_bench_df = pd.DataFrame(mc_results)
    print(mc_bench_df.to_string(index=False))

    # -------------------------------------------------------------
    # 3. Generate High-Quality Performance Plots (PNG, SVG, PDF)
    # -------------------------------------------------------------
    print("\nGenerating performance scaling curves...")
    
    # Plot 1: Feature Engineering Scaling
    plt.figure(figsize=(10, 6))
    plt.plot(feat_bench_df["size"], feat_bench_df["python_time"], 'o-', label="Python Baseline", linewidth=2)
    if omp_active:
        for t in threads_to_test:
            plt.plot(feat_bench_df["size"], feat_bench_df[f"omp_{t}_threads_time"], 's--', label=f"OpenMP ({t} Threads)", linewidth=1.5)
            
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Dataset Size (Number of Rows)")
    plt.ylabel("Execution Time (Seconds)")
    plt.title("Feature Engineering Performance: Python vs Parallel OpenMP")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    for fmt in ["png", "svg", "pdf"]:
        plt.savefig(out_path / f"feature_scaling.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Speedup vs Thread Counts
    if omp_active:
        plt.figure(figsize=(10, 6))
        # Compute speedup relative to 1 thread for the largest size
        base_time = feat_bench_df.iloc[-1]["omp_1_threads_time"]
        speedups = [base_time / feat_bench_df.iloc[-1][f"omp_{t}_threads_time"] for t in threads_to_test]
        
        plt.plot(threads_to_test, speedups, 'o-', color="forestgreen", label="Observed Speedup", linewidth=2.5)
        plt.plot(threads_to_test, threads_to_test, 'k--', label="Ideal Linear Speedup")
        plt.xlabel("Number of OpenMP Threads")
        plt.ylabel("Speedup Factor")
        plt.title("OpenMP Parallel Thread Efficiency Scaling")
        plt.grid(True, ls="--")
        plt.legend()
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(out_path / f"thread_speedup.{fmt}", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Monte Carlo Performance Speedup
    plt.figure(figsize=(10, 6))
    x = np.arange(len(sim_sizes))
    width = 0.35
    
    plt.bar(x - width/2, mc_bench_df["iterative_time"], width, label="Iterative Python Loops", color="tomato")
    plt.bar(x + width/2, mc_bench_df["vectorized_time"], width, label="Vectorized NumPy", color="dodgerblue")
    plt.yscale('log')
    plt.xticks(x, [f"{s:,}" for s in sim_sizes])
    plt.xlabel("Number of Simulated Portfolios")
    plt.ylabel("Execution Time (Seconds, Log Scale)")
    plt.title("Monte Carlo Portfolio Optimization Speedup")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    for fmt in ["png", "svg", "pdf"]:
        plt.savefig(out_path / f"monte_carlo_speedup.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()

    # Save benchmark metrics reports
    # Replace NaN values with None (JSON null) to prevent serialization errors
    feat_bench_clean = feat_bench_df.replace({np.nan: None})
    mc_bench_clean = mc_bench_df.replace({np.nan: None})

    report = {
        "metadata": metadata,
        "real_dataset_benchmark": real_bench,
        "feature_benchmarks": feat_bench_clean.to_dict(orient="records"),
        "portfolio_benchmarks": mc_bench_clean.to_dict(orient="records")
    }
    
    with open(out_path / "benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    # Write summary txt report
    with open(out_path / "benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write("=========================================================================\n")
        f.write("               HPC FINANCIAL PLATFORM: BENCHMARK REPORT                 \n")
        f.write("=========================================================================\n")
        f.write(f"Timestamp   : {metadata['timestamp']}\n")
        f.write(f"Git Commit  : {metadata['git_commit']}\n")
        f.write(f"CPU Details : {metadata['hardware']['processor']}\n")
        f.write(f"GPU Details : {metadata['hardware']['gpu_name']}\n")
        f.write("=========================================================================\n\n")
        
        f.write("1. Real Dataset Feature Engineering Benchmark:\n")
        f.write(f"  Dataset Name       : {real_bench.get('dataset_name')}\n")
        f.write(f"  Constituent Stocks : {real_bench.get('stocks')}\n")
        f.write(f"  Total Processed Rows: {real_bench.get('total_rows'):,}\n")
        py_time = real_bench.get('python_time_sec')
        omp_time = real_bench.get('openmp_time_sec')
        speedup = real_bench.get('speedup')
        f.write(f"  Python Baseline Time: {f'{py_time:.5f}s' if py_time is not None else 'N/A'}\n")
        f.write(f"  OpenMP Exec Time    : {f'{omp_time:.5f}s' if omp_time is not None else 'N/A'}\n")
        f.write(f"  OpenMP Speedup      : {f'{speedup:.2f}x' if speedup is not None else 'N/A'}\n\n")
        
        f.write("2. Synthetic Feature Engineering Scaling (seconds):\n")
        f.write(feat_bench_df.to_string(index=False) + "\n\n")
        f.write("3. Monte Carlo Portfolio Vectorization (seconds):\n")
        f.write(mc_bench_df.to_string(index=False) + "\n")
        
    print(f"\nALL BENCHMARKS RUN AND LOGGED SUCCESSFULLY! Saved files to {output_dir}")
    print("BENCHMARKS_SUCCESS")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HPC Analytics Platform Performance Benchmarking Framework")
    parser.add_argument(
        "--index",
        type=str,
        default="nifty50",
        help="Registry index name to run benchmarks on (e.g. nifty50, nifty100, nifty200, nifty500)."
    )
    args = parser.parse_args()
    
    run_benchmarks(index_name=args.index)
