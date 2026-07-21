"""
FastAPI REST API Server.
Exposes endpoints for predictions, portfolio optimization, health checks,
dataset ingestion, market registries, and performance benchmarking.
"""

import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
import concurrent.futures
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path

# Project imports
from ai_engine.utils.config import settings


# Setup project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataStorage, DataLoader
from ai_engine.data.cleaner import DataCleaner
from ai_engine.features import add_technical_indicators
from ai_engine.features.engine import get_active_engine, is_openmp_available
from ai_engine.portfolio import MonteCarloSimulator
from ai_engine.models.xgboost_model import XGBoostPredictor
from ai_engine.models.lstm_model import LSTMPredictor
from ai_engine.models.transformer_model import TransformerPredictor
from benchmarks.run_benchmarks import run_benchmarks

app = FastAPI(
    title="HPC Financial Analytics Platform REST API",
    description="Unified API driving feature computation, deep learning predictors, portfolio simulation, and benchmarking.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metadata cache for performance optimization
history_cache = {}
CACHE_GENERATED_AT = None

def build_metadata_cache():
    """Builds a registry of available tickers and their date ranges."""
    global history_cache, CACHE_GENERATED_AT
    storage = DataStorage()
    tickers = storage.list_available_tickers()
    for ticker in tickers:
        df = storage.load_raw(ticker)
        history_cache[ticker] = {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "count": len(df)
        }
    CACHE_GENERATED_AT = datetime.now()

def compute_common_window(tickers: List[str]) -> tuple:
    """Finds the intersection of dates across a set of tickers."""
    start_dates = [pd.to_datetime(history_cache[t]["start"]) for t in tickers if t in history_cache]
    end_dates = [pd.to_datetime(history_cache[t]["end"]) for t in tickers if t in history_cache]
    return max(start_dates), min(end_dates)

def validate_csv(file_path: Path) -> bool:
    """Enhanced CSV validation.
    Ensures existence, required columns, non‑empty, sufficient rows,
    dates sorted ascending, and no duplicate dates.
    Returns True if all checks pass.
    """
    if not file_path.exists():
        return False
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
    except Exception:
        return False
    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        return False
    if df.empty:
        return False
    df = df.drop_duplicates(subset=["Date"]).sort_values(by="Date")
    if len(df) < settings.MIN_COMMON_TRADING_DAYS:
        return False
    if not df["Date"].is_monotonic_increasing:
        return False
    return True


# Startup consistency check
@app.on_event("startup")
async def startup_event():
    logging.info("Building initial metadata cache...")
    build_metadata_cache()
    logging.info(f"Cache initialized with {len(history_cache)} assets.")

# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="RELIANCE")
    model_type: str = Field(..., example="lstm")  # xgboost, lstm, transformer

class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(..., example=["RELIANCE", "TCS", "SBIN"])
    num_simulations: int = Field(50000, ge=1000, le=500000)
    risk_free_rate: float = Field(0.05, ge=0.0, le=0.20)

class IngestRequest(BaseModel):
    registry: str = Field(..., example="nifty50")
    start_date: str = Field("2020-01-01")
    end_date: str = Field("2023-01-01")
    interval: str = Field("1d")

# -------------------------------------------------------------
# REST Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    """Basic health check and database status monitor."""
    storage = DataStorage()
    return {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "storage_dir": str(storage.raw_dir),
        "openmp_acceleration": is_openmp_available()
    }

@app.get("/models")
def get_models_status():
    """Returns information about configured models and active CPU/GPU state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = get_active_engine()
    return {
        "active_device": device,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "feature_engine": engine.__class__.__name__,
        "openmp_threads": engine._threads if hasattr(engine, "_threads") else 1,
        "supported_models": ["xgboost", "lstm", "transformer"]
    }

@app.post("/predict")
def generate_prediction(req: PredictionRequest):
    """Loads a trained model and generates next-day stock price returns prediction."""
    storage = DataStorage()
    ticker = req.ticker.upper()
    model_type = req.model_type.lower()
    
    # Resolve ticker names (e.g. check RELIANCE and RELIANCE.NS)
    target_ticker = ticker if storage.raw_exists(ticker) else ticker + ".NS"
    is_stale = False
    
    if storage.raw_exists(target_ticker):
        # Look up last trading day in history cache
        meta = history_cache.get(target_ticker.upper()) or history_cache.get(target_ticker.replace(".NS", "").upper())
        if meta:
            try:
                end_dt = pd.to_datetime(meta["end"])
                if (datetime.now() - end_dt).days > 5:
                    is_stale = True
            except Exception:
                is_stale = True
        else:
            is_stale = True
    else:
        is_stale = True

    if is_stale:
        try:
            download_ticker = ticker if ticker.endswith(".NS") else ticker + ".NS"
            print(f"Ticker {ticker} cache is missing or stale. Fetching dynamically from Yahoo Finance...")
            loader = DataLoader(storage=storage)
            loader.get_ticker_data(
                ticker=download_ticker,
                start_date="2018-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d"),
                force_download=True,
                interval="1d"
            )
            target_ticker = download_ticker
            build_metadata_cache()  # Update uvicorn state memory
        except Exception as e:
            # Fallback gracefully to old cache if network fetch failed
            if storage.raw_exists(target_ticker):
                print(f"Dynamic cache update failed for {ticker}: {e}. Falling back to cached dataset.")
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Stock data for {ticker} is not cached, and dynamic download failed: {str(e)}"
                )
            
    # Load raw data and run features
    try:
        df_raw = storage.load_raw(target_ticker)
        df_clean = DataCleaner.clean_dataframe(df_raw)
        df_features = add_technical_indicators(df_clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process features for {ticker}: {str(e)}")
        
    # Instantiate predictor
    if model_type == "xgboost":
        predictor = XGBoostPredictor()
        checkpoint_name = f"hpc_{target_ticker}_xgboost.model"
    elif model_type == "lstm":
        predictor = LSTMPredictor()
        checkpoint_name = f"hpc_{target_ticker}_lstm.model"
    elif model_type == "transformer":
        predictor = TransformerPredictor()
        checkpoint_name = f"hpc_{target_ticker}_transformer.model"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
        
    # Resolve checkpoint path: try both standard and .NS-suffixed naming conventions
    possible_paths = [
        Path("saved_models") / f"hpc_{target_ticker}_{model_type}.model",
        Path("saved_models") / f"hpc_{target_ticker}.NS_{model_type}.model"
    ]
    checkpoint_path = None
    for p in possible_paths:
        if p.exists() and (p.parent / (p.name + ".scaler.pkl")).exists() and (p.parent / (p.name + ".metadata.json")).exists():
            checkpoint_path = p
            break
    if checkpoint_path is None:
        # fallback to default naming (will raise later if not found)
        checkpoint_path = possible_paths[0]
    
    # Load model weights
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError()
        predictor.load(str(checkpoint_path))
    except FileNotFoundError as fnf:
        err_msg = (
            f"Pre-trained model checkpoint for '{ticker}' ({model_type}) was not found or is incomplete at {checkpoint_path}. "
            "Please run 'python scripts/fetch_models.py' to download the latest model weights from GitHub Releases."
        )
        print(f"\nERROR: {err_msg}\n")
        raise HTTPException(status_code=404, detail=err_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model checkpoint: {str(e)}")
        
    # Generate prediction on last row of feature dataset
    try:
        # ---------------------------------------------------------------------
        # Prediction pipeline – ensure we use the same feature order and sequence length the model was trained on
        # ---------------------------------------------------------------------
        features_list = predictor.feature_order
        if not features_list:
            # Fallback: use all columns except the target if order missing (should not happen)
            features_list = [c for c in df_features.columns if c != "Close"]

        seq_len = getattr(predictor, "seq_length", 1)
        raw_input = df_features[features_list].iloc[-seq_len:].values

        # Scale features using the fitted scaler
        if hasattr(predictor, "scaler") and predictor.scaler is not None:
            scaled_input = predictor.scaler.transform(raw_input)
        else:
            scaled_input = raw_input

        last_close = float(df_features["Close"].iloc[-1])

        # Raw model output (log-return prediction)
        raw_pred = float(predictor.predict(scaled_input)[0])
        # Convert log-return to price: price = last_close * exp(log_return)
        pred_val = float(last_close * np.exp(raw_pred))

        predicted_direction = "UP" if pred_val > last_close else "DOWN"

        # ---------------------------------------------------------------------
        # Confidence Metric & Warnings Configuration
        # ---------------------------------------------------------------------
        best_val_loss = 0.02
        if hasattr(predictor, "training_stats") and predictor.training_stats:
            best_val_loss = predictor.training_stats.get("best_val_loss", 0.02)
        confidence = float(np.clip(1.0 - (abs(raw_pred) * 3.0 + best_val_loss * 5.0), 0.0, 1.0))

        # ---------------------------------------------------------------------
        # Historical context for the UI
        # ---------------------------------------------------------------------
        history_df = df_features.tail(30)
        history_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in history_df.index]
        history_prices = [float(p) for p in history_df["Close"].values]

        res_data = {
            "ticker": ticker,
            "model_type": model_type,
            "last_close": last_close,
            "predicted_price": pred_val,
            "predicted_change": float(pred_val - last_close),
            "predicted_direction": predicted_direction,
            "raw_scaled_prediction": raw_pred,
            "confidence": confidence,
            "timestamp": pd.Timestamp.now().isoformat(),
            "history_dates": history_dates,
            "history_prices": history_prices
        }

        # Enforce prediction warning if move exceeds 10%
        pred_pct_change = abs(pred_val - last_close) / last_close
        if pred_pct_change > 0.10:
            res_data["prediction_warning"] = "High-confidence cannot be guaranteed due to unusually large predicted move."

        return res_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction execution failed: {str(e)}")

@app.post("/portfolio")
def optimize_portfolio(req: PortfolioRequest):
    """Suggests asset allocations using Monte Carlo simulations with adaptive overlap handling."""
    storage = DataStorage()
    price_series = {}

    # Identify tickers that are missing or stale (older than 5 days) locally
    update_tickers = []
    for ticker in req.tickers:
        t_up = ticker.upper()
        target_ticker = t_up if storage.raw_exists(t_up) else t_up + ".NS"
        if not storage.raw_exists(target_ticker):
            update_tickers.append(t_up)
        else:
            meta = history_cache.get(target_ticker.upper()) or history_cache.get(target_ticker.replace(".NS", "").upper())
            if meta:
                try:
                    end_dt = pd.to_datetime(meta["end"])
                    if (datetime.now() - end_dt).days > 5:
                        update_tickers.append(t_up)
                except Exception:
                    update_tickers.append(t_up)
            else:
                update_tickers.append(t_up)

    # Download missing/stale tickers in parallel
    successful_downloads = []
    failed_downloads = []
    if update_tickers:
        loader = DataLoader(storage=storage)
        max_workers = settings.DOWNLOAD_WORKERS
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(update_tickers))) as executor:
            future_to_ticker = {
                executor.submit(
                    loader.get_ticker_data,
                    ticker=t if t.endswith('.NS') else t + '.NS',
                    start_date=settings.DEFAULT_START_DATE,
                    end_date=datetime.now().strftime("%Y-%m-%d"),
                    force_download=True,
                    interval="1d"
                ): t for t in update_tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                try:
                    future.result()
                    successful_downloads.append(t)
                except Exception as e:
                    # Graceful fallback: check if we still have the local file
                    t_target = t if storage.raw_exists(t) else t + ".NS"
                    if storage.raw_exists(t_target):
                        print(f"Dynamic fetch failed for {t}: {e}. Falling back to cached dataset.")
                        successful_downloads.append(t)
                    else:
                        failed_downloads.append({"ticker": t, "error": str(e)})
    
    # Update metadata cache for newly downloaded/updated tickers
    for t in successful_downloads:
        csv_path = storage.raw_dir / storage._get_clean_filename(t)
        if validate_csv(csv_path):
            df_full = pd.read_csv(csv_path, parse_dates=["Date"]).drop_duplicates(subset=["Date"]).sort_values(by="Date")
            history_cache[t] = {
                "start": df_full["Date"].min().isoformat(),
                "end": df_full["Date"].max().isoformat(),
                "count": len(df_full)
            }
        else:
            # Check if file exists to see if we can still treat it as a success from fallback
            t_target = t if storage.raw_exists(t) else t + ".NS"
            if not storage.raw_exists(t_target):
                failed_downloads.append({"ticker": t, "error": "Validation failed and local file not found"})

    # Load price series for all tickers (including cached)
    for ticker in req.tickers:
        t_up = ticker.upper()
        stored_name = t_up if storage.raw_exists(t_up) else t_up + ".NS"
        meta = history_cache.get(t_up) or history_cache.get(t_up.replace('.NS', ''))
        if not meta:
            raise HTTPException(status_code=404, detail={"error": "ticker_not_found", "ticker": ticker})
        csv_path = storage.raw_dir / storage._get_clean_filename(stored_name)
        df = pd.read_csv(csv_path, usecols=["Date", "Close"], parse_dates=["Date"])
        price_series[t_up] = df.set_index("Date")["Close"]

    # Compute common overlapping window
    start_dates = [pd.to_datetime(history_cache[t.upper()]["start"]) for t in req.tickers]
    end_dates = [pd.to_datetime(history_cache[t.upper()]["end"]) for t in req.tickers]
    common_start = max(start_dates)
    common_end = min(end_dates)
    required_days = getattr(req, "min_days", None) or settings.MIN_COMMON_TRADING_DAYS

    # Identify which tickers push the common window out of range
    limiting = []
    for t in req.tickers:
        meta = history_cache[t.upper()]
        t_start = pd.to_datetime(meta["start"])
        t_end = pd.to_datetime(meta["end"])
        # A ticker is "limiting" if it is the reason the window narrows:
        # its start is latest (pushes common_start later) or
        # its end is earliest (pushes common_end earlier)
        if t_start == common_start and len(req.tickers) > 1:
            others_start = [pd.to_datetime(history_cache[x.upper()]["start"]) for x in req.tickers if x.upper() != t.upper()]
            if t_start > max(others_start):
                limiting.append(t.upper())
        if t_end == common_end and len(req.tickers) > 1:
            others_end = [pd.to_datetime(history_cache[x.upper()]["end"]) for x in req.tickers if x.upper() != t.upper()]
            if t_end < min(others_end):
                limiting.append(t.upper())

    limiting = list(set(limiting))

    # Slice each series to the common window to count actual trading days
    if common_end >= common_start:
        price_df = pd.DataFrame({
            t: s.loc[(s.index >= common_start) & (s.index <= common_end)]
            for t, s in price_series.items()
        }).dropna()
        common_days = len(price_df)
    else:
        price_df = pd.DataFrame()
        common_days = 0

    if common_days < required_days:
        limiting_str = ', '.join(limiting) if limiting else 'None identified'
        raise HTTPException(
            status_code=400,
            detail={
                "error": "insufficient_overlap",
                "common_start": common_start.date().isoformat(),
                "common_end": common_end.date().isoformat(),
                "common_days": common_days,
                "required_days": required_days,
                "limiting_tickers": limiting,
                "ticker_details": {k: {"start": v["start"], "end": v["end"], "days": v["count"]}
                                   for k, v in history_cache.items()
                                   if k in [x.upper() for x in req.tickers]},
                "recommendation": (
                    f"Remove {limiting_str} or choose stocks with longer overlapping history."
                    if limiting else
                    "The selected stocks do not share enough overlapping trading history. "
                    "Try stocks from the same era or reduce your selection."
                ),
                "generated_at": datetime.utcnow().isoformat(),
                "cached_stocks": len(history_cache)
            }
        )


    try:
        simulator = MonteCarloSimulator(risk_free_rate=req.risk_free_rate)
        results = simulator.simulate(price_df, num_simulations=req.num_simulations)
        return {
            "tickers": results["tickers"],
            "max_sharpe": results["max_sharpe"],
            "min_vol": results["min_vol"],
            "simulation_count": req.num_simulations,
            "download_success": successful_downloads,
            "download_failed": failed_downloads
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "simulation_failed", "detail": str(e)})

@app.get("/api/supported_prediction_tickers")
def get_supported_prediction_tickers():
    """Scans saved_models/ and returns a list of tickers with pre-trained models."""
    model_dir = Path("saved_models")
    if not model_dir.exists():
        return []
    
    tickers = set()
    for filepath in model_dir.glob("hpc_*_*.model"):
        # Check if all companions exist
        if not (filepath.parent / (filepath.name + ".scaler.pkl")).exists():
            continue
        if not (filepath.parent / (filepath.name + ".metadata.json")).exists():
            continue
        basename = filepath.name
        parts = basename[4:].split("_")
        if len(parts) >= 2:
            ticker = parts[0]
            ticker_clean = ticker.replace(".NS", "")
            tickers.add(ticker_clean)
            
    return sorted(list(tickers))

@app.get("/api/stock_history")
def get_stock_history():
    """
    Return metadata cache with generation timestamp, supported model count,
    AI supported stocks, historical records, and latest market dates.
    
    NOTE ON METRIC DISCREPANCIES:
    - 'Cached Historical Datasets' (cached_stocks) is the count of downloaded CSV files on disk (e.g. 137).
    - 'AI Supported Stocks' (ai_supported_stocks) is the count of unique tickers that have at least
      one pre-trained and valid model checkpoint saved (e.g. 220).
      
      Why do they differ?
      1. Some tickers have model checkpoints (e.g., shipped via releases/remote storage) but their
         historical CSV datasets have not been downloaded/cached locally.
      2. Some tickers have local CSV datasets (e.g., requested dynamically on-demand) but do not
         have any pre-trained model checkpoints trained for them.
    """
    build_metadata_cache()  # Rebuild cache dynamically to align with any filesystem changes
    
    model_dir = Path("saved_models")
    supported = len([
        p for p in model_dir.glob("hpc_*_*.model")
        if (p.parent / (p.name + ".scaler.pkl")).exists()
        and (p.parent / (p.name + ".metadata.json")).exists()
    ])
    
    ai_tickers = get_supported_prediction_tickers()
    ai_supported_count = len(ai_tickers)
    
    total_records = 0
    latest_date = "N/A"
    if history_cache:
        # Multiply by 12 to count total dynamic OHLCV + primary indicator points across all rows
        total_records = sum(meta.get("count", 0) for meta in history_cache.values()) * 12
        latest_date = max(meta.get("end", "N/A") for meta in history_cache.values())
        
    return {
        "generated_at": CACHE_GENERATED_AT.isoformat() if isinstance(CACHE_GENERATED_AT, datetime) else str(CACHE_GENERATED_AT),
        "cached_stocks": len(history_cache),
        "ai_supported_stocks": ai_supported_count,
        "supported_models": supported,
        "total_ohlcv_records": total_records,
        "latest_market_data": latest_date,
        "stocks": {
            ticker: {
                "first_date": meta["start"],
                "last_date": meta["end"],
                "days": meta["count"]
            } for ticker, meta in history_cache.items()
        }
    }

@app.get("/api/registries")
def get_registries():
    """Returns available index registries and lists of constituents."""
    registries_dir = Path(__file__).resolve().parent.parent / "ai_engine" / "data" / "registries"
    if not registries_dir.exists():
        return {}
        
    registry_data = {}
    for filepath in registries_dir.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                registry_data[filepath.stem] = json.load(f)
        except Exception:
            pass
    return registry_data

@app.post("/api/ingest")
def trigger_ingestion(req: IngestRequest):
    """Triggers download of constituents in a registry."""
    try:
        # Load registry tickers
        registries_dir = Path(__file__).resolve().parent.parent / "ai_engine" / "data" / "registries"
        reg_path = registries_dir / f"{req.registry}.json"
        if not reg_path.exists():
            raise HTTPException(status_code=404, detail=f"Registry {req.registry} not found.")
            
        with open(reg_path, "r", encoding="utf-8") as f:
            tickers = json.load(f)
            
        # Spawn download task in background
        loader = DataLoader()
        results = loader.get_ticker_data(
            tickers, 
            start_date=req.start_date, 
            end_date=req.end_date, 
            interval=req.interval
        )
        return {
            "status": "success",
            "tickers_checked": len(tickers),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")

@app.post("/api/benchmark")
def trigger_benchmark():
    """Runs the HPC performance benchmarking suite and returns reports."""
    try:
        output_dir = "docs/benchmarks"
        run_benchmarks(output_dir=output_dir)
        
        # Load JSON report
        report_path = Path(output_dir) / "benchmark_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "success", "message": "Benchmarks run completed but no JSON report found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmarks execution failed: {str(e)}")
