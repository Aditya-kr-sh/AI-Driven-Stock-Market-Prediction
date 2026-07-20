"""
FastAPI REST API Server.
Exposes endpoints for predictions, portfolio optimization, health checks,
dataset ingestion, market registries, and performance benchmarking.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path

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
    target_ticker = ticker
    if not storage.raw_exists(ticker):
        if storage.raw_exists(ticker + ".NS"):
            target_ticker = ticker + ".NS"
        else:
            raise HTTPException(status_code=404, detail=f"Stock data for {ticker} not cached in local storage.")
            
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
        
    checkpoint_path = Path("saved_models") / checkpoint_name
    
    # Load model weights
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError()
        predictor.load(str(checkpoint_path))
    except FileNotFoundError as fnf:
        err_msg = (
            f"Pre-trained model checkpoint for '{ticker}' ({model_type}) was not found at {checkpoint_path}. "
            "Please run 'python scripts/fetch_models.py' to download the latest model weights from GitHub Releases."
        )
        print(f"\nERROR: {err_msg}\n")
        raise HTTPException(status_code=404, detail=err_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model checkpoint: {str(e)}")
        
    # Generate prediction on last row of feature dataset
    try:
        features_list = predictor.feature_order
        last_row = df_features[features_list].iloc[-1:].values
        
        # XGBoost handles standard 2D array, LSTM/Transformer expect sequence window (length 1 for default regressor)
        if model_type in ["lstm", "transformer"]:
            last_row = np.expand_dims(last_row, axis=0) # shape (1, 1, num_features)
            
        pred_val = float(predictor.predict(last_row)[0])
        last_close = float(df_features["Close"].iloc[-1])
        predicted_direction = "UP" if pred_val > last_close else "DOWN"
        
        # Get last 30 historical close prices and dates for charting
        history_df = df_features.tail(30)
        history_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in history_df.index]
        history_prices = [float(p) for p in history_df["Close"].values]
        
        return {
            "ticker": ticker,
            "model_type": model_type,
            "last_close": last_close,
            "predicted_price": pred_val,
            "predicted_change": float(pred_val - last_close),
            "predicted_direction": predicted_direction,
            "timestamp": pd.Timestamp.now().isoformat(),
            "history_dates": history_dates,
            "history_prices": history_prices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction execution failed: {str(e)}")

@app.post("/portfolio")
def optimize_portfolio(req: PortfolioRequest):
    """Suggests asset allocations using Monte Carlo simulations."""
    storage = DataStorage()
    price_series = {}
    
    for ticker in req.tickers:
        t_upper = ticker.upper()
        target = t_upper
        if not storage.raw_exists(t_upper):
            if storage.raw_exists(t_upper + ".NS"):
                target = t_upper + ".NS"
            else:
                raise HTTPException(status_code=404, detail=f"Price history for ticker {ticker} is not cached.")
        try:
            df = storage.load_raw(target)
            price_series[t_upper] = df["Close"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load price series for {ticker}: {str(e)}")
            
    price_df = pd.DataFrame(price_series).dropna()
    if len(price_df) < 10:
        raise HTTPException(status_code=400, detail="Insufficient historical price intersection rows to run optimization.")
        
    try:
        simulator = MonteCarloSimulator(risk_free_rate=req.risk_free_rate)
        results = simulator.simulate(price_df, num_simulations=req.num_simulations)
        
        return {
            "tickers": results["tickers"],
            "max_sharpe": results["max_sharpe"],
            "min_vol": results["min_vol"],
            "simulation_count": req.num_simulations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {str(e)}")

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
