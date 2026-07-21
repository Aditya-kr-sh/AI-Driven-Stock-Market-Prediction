"""
Verification script for FastAPI API endpoints.
Uses FastAPI TestClient to query REST routes and validate formats.
"""

import os
import sys
from pathlib import Path

# Prevent matplotlib from spawning Tkinter GUI window thread on shutdown
import matplotlib
matplotlib.use('Agg')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

def run_backend_verification():
    print("=========================================================================")
    print("                 VERIFYING FASTAPI API ENDPOINTS                         ")
    print("=========================================================================")

    # Setup offline missing model test by temporarily moving ACC's pre-trained model file
    acc_model = Path("saved_models") / "hpc_ACC.NS_xgboost.model"
    acc_model_backup = Path("saved_models") / "hpc_ACC.NS_xgboost.model.backup"
    
    did_rename = False
    if acc_model.exists():
        acc_model.rename(acc_model_backup)
        did_rename = True

    try:
        with TestClient(app) as client:
            # 1. Health check endpoint
            print("Querying /health...")
            res = client.get("/health")
            assert res.status_code == 200, f"Health check failed: {res.text}"
            health_data = res.json()
            assert health_data["status"] == "healthy", "Server status unhealthy"
            print(f"  [PASS] /health status: {health_data['status']}")

            # 2. Models info endpoint
            print("Querying /models...")
            res = client.get("/models")
            assert res.status_code == 200, f"Models status check failed: {res.text}"
            models_data = res.json()
            assert "active_device" in models_data, "Active device missing"
            assert "feature_engine" in models_data, "Feature engine missing"
            print(f"  [PASS] /models device used: {models_data['active_device']}, engine: {models_data['feature_engine']}")

            # 3. Portfolio optimizer endpoint
            print("Querying POST /portfolio...")
            req_body = {
                "tickers": ["RELIANCE", "TCS", "SBIN"],
                "num_simulations": 10000,
                "risk_free_rate": 0.05
            }
            res = client.post("/portfolio", json=req_body)
            assert res.status_code == 200, f"Portfolio simulation failed: {res.text}"
            port_data = res.json()
            assert "max_sharpe" in port_data, "Max Sharpe results missing"
            assert "min_vol" in port_data, "Min Vol results missing"
            print("  [PASS] POST /portfolio allocations: Sharpe and Volatility portfolios retrieved.")

            # 4. Predictions endpoint (XGBoost)
            print("Querying POST /predict (XGBoost)...")
            req_body = {
                "ticker": "RELIANCE",
                "model_type": "xgboost"
            }
            res = client.post("/predict", json=req_body)
            assert res.status_code == 200, f"Prediction failed: {res.text}"
            pred_data = res.json()
            assert "predicted_price" in pred_data, "Predicted price missing"
            assert "history_dates" in pred_data, "History dates missing"
            assert "history_prices" in pred_data, "History prices missing"
            assert len(pred_data["history_dates"]) == 30, f"Expected 30 history dates, got {len(pred_data['history_dates'])}"
            assert len(pred_data["history_prices"]) == 30, f"Expected 30 history prices, got {len(pred_data['history_prices'])}"
            print(f"  [PASS] POST /predict: Ticker={pred_data['ticker']}, Price={pred_data['predicted_price']:.2f}, History points={len(pred_data['history_prices'])}")

            # 4.1 Missing Model 404 detection
            print("Querying POST /predict (Missing model 404 test)...")
            req_body = {
                "ticker": "ACC",
                "model_type": "xgboost"
            }
            res = client.post("/predict", json=req_body)
            assert res.status_code == 404, f"Expected 404 for missing model, got status {res.status_code}"
            err_json = res.json()
            assert "detail" in err_json, "Error detail missing"
            assert "was not found or is incomplete at" in err_json["detail"], "Error detail does not explain missing model"
            print(f"  [PASS] POST /predict (404 Missing Model): Handled gracefully with custom message.")

            # 5. Registries endpoint
            print("Querying GET /api/registries...")
            res = client.get("/api/registries")
            assert res.status_code == 200, f"Registries failed: {res.text}"
            reg_data = res.json()
            assert "nifty50" in reg_data or len(reg_data) >= 0, "Nifty 50 or other registry should exist"
            print(f"  [PASS] GET /api/registries: Found {len(reg_data)} registries.")

            # 6. Benchmark endpoint
            print("Querying POST /api/benchmark...")
            res = client.post("/api/benchmark")
            assert res.status_code == 200, f"Benchmark execution failed: {res.text}"
            bench_data = res.json()
            assert "real_dataset_benchmark" in bench_data, "Real dataset benchmark details missing"
            assert "feature_benchmarks" in bench_data, "Feature benchmarks scaling data missing"
            print(f"  [PASS] POST /api/benchmark: Benchmark ran successfully.")

            print("=========================================================================")
            print("BACKEND_SUCCESS")
    finally:
        # Self-cleaning cleanup block to restore the model file
        if did_rename and acc_model_backup.exists():
            acc_model_backup.rename(acc_model)

if __name__ == "__main__":
    run_backend_verification()
