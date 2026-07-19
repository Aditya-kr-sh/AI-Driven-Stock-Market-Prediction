import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import our core library components
from ai_engine import (
    DataStorage,
    add_technical_indicators,
    prepare_stock_data,
    evaluate_predictions,
    XGBoostPredictor,
    LSTMPredictor,
    TransformerPredictor
)
from ai_engine.utils.config import settings

def clean_test_checkpoints_dir(test_dir: Path):
    """Deletes temporary validation checkpoint folders safely."""
    if test_dir.exists():
        try:
            shutil.rmtree(test_dir)
            print(f"Cleaned test checkpoint directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean directory {test_dir}: {e}")

def run_model_validation(model_class, model_name: str, data_splits: dict, test_dir: Path, ticker: str):
    """Validates fit, save, load, and prediction parity for a model on a given stock."""
    # 1. Instantiate wrapper with minimal configuration for smoke testing
    if model_name == "XGBoost":
        model = model_class(n_estimators=5, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_name == "LSTM":
        model = model_class(hidden_size=8, num_layers=1, epochs=3, batch_size=128, random_state=42)
    elif model_name == "Transformer":
        model = model_class(d_model=8, nhead=2, num_layers=1, epochs=3, batch_size=128, random_state=42)
        
    model.scaler = data_splits["scaler"]
    model.feature_order = data_splits["feature_order"]
    model.target_col = data_splits["target_col"]
    
    # 2. Fit model
    train_data = (data_splits["X_train"], data_splits["y_train"])
    val_data = (data_splits["X_val"], data_splits["y_val"])
    
    model.fit(train_data, val_data)
    
    # 3. Predict on test set before saving
    X_test = data_splits["X_test"]
    preds_before_save = model.predict(X_test)
    
    # 4. Save checkpoint (serializes weights, scaler, and metadata)
    checkpoint_filepath = test_dir / f"test_{ticker}_{model_name.lower()}.model"
    model.save(str(checkpoint_filepath))
    
    # Assert files exist on disk
    assert checkpoint_filepath.exists(), f"Model checkpoint missing: {checkpoint_filepath}"
    assert Path(str(checkpoint_filepath) + ".scaler.pkl").exists()
    assert Path(str(checkpoint_filepath) + ".metadata.json").exists()
    
    # 5. Load model from disk and assert predictions parity
    reloaded_model = model_class()
    reloaded_model.load(str(checkpoint_filepath))
    
    preds_after_load = reloaded_model.predict(X_test)
    
    # Check that predictions are mathematically identical (parity validation)
    np.testing.assert_allclose(
        preds_before_save, 
        preds_after_load, 
        rtol=1e-5, 
        atol=1e-6,
        err_msg=f"Prediction mismatch detected after reloading {model_name} for {ticker}!"
    )

def run_end_to_end_validation():
    print("=========================================================================")
    print("                 AI MODELS END-TO-END PIPELINE VALIDATION                ")
    print("=========================================================================")

    storage = DataStorage()
    tickers_to_test = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    
    # Verify cached data exists for all test stocks
    for ticker in tickers_to_test:
        if not storage.raw_exists(ticker):
            print(f"Error: Raw data for {ticker} must be cached. Run download_data.py first.")
            return

    # Temporary directory for testing checkpoints
    test_checkpoints_dir = settings.BASE_PATH / "saved_models" / "test_models"
    clean_test_checkpoints_dir(test_checkpoints_dir)
    
    try:
        # Validate each representative stock
        for ticker in tickers_to_test:
            print(f"\n--- Smoke Testing Full Pipeline for {ticker} ---")
            df_raw = storage.load_raw(ticker)
            df_features = add_technical_indicators(df_raw)
            
            # Smoke test using default Log_Returns target
            target = "Log_Returns"
            
            # A. Prepare splits for tabular models (XGBoost: seq_length=1)
            splits_tab = prepare_stock_data(df_features, target_col=target, seq_length=1, seed=42)
            run_model_validation(XGBoostPredictor, "XGBoost", splits_tab, test_checkpoints_dir, ticker)
            print(f"  [XGBoost] fit -> save -> load -> prediction parity verified successfully.")
            
            # B. Prepare splits for sequential models (LSTM & Transformer: seq_length=10)
            splits_seq = prepare_stock_data(df_features, target_col=target, seq_length=10, seed=42)
            run_model_validation(LSTMPredictor, "LSTM", splits_seq, test_checkpoints_dir, ticker)
            print(f"  [LSTM]    fit -> save -> load -> prediction parity verified successfully.")
            
            run_model_validation(TransformerPredictor, "Transformer", splits_seq, test_checkpoints_dir, ticker)
            print(f"  [Transformer] fit -> save -> load -> prediction parity verified successfully.")
            
        print("\n=========================================================================")
        print("          ALL SMOKE TESTS ON FIVE TEST STOCKS COMPLETED SUCCESSFULLY!    ")
        print("=========================================================================")
        
    finally:
        # Keep checkpoints folder clean after validation completes
        clean_test_checkpoints_dir(test_checkpoints_dir)

if __name__ == "__main__":
    run_end_to_end_validation()
