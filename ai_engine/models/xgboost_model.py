"""
XGBoost Predictor Implementation.
Wraps the scikit-learn compatible XGBRegressor under the BasePredictor contract.
"""

import time
import os
import pickle
import json
from datetime import datetime
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional, List
from ai_engine.models.base import BasePredictor

class XGBoostPredictor(BasePredictor):
    """
    XGBoost tabular regressor wrapper matching the BasePredictor interface.
    Handles dynamic model parameterization, fitted scaler saving, and metadata tracking.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        random_state: int = 42,
        **kwargs
    ):
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            **kwargs
        }
        self.model = xgb.XGBRegressor(**self.hyperparameters)
        
        # Companion attributes populated during training
        self.scaler: Optional[Any] = None
        self.feature_order: Optional[List[str]] = None
        self.target_col: Optional[str] = None
        self.seq_length: int = 1  # Tabular has window size of 1
        self.training_stats: Dict[str, Any] = {}

    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], **kwargs) -> dict:
        """
        Trains the XGBRegressor on 2D tabular features.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        start_time = time.perf_counter()
        
        # Fit model
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_val, y_val)], 
            verbose=False
        )
        
        end_time = time.perf_counter()
        train_time = end_time - start_time
        
        # Retrieve validation loss curve from training history
        evals_result = self.model.evals_result()
        val_metric = list(evals_result["validation_0"].keys())[0] # usually 'rmse'
        val_losses = evals_result["validation_0"][val_metric]
        best_iteration = int(self.model.best_iteration) if hasattr(self.model, "best_iteration") else len(val_losses)
        
        self.training_stats = {
            "train_losses": [],  # XGBoost regressor doesn't output train losses by default unless specified in eval_set
            "val_losses": [float(v) for v in val_losses],
            "best_epoch": best_iteration,
            "best_val_loss": float(val_losses[best_iteration]) if best_iteration < len(val_losses) else float(val_losses[-1]),
            "device_used": "cpu", # Local XGBoost defaults to CPU
            "total_train_time_sec": train_time,
            "total_val_time_sec": 0.0 # Val time is inside fit call
        }
        
        return self.training_stats

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Generates predictions for tabular data features."""
        # Ensure 2D shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return self.model.predict(data)

    def save(self, filepath: str) -> None:
        """
        Saves the binary model checkpoint, fitted scaler, and companion metadata.
        """
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 1. Save binary checkpoint
        self.model.save_model(filepath)
        
        # 2. Save companion Scaler (Pickle file)
        scaler_path = filepath + ".scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
            
        # 3. Save companion metadata companion file
        meta_path = filepath + ".metadata.json"
        metadata = {
            "model_type": "XGBoost",
            "hyperparameters": self.hyperparameters,
            "features": self.feature_order,
            "target_column": self.target_col,
            "sequence_length": self.seq_length,
            "training_stats": self.training_stats,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def load(self, filepath: str) -> None:
        """
        Loads the binary model checkpoint, companion scaler, and feature configurations.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint path not found: {filepath}")
            
        # 1. Load model weights
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        
        # 2. Load scaler companion
        scaler_path = filepath + ".scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                
        # 3. Load metadata companion (optional helper attributes check)
        meta_path = filepath + ".metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.feature_order = metadata.get("features")
                self.target_col = metadata.get("target_column")
                self.seq_length = metadata.get("sequence_length", 1)
                self.hyperparameters = metadata.get("hyperparameters", self.hyperparameters)
