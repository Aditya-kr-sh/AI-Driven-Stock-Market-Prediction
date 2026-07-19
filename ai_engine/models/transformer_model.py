"""
PyTorch Transformer Predictor.
Implements the PositionalEncoding module, TransformerNet, and TransformerPredictor wrapper.
"""

import time
import os
import pickle
import json
import math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List
from ai_engine.models.base import BasePredictor
from ai_engine.training.data_prep import TimeSeriesDataset, set_all_seeds
from ai_engine.training.trainer import train_pytorch_regressor, get_computation_device

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for time-series sequential order injection."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerNet(nn.Module):
    """Core PyTorch Transformer network module for time-series regression."""
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        # Project inputs to transformer hidden dimension d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Self-attention encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Mapping final sequence step representation to output prediction
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_dim)
        x = self.input_proj(x) # shape: (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x) # shape: (batch_size, seq_length, d_model)
        
        # Pull final timestep
        last_step = out[:, -1, :] # shape: (batch_size, d_model)
        return self.fc(last_step)


class TransformerPredictor(BasePredictor):
    """
    Transformer Predictor wrapper matching the BasePredictor interface.
    Integrates network structure declarations and trainer workflows.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        random_state: int = 42
    ):
        # Validation: embedding size d_model must be divisible by attention heads nhead
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}) for Transformer self-attention.")
            
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        self.model: Optional[TransformerNet] = None
        self.device = get_computation_device()
        
        # Preprocessing companions
        self.scaler: Optional[Any] = None
        self.feature_order: Optional[List[str]] = None
        self.target_col: Optional[str] = None
        self.seq_length: int = 1
        self.training_stats: Dict[str, Any] = {}

    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], **kwargs) -> dict:
        """
        Trains the PyTorch Transformer network.
        
        Args:
            train_data: Tuple of (X_train, y_train) in sequential window shape.
            val_data: Tuple of (X_val, y_val) in sequential window shape.
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Enforce exact seed settings before instantiation
        set_all_seeds(self.random_state)
        
        # Determine input dimensionality dynamically from shape (samples, seq_length, features)
        if self.input_dim is None:
            self.input_dim = X_train.shape[2]
            
        self.seq_length = X_train.shape[1]
        
        # Instantiate network
        self.model = TransformerNet(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Create DataLoaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Run Trainer loop
        self.training_stats = train_pytorch_regressor(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            lr=self.learning_rate,
            device=self.device
        )
        
        return self.training_stats

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Generates predictions for sequential input windows."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
            
        self.model.eval()
        
        # Convert data to tensor and shape if required
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
            
        tensor_x = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor_x).squeeze(-1)
            
        return preds.cpu().numpy()

    def save(self, filepath: str) -> None:
        """Saves model weights, fitted scaler, and companion metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
            
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 1. Save binary PyTorch weights checkpoint
        torch.save(self.model.state_dict(), filepath)
        
        # 2. Save Scaler companion
        scaler_path = filepath + ".scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
            
        # 3. Save metadata companion
        meta_path = filepath + ".metadata.json"
        metadata = {
            "model_type": "Transformer",
            "hyperparameters": {
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "nhead": self.head if hasattr(self, "head") else self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "random_state": self.random_state
            },
            "features": self.feature_order,
            "target_column": self.target_col,
            "sequence_length": self.seq_length,
            "training_stats": self.training_stats,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def load(self, filepath: str) -> None:
        """Loads model weights, fitted scaler, and configuration metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint path not found: {filepath}")
            
        # 1. Load metadata companion first to reconstruct network dimensions
        meta_path = filepath + ".metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                hparams = metadata.get("hyperparameters", {})
                self.input_dim = hparams.get("input_dim")
                self.d_model = hparams.get("d_model", 64)
                self.nhead = hparams.get("nhead", 4)
                self.num_layers = hparams.get("num_layers", 2)
                self.dropout = hparams.get("dropout", 0.2)
                self.seq_length = metadata.get("sequence_length", 1)
                self.feature_order = metadata.get("features")
                self.target_col = metadata.get("target_column")
                
        if self.input_dim is None:
            raise ValueError("Could not reconstruct Transformer network. input_dim missing in metadata.")
            
        # 2. Re-instantiate network structure
        self.model = TransformerNet(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 3. Load model weights state dict
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        
        # 4. Load scaler companion
        scaler_path = filepath + ".scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
