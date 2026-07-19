"""
Time-Series Data Preparation Pipeline.
Handles chronological splitting, standardization scaling, sequence windowing,
reproducible seed initialization, and configurable prediction targets.
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any, Optional

DEFAULT_FEATURE_ORDER = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Middle", "BB_Upper", "BB_Lower",
    "ATR_14", "Daily_Returns", "Log_Returns", "Rolling_Volatility"
]

def set_all_seeds(seed: int = 42):
    """Initializes random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic execution (can decrease speed slightly but ensures parity)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TimeSeriesDataset(Dataset):
    """Standard PyTorch Dataset for sequential data windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sliding_windows(
    features: np.ndarray, 
    targets: np.ndarray, 
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Groups tabular arrays into sequence windows.
    Input shape: (N, D) -> (N - seq_length, seq_length, D)
    Target shape: (N,) -> (N - seq_length,)
    """
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length):
        X_seq.append(features[i : i + seq_length])
        y_seq.append(targets[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def prepare_stock_data(
    df: pd.DataFrame,
    target_col: str = "Log_Returns",
    features: Optional[List[str]] = None,
    seq_length: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Prepares a stock dataframe for training:
    1. Sets seeds for reproducibility.
    2. Drops NaNs and sets target shifted by -1 day (to predict next-day value).
    3. Splits features chronologically to avoid look-ahead leakage.
    4. Scales features (fitting only on the training subset).
    5. Formulates sequence windows for deep learning models if seq_length > 1.
    """
    # 1. Reproducibility
    set_all_seeds(seed)
    
    # Clone df to avoid mutating original data
    df_work = df.copy()
    
    # Set default features if not specified
    if features is None:
        # Filter features that are actually in the dataframe
        features = [col for col in DEFAULT_FEATURE_ORDER if col in df_work.columns]
    else:
        # Filter provided features to exist in columns
        features = [col for col in features if col in df_work.columns]
        
    if not features:
        raise ValueError("No valid features found in DataFrame.")

    # 2. Configure prediction target shifted by -1 (next step target)
    if target_col not in df_work.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    df_work["Target"] = df_work[target_col].shift(-1)
    
    # Drop rows containing NaNs in either feature columns or target column (including the last row)
    df_clean = df_work.dropna(subset=features + ["Target"])
    
    if len(df_clean) <= seq_length:
        raise ValueError(f"Insufficient data rows ({len(df_clean)}) after dropping NaNs for sequence length {seq_length}.")

    # Extract arrays
    X_raw = df_clean[features].values
    y_raw = df_clean["Target"].values
    
    # 3. Temporal splitting (Chronological split index calculation)
    n = len(df_clean)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train_raw = X_raw[:train_end]
    y_train = y_raw[:train_end]
    
    X_val_raw = X_raw[train_end:val_end]
    y_val = y_raw[train_end:val_end]
    
    X_test_raw = X_raw[val_end:]
    y_test = y_raw[val_end:]
    
    # 4. Standardize scaling (fit strictly on training features to prevent validation leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 5. Build sequence windows if seq_length > 1 (sequential models)
    if seq_length > 1:
        X_train, y_train = create_sliding_windows(X_train_scaled, y_train, seq_length)
        X_val, y_val = create_sliding_windows(X_val_scaled, y_val, seq_length)
        X_test, y_test = create_sliding_windows(X_test_scaled, y_test, seq_length)
    else:
        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
        
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_order": features,
        "target_col": target_col,
        "seq_length": seq_length
    }
