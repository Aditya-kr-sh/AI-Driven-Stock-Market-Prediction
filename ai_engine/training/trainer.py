"""
PyTorch Deep Learning Training Engine.
Provides general epoch loops, optimization, validation tracking,
early stopping, and hardware acceleration management.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

def get_computation_device() -> torch.device:
    """Detects and returns the available computational hardware device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_pytorch_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    early_stopping_patience: int = 5,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Core training and validation execution loop for sequential PyTorch models.
    Supports learning rate optimization, early stopping patience check, 
    and performance/training statistics tracking.
    """
    if device is None:
        device = get_computation_device()
        
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float("inf")
    best_model_weights = None
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    
    total_train_time = 0.0
    total_val_time = 0.0
    
    start_train_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        # 1. Training loop
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x).squeeze(-1) # output is (batch, 1) -> squeeze to (batch,)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * len(batch_x)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 2. Validation loop
        model.eval()
        epoch_val_loss = 0.0
        val_start = time.perf_counter()
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x).squeeze(-1)
                loss = criterion(pred, batch_y)
                epoch_val_loss += loss.item() * len(batch_x)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        val_duration = time.perf_counter() - val_start
        total_val_time += val_duration
        
        # 3. Validation Checkpoint & Early Stopping Check
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                # Stop epoch progression early
                break
                
    end_train_time = time.perf_counter()
    total_train_time = end_train_time - start_train_time
    
    # Restore the model weights with best val loss checkpoint
    if best_model_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
        
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "device_used": str(device),
        "total_train_time_sec": total_train_time,
        "total_val_time_sec": total_val_time
    }
