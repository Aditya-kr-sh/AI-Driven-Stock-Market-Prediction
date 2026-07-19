"""
PyTorch Deep Learning Training Engine.
Provides general epoch loops, optimization, validation tracking,
early stopping, learning rate scheduling, mixed precision (AMP), and GPU acceleration.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

# Unified AMP wrappers to resolve deprecation warnings on PyTorch 2.1+
if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    def get_autocast_context(device_type: str, enabled: bool):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    def get_grad_scaler(device_type: str, enabled: bool):
        return torch.amp.GradScaler(device_type, enabled=enabled)
else:
    from torch.cuda.amp import autocast as legacy_autocast, GradScaler as legacy_scaler
    def get_autocast_context(device_type: str, enabled: bool):
        # legacy autocast only supports cuda device type explicitly via keyword-less call
        return legacy_autocast(enabled=enabled)
    def get_grad_scaler(device_type: str, enabled: bool):
        return legacy_scaler(enabled=enabled)


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
    Supports mixed precision (AMP), scheduler updates, early stopping, and multi-GPU checks.
    """
    if device is None:
        device = get_computation_device()
        
    model.to(device)
    
    # Optional DistributedDataParallel (DDP) wrapping for Multi-GPU environments
    if device.type == "cuda" and torch.cuda.device_count() > 1 and torch.distributed.is_initialized():
        model = nn.parallel.DistributedDataParallel(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()
    
    # Initialize Gradient Scaler for Automatic Mixed Precision (AMP)
    amp_scaler = get_grad_scaler(device_type=device.type, enabled=(device.type == "cuda"))
    
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
            # Pinned memory transfers are accelerated with non_blocking=True
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Autocast runs forward operations under mixed FP16/FP32 precision
            with get_autocast_context(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(batch_x).squeeze(-1)
                loss = criterion(pred, batch_y)
                
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            
            epoch_train_loss += loss.item() * len(batch_x)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 2. Validation loop
        model.eval()
        epoch_val_loss = 0.0
        val_start = time.perf_counter()
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with get_autocast_context(device_type=device.type, enabled=(device.type == "cuda")):
                    pred = model(batch_x).squeeze(-1)
                    loss = criterion(pred, batch_y)
                epoch_val_loss += loss.item() * len(batch_x)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        val_duration = time.perf_counter() - val_start
        total_val_time += val_duration
        
        # Update learning rate scheduler based on validation score
        scheduler.step(epoch_val_loss)
        
        # 3. Validation Checkpoint & Early Stopping Check
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            # Handle unwrapping DDP model weights
            unwrap_model = model.module if hasattr(model, "module") else model
            best_model_weights = {k: v.cpu().clone() for k, v in unwrap_model.state_dict().items()}
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                break
                
    end_train_time = time.perf_counter()
    total_train_time = end_train_time - start_train_time
    
    # Restore the model weights with best val loss checkpoint
    if best_model_weights is not None:
        unwrap_model = model.module if hasattr(model, "module") else model
        unwrap_model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
        
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "device_used": str(device),
        "total_train_time_sec": total_train_time,
        "total_val_time_sec": total_val_time
    }
