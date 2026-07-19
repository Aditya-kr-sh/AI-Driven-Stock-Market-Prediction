"""
Abstract Base Predictor Interface.
Defines the standard model execution, serialization, and training contract.
"""

from abc import ABC, abstractmethod
import numpy as np

class BasePredictor(ABC):
    """
    Abstract interface for all model predictors (XGBoost, LSTM, Transformer).
    Ensures consistent API for fitting, inference, saving, and loading.
    """

    @abstractmethod
    def fit(self, train_data, val_data, **kwargs) -> dict:
        """
        Trains the model.
        
        Args:
            train_data: Dict or Tuple containing train features and targets.
            val_data: Dict or Tuple containing validation features and targets.
            **kwargs: Extra model-specific parameters.
            
        Returns:
            A dictionary containing training history and stats (e.g. losses per epoch).
        """
        pass

    @abstractmethod
    def predict(self, data) -> np.ndarray:
        """
        Generates predictions for the given data.
        
        Args:
            data: NumPy array, DataFrame, or DataLoader.
            
        Returns:
            A flat NumPy array of forecasts.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Saves the model state/parameters to disk.
        
        Args:
            filepath: Path to the checkpoint file destination.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Loads the model state/parameters from disk.
        
        Args:
            filepath: Path to the checkpoint file source.
        """
        pass
