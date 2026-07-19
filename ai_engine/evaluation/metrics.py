"""
Quantitative Performance Evaluation Metrics.
Provides statistical and financial regression metrics to evaluate stock predictions.
"""

import numpy as np

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error, ignoring zero values to avoid NaNs."""
    mask = y_true != 0.0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the R-squared Coefficient of Determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Directional Accuracy (%).
    Checks the percentage of instances where predictions share the same sign/direction as the target.
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    matches = (true_sign == pred_sign)
    return float(np.mean(matches) * 100.0)

def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Pearson Correlation Coefficient (R) between actuals and predictions."""
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    corr_matrix = np.corrcoef(y_true, y_pred)
    return float(corr_matrix[0, 1])

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates all regression and financial evaluation metrics for the forecast series.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    
    return {
        "mse": compute_mse(y_t, y_p),
        "rmse": compute_rmse(y_t, y_p),
        "mae": compute_mae(y_t, y_p),
        "mape": compute_mape(y_t, y_p),
        "r2": compute_r2(y_t, y_p),
        "directional_accuracy": compute_directional_accuracy(y_t, y_p),
        "correlation": compute_correlation(y_t, y_p)
    }
