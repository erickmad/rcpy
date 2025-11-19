import numpy as np
from scipy.special import expit
from scipy.stats import gmean

# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
# Different metrics
def compute_errors(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "rmse") -> np.ndarray:
    """
    Compute per-timestep errors (T,) for the requested metric.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape: (T, D).
    y_pred : np.ndarray
        Predicted values. Shape: (T, D).
    metric : str, optional
        The metric to use. Options: 'mse', 'rmse', 'mae', 'nrmse'.

    Returns
    -------
    np.ndarray
        Per-timestep errors. Shape: (T,).
    """
    assert y_true.shape == y_pred.shape, f"Shapes must match (y_true shape: {y_true.shape}; y_pred shape: {y_pred.shape})"
    
    if metric == "mse":
        return np.mean((y_pred - y_true) ** 2, axis=1)
    if metric == "rmse":
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))
    if metric == "mae":
        return np.mean(np.abs(y_pred - y_true), axis=1)
    if metric == "nrmse":
        std_y = np.std(y_true, axis=1)
        std_y = np.where(std_y == 0, 1.0, std_y)
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1)) / std_y
    
    raise ValueError(f"Unknown metric '{metric}'.")