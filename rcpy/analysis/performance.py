import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import gmean
from scipy.stats import pearsonr
from .errors import compute_errors

def expected_forecast_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "rmse",
    threshold: float = 0.2,
    softness: float = 0.02,  # ~10% of threshold
) -> float:
    """
    Differentiable proxy for forecast horizon.

    Parameters
    ----------
    y_true, y_pred : (T,) or (T, D) arrays
    metric         : str, error metric passed to compute_errors
    threshold      : float, "acceptable" error
    softness       : float, controls width of soft boundary

    Returns
    -------
    float
        Expected horizon length.
    """
    # Ensure 2D (T, D) for compute_errors
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    # 1) per-timestep errors
    errors = compute_errors(y_true, y_pred, metric=metric, aggregate=False)  # (T,)
    if errors.ndim > 1:
        # geometric mean over last dimension if D>1
        e_t = gmean(errors, axis=1)
    else:
        e_t = errors

    # 2) soft indicator of "good prediction"
    good_t = expit((threshold - e_t) / softness)

    # 3) survival probability
    surv_t = np.exp(np.cumsum(np.log(good_t)))

    # 4) expected horizon length
    H = np.sum(surv_t)

    return float(H)

    
def compute_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "reference",  # "reference", "pearson", "acc"
    reference: np.ndarray | None = None,
    metric: str = "rmse",
    forecast_length: int | None = None,
    aggregate: bool = True,
    threshold: float = None,
    softness: float = None,
) -> float:
    """
    Compute forecast skill using different approaches.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (T, D).
    y_pred : np.ndarray
        Forecast values (same shape as y_true).
    method : str, default="reference"
        Skill method:
        - "error": return error metric (RMSE, MAE, MAPE).
        - "reference": relative to a reference forecast (climatology, persistence).
        - "efh": expected forecast horizon (requires `threshold` and `softness`).
        - "pearson": Pearson correlation coefficient.
        - "acc": anomaly correlation coefficient (ACC).
    reference : np.ndarray, optional
        Reference forecast (required for method="reference").
    metric : str, default="rmse"
        Error metric to use if method="reference".
    forecast_length : int, optional
        Restrict calculation to first `forecast_length` timesteps.
    aggregate : bool, default=True
        If True, compute one scalar score. If False, return per-timestep array.

    Returns
    -------
    float or np.ndarray
        Skill score.
    """
    # Restrict length if forecast_length is provided
    if forecast_length is not None:
        y_true = y_true[:forecast_length]
        y_pred = y_pred[:forecast_length]
        if reference is not None:
            reference = reference[:forecast_length]

    if method == "error":
        return compute_errors(y_true, y_pred, metric=metric, aggregate=aggregate)

    if method == "reference":
        if reference is None:
            raise ValueError("Reference forecast required for method='reference'.")
        err_forecast = compute_errors(y_true, y_pred, metric=metric, aggregate=aggregate)
        err_ref = compute_errors(y_true, reference, metric=metric, aggregate=aggregate)
        return 1.0 - err_forecast / err_ref
    
    elif method == "efh":
        if threshold is None or softness is None:
            raise ValueError("Both 'threshold' and 'softness' must be provided for method='efh'.")
        efh = expected_forecast_horizon(y_true, y_pred, metric=metric, threshold=threshold, softness=softness)
        return efh

    elif method == "pearson":
        # Flatten across dimensions
        r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        return r

    elif method == "acc":
        # Remove climatology (mean over time) before correlation
        climatology = np.mean(y_true, axis=0, keepdims=True)
        anomalies_true = y_true - climatology
        anomalies_pred = y_pred - climatology
        r, _ = pearsonr(anomalies_true.flatten(), anomalies_pred.flatten())
        return r

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'reference', 'pearson', 'acc'.")

