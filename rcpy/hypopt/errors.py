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


def standard_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "nrmse",
) -> float:
    """
    Multi-step geometric mean error (for stable or unstable systems).

    Computes the geometric mean over time steps (T), then averages over features (D).

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape (T, D).
    y_pred : np.ndarray
        Predicted values. Shape (T, D).
    metric : str, optional
        Error metric to compute. Options: 'mse', 'rmse', 'mae', 'nrmse'.

    Returns
    -------
    float
        Mean of the geometric mean error across features.
    """
    errors = compute_errors(y_true, y_pred, metric=metric)  # Shape (T,)
    geom_mean = gmean(errors)  # Scalar
    return float(geom_mean)


def soft_horizon_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "rmse",
    threshold: float = 0.2,
    softness: float = 0.02,               # ~ 10 % of threshold is a good default
) -> float:
    """
    Differentiable proxy for forecast-horizon.

    - Rewards models that keep the error below *threshold* for as
      many successive steps as possible.
    - Fully continuous and differentiable, usable in gradient-based optimisation.

    Parameters
    ----------
    y_true, y_pred : (B, T, D) arrays
    metric         : str, error metric passed to `_compute_errors`
    threshold      : float, “acceptable” error
    softness       : float, controls the width of the soft boundary
                     (smaller ⇒ harder threshold)

    Returns
    -------
    float
        Loss to *minimise* (more negative ⇒ longer valid horizon).

    Notes
    -----
    - This loss is equivalent to the expected horizon length, which is the sum of the survival probabilities.
    """
    # Convert the arrays into required shape (B, T, D)
    y_true = y_true[None, :, None]  # shape (1, T, 1)
    y_pred = y_pred[None, :, None]  # shape (1, T, 1)

    # 1) per-timestep error, then geometric mean over the batch
    errors = compute_errors(y_true, y_pred, metric)      # (B, T)
    e_t = gmean(errors, axis=0)                           # (T,)

    # 2) soft indicator of “good prediction” at each step
    good_t = expit((threshold - e_t) / softness)          # ∈ (0, 1)

    # 3) probability that all steps up to t are good (soft horizon survival)
    surv_t = np.exp(np.cumsum(np.log(good_t)))              # (T,)

    # 4) expected horizon length
    H = np.sum(surv_t)

    # 5) loss (minimise → maximise H)
    return -float(H)
