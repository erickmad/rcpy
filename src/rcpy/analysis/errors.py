import numpy as np

def compute_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "rmse",
    aggregate: bool = False,
    forecast_length: int | None = None,
) -> np.ndarray | float:
    """
    Compute forecast errors with flexible options.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (T, D) where T = timesteps, D = dimensions.
    y_pred : np.ndarray
        Predicted values (same shape as y_true).
    metric : str, default="rmse"
        Error metric to use: "mse", "rmse", "mae", "nrmse".
    aggregate : bool, default=False
        If False, return per-timestep errors (shape (T,)).
        If True, return a single scalar error over the forecast.
    forecast_length : int, optional
        If given, compute errors only over the first `forecast_length` timesteps.
    
    Returns
    -------
    np.ndarray or float
        Per-timestep error array if aggregate=False, otherwise a single scalar error.
    """
    assert y_true.shape == y_pred.shape, (
        f"Shapes must match (y_true: {y_true.shape}, y_pred: {y_pred.shape})"
    )
    
    diff = y_true - y_pred
    # Ensure at least 2D for axis indexing
    if diff.ndim == 1:
        diff = diff[:, None]  # shape (T, 1)
    
    if not aggregate:  # Per-timestep
        if metric == "mse":
            return np.mean(diff ** 2, axis=1)
        if metric == "rmse":
            return np.sqrt(np.mean(diff ** 2, axis=1))
        if metric == "mae":
            return np.mean(np.abs(diff), axis=1)
        if metric == "nrmse":
            std_y = np.std(y_true, axis=1)
            std_y = np.where(std_y == 0, 1.0, std_y)
            return np.sqrt(np.mean(diff ** 2, axis=1)) / std_y
    else:  # Single aggregated error
        if metric == "mse":
            return float(np.mean(diff ** 2))
        if metric == "rmse":
            return float(np.sqrt(np.mean(diff ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(diff)))
        if metric == "nrmse":
            std_y = np.std(y_true)
            if std_y == 0:
                std_y = 1.0
            return float(np.sqrt(np.mean(diff ** 2)) / std_y)
    
    raise ValueError(f"Unknown metric '{metric}'.")

def compute_cumulative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "rmse"
) -> np.ndarray:
    """
    Compute the cumulative mean forecast error (CME) over time.

    This computes, for each forecast step t:
        CME(t) = mean(error[0:t+1])
    where error(t) is computed using your `compute_errors` function.

    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (T, D)
    y_pred : np.ndarray
        Predicted values, same shape as y_true
    metric : str, default="rmse"
        Metric to use: "mse", "rmse", "mae", "nrmse"

    Returns
    -------
    np.ndarray
        Array of cumulative mean errors over time, shape (T,)
    """
    # Import your base error function
    from rcpy.analysis import compute_errors

    # Step 1: Compute per-timestep errors
    per_step_errors = compute_errors(
        y_true=y_true,
        y_pred=y_pred,
        metric=metric,
        aggregate=False
    )  # shape: (T,)

    # Step 2: Compute cumulative mean error
    cumulative_sum = np.cumsum(per_step_errors)
    timesteps = np.arange(1, len(per_step_errors) + 1)
    cumulative_mean_error = cumulative_sum / timesteps

    return cumulative_mean_error
