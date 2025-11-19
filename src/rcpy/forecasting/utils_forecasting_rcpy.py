import numpy as np


# ================================
# 1. Utilities to filter forecasts
# ================================
def is_stable_forecast(forecast, max_abs_value=1e3):
    """Check if forecast is numerically stable."""
    return (
        np.all(np.isfinite(forecast)) and
        np.max(np.abs(forecast)) < max_abs_value
    )

def filter_forecasts(
    forecasts,
    max_abs_value=1e3,
    method="zscore",        # "zscore" or "iqr"
    z_thresh=2.5,           # |z| threshold for outlier rejection
    iqr_k=1.5               # multiplier for IQR method
):
    """
    Filter forecasts by:
    1. Stability (finite values, not exploding)
    2. Dispersion (using z-score or IQR of forecast std or variance)

    Parameters
    ----------
    forecasts : list of np.ndarray
        List of forecast sequences (1D arrays).
    max_abs_value : float
        Threshold for detecting exploding values.
    method : {"zscore", "iqr"}
        Method for filtering forecasts by dispersion.
    z_thresh : float
        Threshold for z-score filtering (e.g. 2.5).
    iqr_k : float
        Multiplier for IQR range (e.g. 1.5).

    Returns
    -------
    filtered_forecasts : list of np.ndarray
        Forecasts that passed both filters.
    indices : list of int
        Indices of forecasts that were kept.
    scores : list of float
        Dispersion scores (variance) of each kept forecast.
    """

    # Step 1: Stability filter
    stable_forecasts, stable_indices = [], []
    for i, f in enumerate(forecasts):
        if is_stable_forecast(f, max_abs_value):
            stable_forecasts.append(f)
            stable_indices.append(i)

    if not stable_forecasts:
        return [], [], []

    # Step 2: Compute dispersion (variance)
    dispersions = np.array([np.var(f) for f in stable_forecasts])

    # Step 3: Apply filtering method
    if method.lower() == "zscore":
        mean = np.mean(dispersions)
        std = np.std(dispersions)
        z_scores = (dispersions - mean) / (std + 1e-12)
        keep_mask = np.abs(z_scores) <= z_thresh

    elif method.lower() == "iqr":
        q1, q3 = np.percentile(dispersions, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
        keep_mask = (dispersions >= lower) & (dispersions <= upper)

    else:
        raise ValueError("method must be either 'zscore' or 'iqr'")

    # Step 4: Filter
    filtered_forecasts = [f for f, keep in zip(stable_forecasts, keep_mask) if keep]
    filtered_indices = [i for i, keep in zip(stable_indices, keep_mask) if keep]
    filtered_scores = [v for v, keep in zip(dispersions, keep_mask) if keep]

    return filtered_forecasts, filtered_indices, filtered_scores

