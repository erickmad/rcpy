import numpy as np
import matplotlib.pyplot as plt

# ================================
# 0. Utilities to make forecasts
# ================================
def persistence_forecast(truth, n_leads):
    """
    Build a hindcast-style persistence forecast matrix from truth.
    
    Parameters
    ----------
    truth : array-like, shape (T,)
        Observed time series (1D).
    n_leads : int
        Number of lead times (columns) for the forecast matrix.
    
    Returns
    -------
    forecast : ndarray, shape (n_inits, n_leads)
        Persistence forecasts where forecast[i, l] = truth[i].
        n_inits = len(truth) - n_leads
    """
    truth = np.asarray(truth).flatten()
    if n_leads < 1:
        raise ValueError("n_leads must be >= 1")
    n_inits = len(truth) - n_leads
    if n_inits <= 0:
        raise ValueError("truth is too short for the requested n_leads")

    # persistence: forecast value at all leads equals value at initialization time truth[i]
    forecast = np.empty((n_inits, n_leads), dtype=float)
    init_values = truth[:n_inits]                 # values at initialization times (t = 0 .. n_inits-1)
    forecast[:] = init_values[:, np.newaxis]      # broadcast so each row repeats the init value across leads

    return forecast


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
    method=None,        # "zscore" or "iqr"
    z_thresh=2.5,           # |z| threshold for outlier rejection
    iqr_k=1.5,               # multiplier for IQR method
    forecast_length=None
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
    method : {"zscore", "iqr", "none", None}
        Method for filtering forecasts by dispersion.
        If "none" or None, only stability filtering is applied.
    z_thresh : float
        Threshold for z-score filtering (e.g. 2.5).
    iqr_k : float
        Multiplier for IQR range (e.g. 1.5).
    forecast_length : int, optional
        Number of timesteps to consider.

    Returns
    -------
    filtered_forecasts : list of np.ndarray
        Forecasts that passed both filters.
    indices : list of int
        Indices of forecasts that were kept.
    scores : list of float
        Dispersion scores (variance) of each kept forecast.
    """

    if forecast_length is not None:
        forecasts = [f[:forecast_length] for f in forecasts]

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
    if method is None or method.lower() == "none":
        keep_mask = np.ones_like(dispersions, dtype=bool)

    elif method.lower() == "zscore":
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
        raise ValueError("method must be 'zscore', 'iqr', or 'none'")

    # Step 4: Filter
    filtered_forecasts = [f for f, keep in zip(stable_forecasts, keep_mask) if keep]
    filtered_indices = [i for i, keep in zip(stable_indices, keep_mask) if keep]
    filtered_scores = [v for v, keep in zip(dispersions, keep_mask) if keep]

    return np.array(filtered_forecasts), filtered_indices, filtered_scores

# ================================
# 2. Plotting multiple forecasts
# ================================
def plot_multiforecasts(true_data, forecasts, forecast_length=None, dim=0):
    """
    Plot multiple forecasts with optional ensemble mean.

    Parameters
    ----------
    true_data : np.ndarray
        True data, shape (T,) or (T, D)
    forecasts : np.ndarray
        Forecasts array, shape (N, T, D)
    forecast_length : int, optional
        Number of timesteps to plot. Defaults to length of true_data.
    dim : int, optional
        Which dimension to plot if D > 1 (default: 0)
    """

    # Select the desired dimension
    forecasts = forecasts[:, :, dim]  # shape (N, T)
    if true_data.ndim > 1:
        true_data = true_data[:, dim]

    mean_forecast = np.mean(forecasts, axis=0)

    if forecast_length is None:
        forecast_length = len(true_data)

    # Plot
    fig = plt.figure(figsize=(7, 3))
    plt.plot(true_data[:forecast_length], label="Real Data", color="tab:blue", linewidth=2)

    # Plot each forecast in gray
    for forecast in forecasts:
        plt.plot(forecast[:forecast_length], color='tab:gray', lw=0.5, alpha=0.1, zorder=0)

    # Plot ensemble mean
    plt.plot(mean_forecast[:forecast_length], label='Mean Forecast', color='tab:orange')

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    #plt.tight_layout()
    return fig
