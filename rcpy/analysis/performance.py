import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import gmean
from scipy.stats import pearsonr
from .errors import compute_errors

# ==================================
# 1. Expected Forecast Horizon (EFH)
# ==================================
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


# ================
# 2. Compute Skill
# ================
def compute_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "error",             # "error", "reference", "pearson", "acc"
    reference: np.ndarray | None = None,
    metric: str = "rmse",              # used for "error" or "reference"
    forecast_length: int | None = None,
    mode: str = "aggregate",           # "per_step", "aggregate", "cumulative"
    threshold: float = None,           # for EFH if implemented
    softness: float = None             # for EFH if implemented
) -> np.ndarray | float:
    """
    Compute forecast skill or error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (T, D)
    y_pred : np.ndarray
        Predicted values (T, D)
    method : str
        "error", "reference", "pearson", "acc"
    reference : np.ndarray, optional
        Reference forecast for method="reference"
    metric : str
        Metric for error-based methods ("rmse", "mae", etc.)
    forecast_length : int, optional
        Restrict calculation to first forecast_length timesteps
    mode : str, default="aggregate"
        "per_step"   -> return per-timestep skill/error
        "aggregate"  -> single scalar over all timesteps
        "cumulative" -> cumulative skill/error over time
    threshold, softness : float, optional
        Required for EFH method

    Returns
    -------
    float or np.ndarray
        Skill/error array or scalar depending on mode
    """

    # Restrict length if needed
    if forecast_length is not None:
        y_true = y_true[:forecast_length]
        y_pred = y_pred[:forecast_length]
        if reference is not None:
            reference = reference[:forecast_length]

    T = y_true.shape[0]

    # --- Helper function: compute scalar skill/error ---
    def _compute_scalar(y_t, y_p, ref_t=None):
        if method == "error":
            return compute_errors(y_t, y_p, metric=metric, aggregate=True)
        elif method == "reference":
            if ref_t is None:
                raise ValueError("Reference forecast required for method='reference'.")
            err_forecast = compute_errors(y_t, y_p, metric=metric, aggregate=True)
            err_ref = compute_errors(y_t, ref_t, metric=metric, aggregate=True)
            return 1.0 - err_forecast / err_ref
        elif method == "pearson":
            r, _ = pearsonr(y_t.flatten(), y_p.flatten())
            return r
        elif method == "acc":
            climatology = np.mean(y_t, axis=0, keepdims=True)
            anomalies_true = y_t - climatology
            anomalies_pred = y_p - climatology
            r, _ = pearsonr(anomalies_true.flatten(), anomalies_pred.flatten())
            return r
        elif method == "efh":
            if threshold is None or softness is None:
                raise ValueError("Both 'threshold' and 'softness' must be provided for EFH method.")
            return expected_forecast_horizon(y_t, y_p, metric=metric, threshold=threshold, softness=softness)
        else:
            raise ValueError(f"Unknown method '{method}'")

    # --- Mode handling ---
    if mode == "aggregate":
        return _compute_scalar(y_true, y_pred, reference)

    elif mode == "per_step":
        # Compute skill/error for each timestep independently
        result = np.zeros(T, dtype=float)
        for t in range(T):
            result[t] = _compute_scalar(y_true[t:t+1], y_pred[t:t+1],
                                        reference[t:t+1] if reference is not None else None)
        return result

    elif mode == "cumulative":
        if method == "error":
            # Match old compute_cumulative_error: running mean of per-timestep errors
            e = compute_errors(y_true, y_pred, metric=metric, aggregate=False)
            return np.cumsum(e) / np.arange(1, T+1)
        else:
            # For other skills, use running computation over slices
            '''result = np.zeros(T, dtype=float)
            for t in range(T):
                result[t] = _compute_scalar(y_true[:t+1], y_pred[:t+1],
                                            reference[:t+1] if reference is not None else None)'''
            start_idx = 0 if method == "error" else 1
            result = np.full(T, np.nan)  # fill with NaN to keep consistent length

            for t in range(start_idx, T):
                result[t] = _compute_scalar(
                    y_true[:t+1],
                    y_pred[:t+1],
                    reference[:t+1] if reference is not None else None,
                )

            return result
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from 'aggregate', 'per_step', 'cumulative'.")


# ==========================================================
# 3. Skill Matrix
# ==========================================================
def skill_matrix(
    forecast,
    truth,
    cycle_length=12,
    max_lag=None,
    method="pearson",             # corresponds to compute_skill(method)
    metric="rmse",                # used only for error-based methods
    reference=None,               # optional reference forecast
    mode="aggregate",             # per_step, aggregate, cumulative
    threshold=None,               # for EFH if implemented
    softness=None,                # for EFH if implemented
):
    """
    Compute a cycle-based forecast skill matrix using `compute_skill()`.

    Parameters
    ----------
    forecast : array-like, shape (n_inits, n_leads)
        Forecasts where each row is a forecast initialization, and
        each column corresponds to a lead time (1 = 1 step ahead, etc.).
    truth : array-like, shape (n_total,)
        True observed time series aligned in time with forecast targets.
    cycle_length : int, optional (default=12)
        Number of time steps in one full cycle (e.g., 12 for months).
    max_lag : int, optional
        Maximum forecast lead time to include. If None, uses all leads.
    method : str, optional (default="pearson")
        Skill computation method for `compute_skill()`
        ("error", "reference", "pearson", "acc", "efh").
    metric : str, optional (default="rmse")
        Error metric for error-based methods ("rmse", "mae", etc.).
    reference : np.ndarray, optional
        Reference forecast for method="reference".
    mode : str, optional (default="aggregate")
        Passed to `compute_skill()`: "aggregate", "per_step", or "cumulative".
    threshold, softness : float, optional
        Parameters for EFH method.

    Returns
    -------
    skill : ndarray, shape (cycle_length, max_lag)
        Skill matrix where rows correspond to initialization month
        (0 = first month in the cycle), and columns correspond to lead times.
    """

    forecast = np.asarray(forecast)
    truth = np.asarray(truth).flatten()

    n_inits, n_leads = forecast.shape
    if max_lag is None:
        max_lag = n_leads
    else:
        max_lag = min(max_lag, n_leads)

    skill = np.full((cycle_length, max_lag), np.nan)

    for start_idx in range(cycle_length):
        # Initialization indices for this month
        init_indices = np.arange(start_idx, n_inits, cycle_length)

        for lag in range(max_lag):
            f_vals, t_vals = [], []

            for i in init_indices:
                if i + lag + 1 < len(truth):
                    f_vals.append(forecast[i, lag])
                    t_vals.append(truth[i + lag + 1])

            if len(f_vals) > 1:
                f_vals = np.array(f_vals)
                t_vals = np.array(t_vals)

                skill[start_idx, lag] = compute_skill(
                    y_true=t_vals[:, None],          # shape (T, D)
                    y_pred=f_vals[:, None],
                    method=method,
                    metric=metric,
                    reference=reference,
                    mode=mode,
                    threshold=threshold,
                    softness=softness,
                )

    return skill