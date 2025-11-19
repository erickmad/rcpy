import numpy as np


# ==========================================================
# 1. Persistence forecasts
# ==========================================================
def make_persistence_forecast(truth, n_leads):
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
