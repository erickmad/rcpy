import numpy as np
import pandas as pd

def ulam_map(x0: float, n_steps: int, discard: int = 100) -> np.ndarray:
    """
    Generate a time series from the Ulam map: x_{n+1} = 1 - 2x_n^2.
    
    Parameters
    ----------
    x0 : float
        Initial condition (should be in [-1, 1]).
    n_steps : int
        Number of time steps to return (after discarding transients).
    discard : int, optional
        Number of initial transient iterations to discard (default = 100).
    
    Returns
    -------
    series : np.ndarray
        Array of length n_steps containing the Ulam map time series.
    """
    # Ensure initial condition is within valid range
    x = np.clip(x0, -1.0, 1.0)
    
    # Burn-in phase to remove transients
    for _ in range(discard):
        x = 1 - 2 * x**2
    
    # Generate the actual series
    series = np.empty(n_steps)
    for i in range(n_steps):
        x = 1 - 2 * x**2
        series[i] = x
    
    return series
