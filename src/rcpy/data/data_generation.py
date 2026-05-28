import numpy as np
import pandas as pd
from reservoirpy.datasets import rossler, lorenz, mackey_glass, logistic_map, henon_map


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


def generate_time_series(system, data_length, discard_transient, **kwargs):
    """
    Generate a 1D time series from a specified dynamical system.

    Parameters
    ----------
    system : str
        One of {'rossler', 'lorenz', 'mackey_glass', 'logistic_map', 'henon_map'}.
    data_length : int
        Length of the returned time series (after discarding transients).
    discard_transient : int
        Number of initial samples to discard.
    dt : float, optional
        Time step for systems that require it (e.g., Mackey-Glass).
    tau : int, optional
        Delay parameter for systems that require it (e.g., Mackey-Glass).
    Returns
    -------
    np.ndarray
        1D NumPy array of length `data_length`.

    Raises
    ------
    ValueError
        If the system name is not recognized.
    """
    total_length = data_length + discard_transient

    if system == 'rossler':
        data_raw = rossler(total_length, **kwargs)[discard_transient:, 0]

    elif system == 'lorenz':
        data_raw = lorenz(total_length, **kwargs)[discard_transient:, 0]

    elif system == 'mackey_glass':
        data_raw = mackey_glass(total_length, **kwargs)[discard_transient:, 0]

    elif system == 'logistic_map':
        data_raw = logistic_map(total_length, **kwargs)[discard_transient:, 0]

    elif system == 'henon_map':
        data_raw = henon_map(total_length, **kwargs)[discard_transient:, 0]

    else:
        raise ValueError(f"System '{system}' not recognized.")

    return data_raw