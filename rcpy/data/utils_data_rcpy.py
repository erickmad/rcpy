import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reservoirpy.datasets import henon_map, logistic_map, mackey_glass
from .data_retrieval import ClimateIndex, load_NOAA_data
#import tsdynamics as tsd

def generate_raw_data(config, system, seed=None):
    """
    Generate raw data according to the specified system.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing data parameters.
        Must have keys: config["data"]["length"], config["data"]["transient"]
    system : str
        Name of the system: "random", "henon", "logistic", "constant"
    seed : int, optional
        Seed for random number generator (only used if system == "random")

    Returns
    -------
    data : np.ndarray
        Array of shape (data_length, 1) with the generated data.
    """
    data_length = config["system"].get("data_length")
    transient = config["preprocessing"].get("data_transient", 0)
    
    if system == "constant":
        data = np.zeros((data_length, 1))
    
    elif system == "henon":
        full_data = henon_map(n_timesteps=data_length + transient)
        data = full_data[transient:, 0].reshape(-1, 1)
    
    elif system == "logistic":
        full_data = logistic_map(data_length + transient, r=4)
        data = full_data[transient:, 0].reshape(-1, 1)
    
    elif system == "mackeyglass":
        full_data = mackey_glass(n_timesteps=data_length + transient)
        data = full_data[transient:, 0].reshape(-1, 1)
    
    elif system == "enso":
        full_data = load_NOAA_data(ClimateIndex.NINO1870)['index']
        data = full_data[:-12]

    elif system == "random":
        rng = np.random.default_rng(seed)
        data = rng.uniform(-1, 1, (data_length, 1))
    
    else:
        raise ValueError(f"Unknown system name: {system}")
    
    return data


# ------------------------------------------------------------------
# Load data and preprocess it with optional min-max normalization to [-1, 1]
# ------------------------------------------------------------------

def load_data_rcpy(data_file):
    if data_file.endswith(".csv"):
        data = np.loadtxt(data_file, delimiter=',')
    elif data_file.endswith(".npy"):
        data = np.load(data_file)
    else:
        raise ValueError("Unsupported file type. Use .csv or .npy")

    # Ensure data is 2D: (T,) -> (T, 1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data

def make_scaler(train_data):
    train_min = np.min(train_data, axis=0)
    train_max = np.max(train_data, axis=0)
    denom = np.where(train_max - train_min == 0, 1.0, train_max - train_min)

    def scale(x):
        return 2 * (x - train_min) / denom - 1

    return scale, train_min, train_max

def preprocess_data_rcpy(
    data,
    init_discard=0,
    train_length=500,
    val_length=500,
    normalize=True,
):
    """
    Preprocess time series data for reservoir computing.

    Parameters
    ----------
    data : np.ndarray
        Input time series of shape (T,) or (T, D).
    init_discard : int, optional
        Number of initial samples to discard (default: 0).
    train_length : int, optional
        Length of the training set (default: 500).
    val_length : int, optional
        Length of the validation set (default: 500).
    normalize : bool, optional
        Whether to scale features to [-1, 1] using training data (default: True).

    Returns
    -------
    dict
        Dictionary with training, validation and test splits,
        as well as normalization parameters.
    """

    # Ensure 2D shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Discard initial samples
    data = data[init_discard:]
    T = data.shape[0]

    # Compute split indices
    train_end = train_length
    val_end = train_end + val_length

    # Extract splits
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Normalization (based on training data)
    if normalize:
        scale, train_min, train_max = make_scaler(train_data)
        train_data = scale(train_data)
        val_data = scale(val_data)
        test_data = scale(test_data)
    else:
        train_min, train_max = None, None

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_min": train_min,
        "train_max": train_max,
    }


def denormalize_data_rcpy(x_scaled, train_min, train_max):
    x_scaled = np.asarray(x_scaled)
    train_min = np.asarray(train_min)
    train_max = np.asarray(train_max)
    
    if x_scaled.shape[-1] != train_min.shape[-1]:
        raise ValueError("Shape mismatch: x_scaled and train_min/max must have matching number of features.")
    
    return 0.5 * (x_scaled + 1) * (train_max - train_min) + train_min



def ulam_map_series(x0: float, n_steps: int, discard: int = 100) -> np.ndarray:
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
