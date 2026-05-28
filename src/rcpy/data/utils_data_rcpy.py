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
    val_length=None,
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
        Length of validation set taken from the END of training data.
        If None or 0, no validation split is created.
    normalize : bool, optional
        Whether to scale features to [-1, 1] using training data (default: True).

    Returns
    -------
    dict
        Dictionary containing train/val/test splits and scaling params.
    """

    # Ensure 2D shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Discard initial samples
    data = data[init_discard:]
    T = data.shape[0]

    # --------------------------------------------------
    # Split train(+val) and test
    # --------------------------------------------------
    train_block = data[:train_length]
    test_data = data[train_length:]

    # --------------------------------------------------
    # Optional validation split
    # --------------------------------------------------
    if val_length is None or val_length == 0:
        train_data = train_block
        val_data = np.empty((0, train_block.shape[1]))
    else:
        if val_length >= train_length:
            raise ValueError("val_length must be smaller than train_length.")

        train_data = train_block[:-val_length]
        val_data = train_block[-val_length:]

    # --------------------------------------------------
    # Normalization (fit ONLY on train_data)
    # --------------------------------------------------
    if normalize:
        scale, train_min, train_max = make_scaler(train_data)

        train_data = scale(train_data)
        val_data = scale(val_data) if len(val_data) > 0 else val_data
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

def normalize_with_reference(x, scale_min, scale_max):
    """
    Normalize to [-1, 1] using reference min/max.
    """
    x = np.asarray(x)
    return 2 * (x - scale_min) / (scale_max - scale_min) - 1

def denormalize_data_rcpy(x_scaled, train_min, train_max):
    x_scaled = np.asarray(x_scaled)
    train_min = np.asarray(train_min)
    train_max = np.asarray(train_max)
    
    if x_scaled.shape[-1] != train_min.shape[-1]:
        raise ValueError("Shape mismatch: x_scaled and train_min/max must have matching number of features.")
    
    return 0.5 * (x_scaled + 1) * (train_max - train_min) + train_min

def add_noise(
    x,
    sigma,
    *,
    relative=False,
    rng=None,
):
    """
    Add Gaussian observational noise to data.

    Parameters
    ----------
    x : ndarray
        Clean training data, shape (T, d)
    sigma : float
        Noise standard deviation
    relative : bool, optional
        If True, sigma is interpreted relative to the data std
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    x_noisy : ndarray
        Noisy observations
    """
    if sigma == 0.0:
        return x.copy()

    if rng is None:
        rng = np.random.default_rng()

    scale = sigma
    if relative:
        scale = sigma * np.std(x, axis=0, keepdims=True)

    noise = rng.normal(loc=0.0, scale=scale, size=x.shape)
    return x + noise

