import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    warmup_length=300,
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
    warmup_length : int, optional
        Length of the warmup set, which ends with val_data and starts
        warmup_length steps earlier (default: 300).
    normalize : bool, optional
        Whether to scale features to [-1, 1] using training data (default: True).

    Returns
    -------
    dict
        Dictionary with training, validation, test, and warmup splits,
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

    # Warmup ends at val_end and starts warmup_length before
    warmup_start = max(0, val_end - warmup_length)
    warmup_data = data[warmup_start:val_end]

    # Normalization (based on training data)
    if normalize:
        scale, train_min, train_max = make_scaler(train_data)
        train_data = scale(train_data)
        val_data = scale(val_data)
        test_data = scale(test_data)
        warmup_data = scale(warmup_data)
    else:
        train_min, train_max = None, None

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "warmup_data": warmup_data,
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


def plot_preprocessed_data_rcpy(
    data_raw,
    data,
    init_transient,
    transient_length,
    train_length,
    warmup_length,
    system_name="System",
    dates=None,
):
    """
    Plot preprocessed data sections for one or more variables.
    
    Parameters:
    - data_raw: ndarray of shape (T,) or (T, D)
    - data: dict with keys like 'transient_data', 'train_data', etc.
    - init_transient, transient_length, train_length, warmup_length: int
    - system_name: str, title of the plot
    - dates: optional array-like of length T (e.g., pd.date_range)
    """
    data_raw = np.atleast_2d(data_raw)
    if data_raw.shape[0] < data_raw.shape[1]:
        data_raw = data_raw.T  # ensure shape (T, D)
    
    T, D = data_raw.shape
    if dates is None:
        dates = np.arange(T)
    
    # Compute time indices for each section
    transient_start = init_transient
    transient_end = transient_start + transient_length
    train_start = transient_end
    train_end = train_start + train_length
    warmup_start = train_end - warmup_length
    warmup_end = warmup_start + warmup_length
    validation_start = warmup_end
    validation_end = validation_start + data['val_data'].shape[0]

    # Plot
    fig, axs = plt.subplots(D, 1, figsize=(9, 3 * D), sharex=True)
    axs = np.atleast_1d(axs)  # ensure it's iterable even if D = 1

    for i in range(D):
        ax = axs[i]
        ax.plot(dates, data_raw[:, i], color='lightgray', lw=2, label='Raw Data')
        ax.plot(dates[transient_start:transient_end], data['transient_data'][:, i], lw=2, label='transient data', color='orange')
        ax.plot(dates[train_start:train_end], data['train_data'][:, i], lw=2, label='training data', color='blue')
        ax.plot(dates[warmup_start:warmup_end], data['warmup_data'][:, i], lw=5, alpha=0.5, label='warm-up data', color='green')
        ax.plot(dates[validation_start:validation_end], data['val_data'][:, i], lw=2, label='validation data', color='red')

        # Boundaries
        for x in [dates[transient_start], dates[train_start], dates[train_end], dates[warmup_end]]:
            ax.axvline(x=x, color='gray', linestyle='--', lw=1)

        ax.set_ylabel(f'Variable {i+1}')
        #ax.legend(loc='lower left')

    axs[-1].set_xlabel('Time Step' if isinstance(dates[0], (int, np.integer)) else 'Date')
    fig.suptitle(f'Data preprocessing for {system_name}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig

