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

def preprocess_data_rcpy(
    data,
    init_transient,
    transient_length,
    warmup_length,
    train_length,
    normalize=True,
):

    # Ensure 2D shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Remove initial transient
    data = data[init_transient:]
    T = data.shape[0]

    # Define split indices
    train_index = transient_length + train_length
    val_length = T - train_index - 1

    # Extract sections
    transient_data = data[:transient_length]
    train_data = data[transient_length:train_index]
    train_target = data[transient_length + 1 : train_index + 1]
    warmup_data = train_data[-warmup_length:]
    val_data = data[train_index:train_index + val_length]
    val_target = data[train_index + 1:train_index + val_length + 1]

    # Normalization (per feature)
    if normalize:
        train_min = np.min(train_data, axis=0)  # shape: (D,)
        train_max = np.max(train_data, axis=0)  # shape: (D,)
        scale = lambda x: 2 * (x - train_min) / (train_max - train_min) - 1

        transient_data = scale(transient_data)
        train_data = scale(train_data)
        train_target = scale(train_target)
        warmup_data = scale(warmup_data)
        val_data = scale(val_data)
        val_target = scale(val_target)
    else:
        train_min = None
        train_max = None

    return {
        "transient_data": transient_data,
        "train_data": train_data,
        "train_target": train_target,
        "warmup_data": warmup_data,
        "val_data": val_data,
        "val_target": val_target,
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

