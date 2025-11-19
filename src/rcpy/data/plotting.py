import numpy as np
import matplotlib.pyplot as plt

def plot_data_rcpy(data_raw, cutoff=None):
    """
    Plot raw data for one or more variables.
    
    Parameters
    ----------
    data_raw : ndarray
        Array of shape (T,) or (T, D).
    cutoff : int, optional
        Number of initial steps to discard before plotting.
    """
    # Ensure data is 2D
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape(-1, 1)
    num_vars = data_raw.shape[1]

    # Apply cutoff safely
    data_raw_cutoff = data_raw[:cutoff, :] if cutoff is not None else data_raw

    fig = plt.figure(figsize=(10, 4))

    # ---- 1 variable ----
    if num_vars == 1:
        ax_left = fig.add_subplot(121)
        ax_left.plot(data_raw_cutoff[:,0], label="Variable 1", color="tab:blue")
        ax_left.set_title('Time Series')
        ax_left.set_xlabel('Steps')
        ax_left.set_ylabel('Value')
        #ax_left.legend()

        ax_right = fig.add_subplot(122)
        ax_right.scatter(data_raw[:-1], data_raw[1:], s=1, color="gray")
        ax_right.set_title('Return Map')
        ax_right.set_xlabel('x(t)')
        ax_right.set_ylabel('x(t+1)')

    # ---- 2 variables ----
    elif num_vars == 2:
        ax_left1 = fig.add_subplot(321)
        ax_left1.plot(data_raw_cutoff[:, 0], color="tab:blue", label="x")
        ax_left1.set_title('Time Series')
        ax_left1.set_xlabel('Steps')
        ax_left1.set_ylabel('x')

        ax_left2 = fig.add_subplot(323)
        ax_left2.plot(data_raw_cutoff[:, 1], color="tab:orange", label="y")
        ax_left2.set_xlabel('Steps')
        ax_left2.set_ylabel('y')

        ax_right = fig.add_subplot(122)
        ax_right.scatter(data_raw[:, 0], data_raw[:, 1], s=1, color="tab:green")
        ax_right.set_title('x vs y')
        ax_right.set_xlabel('x')
        ax_right.set_ylabel('y')

    # ---- 3 variables ----
    elif num_vars == 3:
        ax_left1 = fig.add_subplot(321)
        ax_left1.plot(data_raw_cutoff[:, 0], color="tab:blue", label="x")
        ax_left1.set_title('Time Series')
        ax_left1.set_xlabel('Steps')
        ax_left1.set_ylabel('x')

        ax_left2 = fig.add_subplot(323)
        ax_left2.plot(data_raw_cutoff[:, 1], color="tab:orange", label="y")
        ax_left2.set_xlabel('Steps')
        ax_left2.set_ylabel('y')

        ax_left3 = fig.add_subplot(325)
        ax_left3.plot(data_raw_cutoff[:, 2], color="tab:green", label="z")
        ax_left3.set_xlabel('Steps')
        ax_left3.set_ylabel('z')

        ax_right = fig.add_subplot(122, projection='3d')
        ax_right.plot(data_raw[:, 0], data_raw[:, 1], data_raw[:, 2], lw=0.5, color="tab:purple")
        ax_right.set_title('Phase Space')
        ax_right.set_xlabel('x')
        ax_right.set_ylabel('y')
        ax_right.set_zlabel('z')

    for ax in fig.get_axes():
        ax.label_outer()
    plt.tight_layout()
    return fig


def plot_preprocessed_data_rcpy(
    data_raw,
    data,
    init_discard,
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
    
    # Compute time indices for each section
    transient_start = 0
    transient_end = init_discard
    train_start = transient_end
    train_end = train_start + len(data['train_data'])

    validation_start = train_end
    validation_end = validation_start + len(data['val_data'])

    test_start = validation_end
    test_end = test_start + len(data['val_data'])

    # Plot
    fig, axs = plt.subplots(D, 1, figsize=(9, 3 * D), sharex=True)
    axs = np.atleast_1d(axs)  # ensure it's iterable even if D = 1

    time = np.arange(T)
    for i in range(D):
        ax = axs[i]
        ax.plot(time[:test_end], data_raw[:test_end, i], color='lightgray', lw=2, label='Raw Data')
        ax.plot(time[train_start:train_end], data['train_data'][:, i], lw=2, label='training data', color='blue')
        ax.plot(time[validation_start:validation_end], data['val_data'][:, i], lw=2, label='validation data', color='orange')
        ax.plot(time[test_start:test_end], data['test_data'][:len(data['val_data']), i], lw=2, label='test data', color='red')

        # Boundaries
        for x in [time[transient_start], time[train_start], time[train_end], time[test_start]]:
            ax.axvline(x=x, color='gray', linestyle='--', lw=1)

        ax.set_ylabel(f'Variable {i+1}')
        #ax.set_xlim(dates[0], dates[:-100])
        ax.legend()

    axs[-1].set_xlabel('Time Steps')
    #fig.suptitle(f'Data preprocessing for {system_name}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.show()
    return fig