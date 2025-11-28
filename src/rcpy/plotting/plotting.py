import numpy as np
import matplotlib.pyplot as plt

from rcpy.data import load_data_rcpy, preprocess_data_rcpy
from rcpy.analysis import compute_skill


def plot_forecast(Y_pred, Y_true, efh, rmse_cumulative, corr_cumulative, plot_cutoff=None, title='Multi-step forecasting'):
    """
    Plot multi-step forecasts with RMSE and correlation skill metrics.

    Parameters
    ----------
    Y_pred : np.ndarray
        Predicted values from the model.
    Y_true : np.ndarray
        True values to compare predictions against.
    rmse_cumulative: np.ndarray
    corr_cumulative: np.ndarray
    plot_cutoff : int, optional
        Number of points to plot. If None, automatically determined from EFH + 100.
    title : str, optional
        Title of the top plot (default: 'Multi-step forecasting').
    """

    if plot_cutoff is None:
        plot_cutoff = int(efh + 100)
    
    # Create figure
    fig = plt.figure(figsize=(7,5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top plot: True vs Predicted
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(Y_true[:plot_cutoff], label='True')
    ax1.plot(Y_pred[:plot_cutoff], label='Predicted')
    ax1.axvline(efh, color='tab:red', linestyle='--', label='EFH')
    ax1.set_title(title)
    ax1.legend()
    
    # Bottom-left: RMSE
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rmse_cumulative[:plot_cutoff], color="tab:olive")
    ax2.axvline(efh, color='tab:red', linestyle='--', label='EFH')
    ax2.set_title(f'rmse: {rmse_cumulative[plot_cutoff]:.2f}')
    
    # Bottom-right: Correlation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(corr_cumulative[:plot_cutoff], color="tab:cyan")
    ax3.axvline(efh, color='tab:red', linestyle='--', label='EFH')
    ax3.set_title(f'corr: {corr_cumulative[plot_cutoff]:.2f}')
    
    plt.tight_layout()
    plt.show()


def plot_multiforecasts(true_data, forecasts, forecast_length=None):
    
    if hasattr(forecasts, "to_numpy"):
        forecasts = forecasts.to_numpy()
    mean_forecast = np.mean(forecasts, axis=0)

    if forecast_length is None:
        forecast_length = len(true_data)

    # Plot
    fig = plt.figure(figsize=(7, 3))
    plt.plot(true_data[:forecast_length], label="Real Data", color="tab:blue", linewidth=2)
    # Plot each forecast in gray
    for forecast in forecasts:
        plt.plot(forecast[:forecast_length], color='tab:gray', lw=0.5, alpha=0.1, zorder=0)
    plt.plot(mean_forecast[:forecast_length], label='Mean Forecast', color='tab:orange')

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    #plt.tight_layout()
    return fig
