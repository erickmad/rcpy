import numpy as np
from rcpy.data import load_data_rcpy, preprocess_data_rcpy
import matplotlib.pyplot as plt

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
