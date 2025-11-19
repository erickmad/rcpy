import yaml
import numpy as np
import pandas as pd



from rcpy.forecasting import filter_forecasts



# ==================================
# 1. Read multi-forecasts
# ==================================
def read_multiforecasts(
    system,
    forecasts_file,
    cutoff=None,
    filter=False,
    max_abs_value=1e3,
    method="zscore",
    z_thresh=2.5, 
    iqr_k=1.5
):
    """
    Read multi-forecast results and optionally filter them.

    Parameters
    ----------
    system : str
        Name of the system (e.g. 'lorenz', 'ikeda', etc.).
    results_dir : Path
        Path to the directory containing the results.
    offset : int, optional
        Forecast offset identifier (default: 0).
    cutoff : int, optional
        Number of initial time steps to discard (default: None).
    filter : bool, optional
        Whether to apply forecast filtering (default: False).
    max_abs_value : float, optional
        Threshold for detecting exploding values.
    var_q_low, var_q_high : float, optional
        Quantile bounds for variance filtering.

    Returns
    -------
    If filter == False:
        forecasts_df : pd.DataFrame
            DataFrame of all forecasts.
    If filter == True:
        filtered_df : pd.DataFrame
            DataFrame of filtered forecasts.
        indices : list of int
            Indices of forecasts kept.
        variances : list of float
            Variance of each kept forecast.
    """

    # Load forecasts as DataFrame
    if cutoff is not None:
        df = pd.read_csv(forecasts_file, index_col=0).T.iloc[:cutoff]
    else:
        df = pd.read_csv(forecasts_file, index_col=0).T

    # Return early if no filtering requested
    if not filter:
        return df

    # Apply filtering
    forecasts = [df[col].values for col in df.columns]
    filtered, indices, scores = filter_forecasts(
        forecasts,
        max_abs_value=max_abs_value,
        method=method,
        z_thresh=z_thresh if method=='zscore' else None,
        iqr_k=iqr_k if method=='iqr' else None
    )

    if not filtered:
        print(f"[Warning] No forecasts passed the filter for {system} (offset={offset})")
        return pd.DataFrame(), [], []

    # Keep only the filtered columns
    filtered_cols = [df.columns[i] for i in indices]
    filtered_df = df[filtered_cols]

    return filtered_df
