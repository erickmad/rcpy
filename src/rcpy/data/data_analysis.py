import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats

# ==========================================================
# 1. Time series: rolling mean, variance, skewness, kurtosis
# ==========================================================
def analyze_basic_stats(series, window=24, plot=True):
    """
    Compute rolling mean/variance and global skewness & kurtosis.
    Parameters
    ----------
    series : pandas Series or 1D array
    window : int, rolling window length (in samples)
    plot   : bool, whether to plot results
    """
    s = pd.Series(series).dropna()
    roll_mean = s.rolling(window).mean()
    roll_var = s.rolling(window).var()
    skewness = stats.skew(s)
    kurt = stats.kurtosis(s)

    print(f"Global mean = {s.mean():.3f}")
    print(f"Global variance = {s.var():.3f}")
    print(f"Skewness = {skewness:.3f}")
    print(f"Kurtosis = {kurt:.3f}")

    if plot:
        fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=False)
        ax[0].plot(s, label='Time series')
        ax[0].plot(roll_mean, label=f'Rolling mean ({window})', lw=2)
        ax[0].legend()
        ax[0].set_title("Time series and rolling mean")

        ax[1].plot(roll_var, color='orange')
        ax[1].set_title(f"Rolling variance ({window})")

        x = np.linspace(min(s), max(s), 1000)
        (mu, sigma) = stats.norm.fit(s)
        gaussian = stats.norm.pdf(x, loc=mu, scale=sigma)

        ax[2].hist(s, bins=30, density=True, color='tab:blue')
        ax[2].plot(x, gaussian, label='Gaussian', linestyle='--', color='tab:red')

        ax[2].set_xlim(-abs(s.max()+3), abs(s.max()+3))
        ax[2].set_title("Histogram")
        plt.tight_layout()
        plt.show()

    return {'mean': s.mean(), 'var': s.var(),
            'skewness': skewness, 'kurtosis': kurt}


# ==================
# 2. Autocorrelation
# ==================
def plot_autocorrelation(series, max_lag=60, scale=None):
    """
    Compute and plot autocorrelation function up to max_lag.
    """
    s = pd.Series(series).dropna()
    lags = np.arange(max_lag+1)
    ac = [s.autocorr(lag) for lag in lags]

    plt.figure(figsize=(6,4))
    plt.stem(lags, ac)
    if scale == 'semilog':
        plt.yscale('symlog', linthresh=1e-2)
    elif scale == 'loglog':
        plt.yscale('log')
        plt.xscale('log')
    plt.axhline(1/np.e, color='r', ls='--', label='1/e')
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function")
    plt.legend()
    plt.grid(True)
    plt.show()
    return pd.Series(ac, index=lags)


# =========================
# 2.5 Autocorrelation matrix
# =========================
def autocorr_matrix(data, cycle_length=12, max_lag=12):
    """
    Compute a cycle-based autocorrelation matrix for a 1D time series.
    
    Parameters
    ----------
    data : array-like
        The 1D time series data (e.g., NINO3.4 index values).
    cycle_length : int, optional (default=12)
        Number of time steps in one full cycle (e.g., 12 for months, 4 for quarters, etc.).
    max_lag : int, optional (default=12)
        Maximum lag to compute correlations for.
    
    Returns
    -------
    corr_matrix : ndarray
        A 2D array of shape (cycle_length, max_lag) containing the lag correlations.
        Rows correspond to the starting position within a cycle, and
        columns correspond to lags (1 to max_lag).
    """

    data = np.asarray(data).flatten()

    # Remove incomplete cycles at the end
    remainder = len(data) % cycle_length
    if remainder != 0:
        data = data[:-remainder]

    # Reshape into full cycles
    n_cycles = len(data) // cycle_length
    reshaped = data.reshape(n_cycles, cycle_length)

    corr_matrix = np.full((cycle_length, max_lag), np.nan)

    # Compute lag correlations
    for start_idx in range(cycle_length):
        for lag in range(1, max_lag + 1):
            x, y = [], []
            for c in range(n_cycles):
                if start_idx + lag < cycle_length:
                    # Same cycle
                    if c < n_cycles:
                        x_val = reshaped[c, start_idx]
                        y_val = reshaped[c, start_idx + lag]
                    else:
                        continue
                else:
                    # Wrap to next cycle
                    if c + 1 < n_cycles:
                        x_val = reshaped[c, start_idx]
                        y_val = reshaped[c + 1, (start_idx + lag) % cycle_length]
                    else:
                        continue
                x.append(x_val)
                y.append(y_val)
            if len(x) > 1:
                corr_matrix[start_idx, lag - 1] = np.corrcoef(x, y)[0, 1]

    return corr_matrix


# =====================
# 3. Power spectrum (FFT)
# =====================
def plot_power_spectrum(series, fs=1.0, scale=None):
    """
    Compute and plot power spectral density using a periodogram.
    fs: sampling frequency (1 for yearly, 12 for monthly data, etc.)
    """
    f, Pxx = signal.periodogram(series, fs=fs, window='hann', scaling='density')
    plt.figure(figsize=(6,4))
    plt.semilogy(f, Pxx)
    if scale == 'loglog':
        plt.xscale('log')
        plt.yscale('log')
    plt.title("Power Spectrum (Periodogram)")
    plt.xlabel("Frequency [cycles per unit time]")
    plt.ylabel("Power spectral density")
    plt.grid(True)
    plt.show()
    return f, Pxx


# ===============
# 4. Return map
# ===============
def plot_return_map(series, lag=1):
    """
    Plot return map (x_t vs x_{t+lag})
    """
    s = np.array(series)
    x = s[:-lag]
    y = s[lag:]

    plt.figure(figsize=(5,4))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel(f"x(t)")
    plt.ylabel(f"x(t+{lag})")
    plt.title(f"Return Map (lag={lag})")
    plt.grid(True)
    plt.show()


# ========================
# 5. Predictability metrics
# ========================
