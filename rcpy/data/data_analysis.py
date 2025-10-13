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
def plot_autocorrelation(series, max_lag=60):
    """
    Compute and plot autocorrelation function up to max_lag.
    """
    s = pd.Series(series).dropna()
    lags = np.arange(max_lag+1)
    ac = [s.autocorr(lag) for lag in lags]

    plt.figure(figsize=(6,4))
    plt.stem(lags, ac)
    plt.axhline(1/np.e, color='r', ls='--', label='1/e')
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function")
    plt.legend()
    plt.grid(True)
    plt.show()
    return pd.Series(ac, index=lags)


# =====================
# 3. Power spectrum (FFT)
# =====================
def plot_power_spectrum(series, fs=1.0):
    """
    Compute and plot power spectral density using a periodogram.
    fs: sampling frequency (1 for yearly, 12 for monthly data, etc.)
    """
    f, Pxx = signal.periodogram(series, fs=fs, window='hann', scaling='density')
    plt.figure(figsize=(6,4))
    plt.semilogy(f, Pxx)
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
