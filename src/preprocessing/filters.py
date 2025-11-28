# filters.py
import numpy as np
from scipy.signal import butter, filtfilt, welch, stft

def bandpass(x, fs, lo, hi, order=3):
    lo = max(lo, 1e-3)
    hi = min(hi, fs/2 - 1e-3)
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def lowpass(x, fs, fc, order=3):
    fc = min(fc, fs/2 - 1e-3)
    b, a = butter(order, fc/(fs/2), btype="low")
    return filtfilt(b, a, x)

def highpass(x, fs, fc, order=3):
    fc = max(fc, 1e-3)
    b, a = butter(order, fc/(fs/2), btype="high")
    return filtfilt(b, a, x)

def power_welch(x, fs, band=(0.1, 0.4), nperseg=None):
    if nperseg is None:
        nperseg = min(256, len(x))
    f, pxx = welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.trapz(pxx[m], f[m])) if np.any(m) else 0.0

def stft_bandpower(x, fs, band, win_sec=4.0, hop_ratio=0.5):
    nperseg = int(win_sec * fs)
    noverlap = int(nperseg * hop_ratio)
    f, t, Z = stft(x - np.mean(x), fs=fs, nperseg=nperseg, noverlap=noverlap)
    p = np.abs(Z)**2
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m): return 0.0
    bp = p[m, :].mean()
    return float(bp)

def zscore(x):
    x = np.asarray(x)
    s = np.std(x) + 1e-9
    return (x - np.mean(x)) / s
