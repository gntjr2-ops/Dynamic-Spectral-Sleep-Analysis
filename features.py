# features.py
import numpy as np
from scipy.signal import welch
from filters import power_welch, stft_bandpower

# ---------- ECG/PPG ----------
def ibi_from_peaks(peaks, fs):
    if peaks is None or len(peaks) < 2: return None
    return np.diff(peaks) / fs

def hr_from_ibi(ibi):
    if ibi is None or len(ibi) == 0: return None
    return 60.0 / float(np.mean(ibi))

def sdnn(ibi):
    if ibi is None or len(ibi) == 0: return None
    return float(np.std(ibi))

def rmssd(ibi):
    if ibi is None or len(ibi) < 2: return None
    return float(np.sqrt(np.mean(np.diff(ibi)**2)))

def lfhf_ratio_from_rr(ibi, fs_rr=4.0):
    # RR 보간 없이 간이 추정: 길이가 짧으면 None
    if ibi is None or len(ibi) < 16:  # 최소 비트 수 확보
        return None
    rr = ibi - np.mean(ibi)
    if np.allclose(rr, 0): return None
    f, pxx = welch(rr, fs=fs_rr, nperseg=min(256, len(rr)))
    lf = np.trapz(pxx[(f>=0.04) & (f<=0.15)], f[(f>=0.04) & (f<=0.15)])
    hf = np.trapz(pxx[(f>=0.15) & (f<=0.40)], f[(f>=0.15) & (f<=0.40)])
    if hf <= 0: return None
    return float(lf / hf)

def ptt_from_pairs(rpk, feet, fs):
    if rpk is None or feet is None or len(rpk)==0 or len(feet)==0: return None
    i=j=0
    diffs=[]
    while i < len(rpk) and j < len(feet):
        if feet[j] <= rpk[i]:
            j+=1; continue
        diffs.append((feet[j]-rpk[i])/fs)
        i+=1; j+=1
    return float(np.mean(diffs)) if diffs else None

# ---------- RSA / 호흡 추정(간이) ----------
def rsa_power_from_rr(ibi, fs_rr=4.0):
    # HF(0.15~0.4 Hz) 파워를 RSA proxy로 사용
    if ibi is None or len(ibi) < 16: return None
    rr = ibi - np.mean(ibi)
    f, pxx = welch(rr, fs=fs_rr, nperseg=min(256, len(rr)))
    hf = np.trapz(pxx[(f>=0.15) & (f<=0.40)], f[(f>=0.15) & (f<=0.40)])
    return float(hf)

# ---------- 스펙트럼-시간 특징 ----------
def bandpower_stft(x, fs, band, win_sec=4.0, hop_ratio=0.5):
    return stft_bandpower(x, fs, band, win_sec=win_sec, hop_ratio=hop_ratio)

def relative_bandpowers(x, fs, bands):
    total = power_welch(x, fs, (0.0, min(2.0, fs/2 - 1e-3)))
    out = {}
    for name, b in bands.items():
        bp = power_welch(x, fs, b)
        out[name] = float(bp / (total + 1e-9))
    return out
