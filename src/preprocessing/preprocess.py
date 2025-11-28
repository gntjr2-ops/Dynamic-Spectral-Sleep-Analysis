# preprocess.py
import numpy as np
from scipy.signal import find_peaks
from filters import bandpass, lowpass, highpass

# ---------- 전처리 ----------
def preprocess_ecg(ecg, fs):
    # QRS 대역 5~30 Hz
    return bandpass(ecg, fs, 5.0, 30.0, order=3)

def preprocess_ppg(ppg, fs):
    # 맥파 대역 0.5~8 Hz
    return bandpass(ppg, fs, 0.5, 8.0, order=3)

def preprocess_imu(acc, fs):
    if acc is None or len(acc) < 10:
        return acc
    # (N,3) → 채널별 필터링
    if acc.ndim == 2:
        acc_f = np.zeros_like(acc)
        for i in range(acc.shape[1]):
            acc_f[:, i] = highpass(acc[:, i], fs, 0.2, order=2)
        return acc_f
    else:
        return highpass(acc, fs, 0.2, order=2)

def preprocess_eda(eda, fs):
    # 저역통과 2 Hz (tonic 추출)
    return lowpass(eda, fs, 2.0, order=2)

# ---------- 검출 ----------
def detect_r_peaks(ecg_f, fs):
    # 간단 threshold + 최소 간격 0.3s
    distance = int(0.3 * fs)
    height = np.percentile(ecg_f, 85)
    peaks, _ = find_peaks(ecg_f, distance=distance, height=height)
    return peaks

def detect_ppg_peaks(ppg_f, fs):
    distance = int(0.4 * fs)
    height = np.percentile(ppg_f, 80)
    peaks, _ = find_peaks(ppg_f, distance=distance, height=height)
    return peaks

def estimate_ppg_foot(ppg_f, fs):
    # 매우 간단한 foot 근사: 1차 미분 최소점 근처
    d1 = np.gradient(ppg_f)
    inv = -d1
    distance = int(0.4 * fs)
    feet, _ = find_peaks(inv, distance=distance, height=np.percentile(inv, 80))
    return feet

def imu_activity_index(acc_f, fs, win_sec=30):
    # 가속도 벡터 norm → 이동 RMS(윈도우 전체)
    v = np.sqrt(np.sum(acc_f**2, axis=1)) if acc_f.ndim == 2 else np.abs(acc_f)
    return float(np.sqrt(np.mean(v**2)))
