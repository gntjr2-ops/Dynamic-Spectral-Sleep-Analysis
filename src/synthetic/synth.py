import numpy as np

# ------------------------------
# RR 시리즈 합성 (HR/HRV/LFHF 조절)
# ------------------------------
def synth_rr_series(hr_bpm, win_sec, sdnn_target, lf_amp=0.02, hf_amp=0.01,
                    lf_hz=0.08, hf_hz=0.25, seed=0):
    """
    RR(=IBI) 시계열 합성
    - 평균 RR = 60 / HR
    - 저주파/고주파 변조를 섞어 HRV(LF/HF) 특성을 제어
    """
    rng = np.random.default_rng(seed)
    n_beats = int(round(hr_bpm/60.0 * win_sec))
    base_rr = 60.0 / hr_bpm  # 초 단위

    t = np.linspace(0, win_sec, n_beats, endpoint=False)
    rr = (base_rr
          + lf_amp*np.sin(2*np.pi*lf_hz*t)
          + hf_amp*np.sin(2*np.pi*hf_hz*t)
          + rng.normal(scale=sdnn_target*0.25, size=n_beats))

    rr = np.clip(rr, 0.3, 2.0)  # 안전 범위
    return rr

# ------------------------------
# RR → R-peak 인덱스 변환
# ------------------------------
def rr_to_peaks(rr, fs):
    """
    RR 시리즈 누적합 → R-peak 인덱스(샘플 단위)
    """
    times = np.cumsum(rr)
    times -= times[0]
    idx = np.unique(np.maximum(0, np.round(times*fs).astype(int)))
    return idx

# ------------------------------
# 단계별 합성 함수
# ------------------------------
def synth_window(stage="light", fs_ecg=128, fs_ppg=64, fs_imu=32, win_sec=60, seed=42):
    """
    수면 단계별 합성 ECG/PPG/IMU 신호 + Ground Truth
    """
    rng = np.random.default_rng(seed)

    # stage 입력 표준화
    stage = stage.lower()

    if stage == "deep":
        hr_bpm, ptt_s = 55, 0.30
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.40,
                             lf_amp=0.06, hf_amp=0.03, seed=seed)
        act_level = 0.01

    elif stage == "light":
        hr_bpm, ptt_s = 75, 0.25
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.10,
                             lf_amp=0.03, hf_amp=0.02, seed=seed+1)
        act_level = 0.08

    elif stage == "rem":
        hr_bpm, ptt_s = 85, 0.22
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.07,
                             lf_amp=0.02, hf_amp=0.03, seed=seed+2)
        act_level = 0.05

    elif stage == "wake":
        hr_bpm, ptt_s = 100, 0.20
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.05,
                             lf_amp=0.01, hf_amp=0.02, seed=seed+3)
        act_level = 0.30

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # ----------------- ECG 합성 -----------------
    r_peaks = rr_to_peaks(rr, fs_ecg)
    N_ecg = fs_ecg * win_sec
    ecg = np.zeros(N_ecg, dtype=np.float32)

    width = int(0.02*fs_ecg)
    k = np.arange(-width, width+1)
    gauss = np.exp(-0.5*(k/(0.007*fs_ecg))**2)

    for rp in r_peaks:
        if rp - width < 0 or rp + width >= N_ecg: continue
        ecg[rp-width:rp+width+1] += gauss
    ecg += 0.005*rng.standard_normal(N_ecg)

    # ----------------- PPG 합성 -----------------
    delay = int(round(ptt_s * fs_ppg))
    ppg_feet = (r_peaks / fs_ecg * fs_ppg).astype(int) + delay

    N_ppg = fs_ppg * win_sec
    ppg = np.zeros(N_ppg, dtype=np.float32)
    tail = int(0.30*fs_ppg)
    for ft in ppg_feet:
        if ft >= N_ppg: continue
        end = min(N_ppg, ft+tail)
        idx = np.arange(end-ft)
        wave = np.exp(-idx/(0.08*fs_ppg))
        ppg[ft:end] += wave.astype(np.float32)
    ppg = (ppg - ppg.mean())/(ppg.std()+1e-8)
    ppg += 0.01*rng.standard_normal(N_ppg)

    # ----------------- IMU 합성 -----------------
    N_imu = fs_imu * win_sec
    imu = act_level * rng.standard_normal((N_imu, 3))

    # ----------------- Ground Truth Metrics -----------------
    params = {
        "HR": hr_bpm,
        "PTT": ptt_s,
        "ACT": act_level
    }

    return ecg, ppg, imu, params
