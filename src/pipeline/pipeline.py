# pipeline.py
import numpy as np
from typing import Dict
from preprocess import (
    preprocess_ecg, preprocess_ppg, preprocess_imu, preprocess_eda,
    detect_r_peaks, detect_ppg_peaks, estimate_ppg_foot, imu_activity_index
)
from features import (
    ibi_from_peaks, hr_from_ibi, sdnn, rmssd, lfhf_ratio_from_rr,
    ptt_from_pairs, rsa_power_from_rr, bandpower_stft, relative_bandpowers
)
from heuristics import classify_sleep
from fusion import assemble_feature_vector

class SleepStagePipeline:
    def __init__(self, fs_ecg=128, fs_ppg=64, fs_imu=32, fs_eda=32, win_sec=30):
        self.fs_ecg = fs_ecg
        self.fs_ppg = fs_ppg
        self.fs_imu = fs_imu
        self.fs_eda = fs_eda
        self.win_ecg = win_sec * fs_ecg
        self.win_ppg = win_sec * fs_ppg
        self.win_imu = win_sec * fs_imu
        self.win_eda = win_sec * fs_eda
        # PPG 상대대역 정의(호흡/저주파/고주파)
        self.ppg_bands = {"LF": (0.04, 0.15), "HF": (0.15, 0.40)}

    def process_window(self, ecg, ppg, imu_xyz, eda=None) -> Dict:
        # 1) 전처리
        ecg_f = preprocess_ecg(ecg, self.fs_ecg)
        ppg_f = preprocess_ppg(ppg, self.fs_ppg)
        imu_f = preprocess_imu(imu_xyz, self.fs_imu)

        # 2) 검출
        rpk = detect_r_peaks(ecg_f, self.fs_ecg)
        ppk = detect_ppg_peaks(ppg_f, self.fs_ppg)
        feet = estimate_ppg_foot(ppg_f, self.fs_ppg)

        # 3) ECG 특징
        ibi = ibi_from_peaks(rpk, self.fs_ecg)
        HR    = hr_from_ibi(ibi)
        SDNN  = sdnn(ibi)
        RMSSD = rmssd(ibi)
        LFHF  = lfhf_ratio_from_rr(ibi, fs_rr=4.0)
        RSA   = rsa_power_from_rr(ibi, fs_rr=4.0)

        # 4) PPG 특징
        PTT = ptt_from_pairs(rpk, feet, self.fs_ppg)
        rel = relative_bandpowers(ppg_f, self.fs_ppg, self.ppg_bands)
        PPG_LFrel = rel.get("LF", 0.0)
        PPG_HFrel = rel.get("HF", 0.0)

        # 5) IMU 특징
        ACT = imu_activity_index(imu_f, self.fs_imu, win_sec=int(len(imu_f)/self.fs_imu))

        # 6) EDA(옵션)
        # 필요 시 추가 지표 사용 가능 (여기서는 생략)

        # 7) 멀티모달 특징 결합
        feat = dict(HR=HR, SDNN=SDNN, RMSSD=RMSSD, LFHF=LFHF, RSA=RSA,
                    PTT=PTT, PPG_LFrel=PPG_LFrel, PPG_HFrel=PPG_HFrel,
                    ACT=ACT)

        # 8) 판별
        label, reason = classify_sleep(feat)
        vec, keys = assemble_feature_vector(feat)
        return {
            "Label": label,
            "Reason": reason,
            "Features": feat,
            "Vector": vec,
            "Keys": keys,
            "R_peaks": rpk, "PPG_peaks": ppk, "PPG_feet": feet
        }
