# fusion.py
def assemble_feature_vector(fe):
    """
    fe: dict with keys (some may be None)
    - ECG: HR, SDNN, RMSSD, LFHF, RSA
    - PPG: PTT, PPG_LFrel, PPG_HFrel
    - IMU: ACT, POSTURE(optional)
    """
    keys = [
        "HR", "SDNN", "RMSSD", "LFHF", "RSA",
        "PTT", "PPG_LFrel", "PPG_HFrel",
        "ACT"
    ]
    vec = [fe.get(k, None) for k in keys]
    # NaN/None → 열 중앙값 대체는 파이프라인에서 처리 가능 (여긴 원형 유지)
    return vec, keys
