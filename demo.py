# demo.py
from synth import synth_window
from pipeline import SleepStagePipeline

def print_res(name, res):
    f = res["Features"]
    HR = f.get("HR"); SDNN=f.get("SDNN"); RMSSD=f.get("RMSSD")
    LFHF=f.get("LFHF"); RSA=f.get("RSA"); PTT=f.get("PTT")
    LF=f.get("PPG_LFrel"); HF=f.get("PPG_HFrel"); ACT=f.get("ACT")
    def s(v, fmt="{:.3f}"): return (fmt.format(v) if v is not None else "N/A")
    print(f"\n=== {name} ===")
    print(f"Label: {res['Label']} | Reason: {res['Reason']}")
    print(f"HR={s(HR, '{:.1f}')} BPM, SDNN={s(SDNN)} s, RMSSD={s(RMSSD)} s, LF/HF={s(LFHF, '{:.2f}')}, RSA={s(RSA)}")
    print(f"PTT={s(PTT)} s | ACT={s(ACT)}")

if __name__ == "__main__":
    pipe = SleepStagePipeline(fs_ecg=128, fs_ppg=64, fs_imu=32, win_sec=30)

    for stage in ["deep", "light", "rem", "wake"]:
        ecg, ppg, imu, params = synth_window(stage=stage, fs_ecg=128, fs_ppg=64, fs_imu=32, win_sec=60, seed=42)
        res = pipe.process_window(ecg, ppg, imu)
        print_res(stage, res)