# heuristics.py
from typing import Dict, Tuple

# 수면 단계: Wake / REM / Light (N1-2) / Deep (N3)
RULES = {
    "wake": {   # 각성: HR↑, ACT↑, PPG 변동↑
        "hr_min": 80.0,
        "act_min": 0.15
    },
    "rem": {    # REM: HR 변동↑(RMSSD↓ or 불규칙), ACT는 낮거나 미세, RSA 낮음
        "hr_min": 65.0,
        "rmssd_max": 0.05,      # 심박의 고주파 변동(연속)은 낮게
        "act_max": 0.10,
    },
    "deep": {   # Deep: HR↓, HRV↑(SDNN↑, RSA↑), ACT 매우 낮음
        "hr_max": 65.0,
        "sdnn_min": 0.08,
        "act_max": 0.05
    },
    "light": {  # Light: 중간값, 나머지
    }
}

def _ok(val, cmp, thr):
    if val is None: return False
    return (val >= thr) if cmp == ">=" else (val <= thr)

def classify_sleep(fe: Dict) -> Tuple[str, str]:
    hr    = fe.get("HR")
    sdnn  = fe.get("SDNN")
    rmssd = fe.get("RMSSD")
    rsa   = fe.get("RSA")
    act   = fe.get("ACT")
    # ptt / ppg bands 등을 부가 근거로 활용 가능

    # 1) WAKE 우선
    w = RULES["wake"]
    if _ok(hr, ">=", w["hr_min"]) and _ok(act, ">=", w["act_min"]):
        return "Wake", f"HR={hr:.1f}≥{w['hr_min']}, ACT={act:.3f}≥{w['act_min']}"

    # 2) DEEP
    d = RULES["deep"]
    deep_ok = True
    deep_ok &= _ok(hr, "<=", d["hr_max"])
    deep_ok &= _ok(sdnn, ">=", d["sdnn_min"])
    deep_ok &= _ok(act, "<=", d["act_max"])
    if deep_ok:
        return "Deep", f"HR={hr:.1f}≤{d['hr_max']}, SDNN={sdnn:.3f}≥{d['sdnn_min']}, ACT={act:.3f}≤{d['act_max']}"

    # 3) REM
    r = RULES["rem"]
    rem_ok = True
    rem_ok &= _ok(hr, ">=", r["hr_min"])
    rem_ok &= _ok(rmssd, "<=", r["rmssd_max"])
    rem_ok &= _ok(act, "<=", r["act_max"])
    if rem_ok:
        return "REM", f"HR={hr:.1f}≥{r['hr_min']}, RMSSD={rmssd:.3f}≤{r['rmssd_max']}, ACT={act:.3f}≤{r['act_max']}"

    # 4) 나머지 LIGHT
    return "Light", "기타 조건 → Light"
